"""Arena gate: candidate vs champion, GnnMcts both sides (spec §2 ARENA).

AGZ-style evaluation: 2 candidate + 2 champion seats, 4 rotations x
games/4 shared seeds, greedy search (no Dirichlet, no temperature — arena
measures strength, not exploration). Promotion needs strictly >
promote_threshold AND timeout_rate <= cap (the e5 lesson: wall-clock-
censored winrates are artifacts, so an over-cap arena is "invalid", never
"promote").

Resumable: one results.jsonl line per finished game; rerun skips seeds
already present.
"""
from __future__ import annotations

import asyncio
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

_BASE = ["cand", "champ", "cand", "champ"]


def seating_for_rotation(rot: int) -> list[str]:
    return _BASE[rot:] + _BASE[:rot]


def seed_plan(*, seed_base: int, games: int) -> list[tuple[int, int]]:
    """(rotation, seed) pairs: games/4 per rotation, distinct seed per game.

    The same seed appears once per run (not once per rotation) — board luck
    is balanced by the candidate sitting in every seat across rotations.
    """
    per_rot = games // 4
    plan = []
    for rot in range(4):
        for i in range(per_rot):
            plan.append((rot, seed_base + rot * 10_000 + i))
    return plan


@dataclass
class ArenaResult:
    wins_cand: int = 0
    wins_champ: int = 0
    draws: int = 0
    timeouts: int = 0
    per_rotation: list = field(default_factory=lambda: [0, 0, 0, 0])

    @property
    def games(self) -> int:
        return self.wins_cand + self.wins_champ + self.draws

    @property
    def winrate_cand(self) -> float:
        return self.wins_cand / self.games if self.games else 0.0

    @property
    def timeout_rate(self) -> float:
        total = self.games + self.timeouts
        return self.timeouts / total if total else 0.0

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "ArenaResult":
        return cls(**json.loads(Path(path).read_text()))


def should_promote(result: ArenaResult, cfg) -> str:
    """'promote' | 'hold' | 'invalid' (timeout-censored, not trustworthy)."""
    if result.timeout_rate > cfg.arena_timeout_rate_max:
        return "invalid"
    if result.winrate_cand > cfg.promote_threshold:
        return "promote"
    return "hold"


# --------------------------------------------------------------------------
# Async game driver (GPU path; exercised by the integration test).
# --------------------------------------------------------------------------

async def _play_arena_game(*, game, seed: int, seating: list[str],
                           mcts_cand, mcts_champ, sims: int) -> tuple[int, bool]:
    """Returns (winner_seat or -1, timed_out). Mirrors the e10g driver:
    chance fast-path, single-legal fast-path, step cap."""
    state = game.new_initial_state(seed=seed)
    chance_rng = random.Random(seed)
    steps, max_steps = 0, 200_000
    while not state.is_terminal() and steps < max_steps:
        if state.is_chance_node():
            outs = state.chance_outcomes()
            r = chance_rng.random()
            cum, chosen = 0.0, outs[-1][0]
            for v, p in outs:
                cum += p
                if r <= cum:
                    chosen = v
                    break
            state.apply_action(int(chosen))
            steps += 1
            continue
        legal = state.legal_actions()
        if len(legal) == 1:
            state.apply_action(int(legal[0]))
            steps += 1
            continue
        mcts = mcts_cand if seating[state.current_player()] == "cand" else mcts_champ
        visit_counts = await mcts.search(state, n_sims=sims)
        state.apply_action(int(mcts.best_action(visit_counts)))
        steps += 1
    if not state.is_terminal():
        return -1, True
    rs = state.returns()
    return (rs.index(1.0) if 1.0 in rs else -1), False


def _load_model(ckpt: Path, cfg, device: str):
    import torch
    from catan_gnn.gnn_model import GnnModel
    model = GnnModel(hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers)
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)
    return model.to(device).eval()


def run_arena(*, candidate_ckpt: Path, champion_ckpt: Path, cfg,
              out_dir: Path, seed_base: int = 30_000_000,
              device: str = "cuda", n_concurrent: int = 16) -> ArenaResult:
    """Run (or resume) the full arena; returns the aggregated result."""
    from catan_mcts.adapter import CatanGame
    from catan_mcts.async_mcts import AsyncMcts
    from catan_mcts.batched_evaluator import BatchedGnnEvaluator

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done: dict[int, dict] = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["seed"]] = rec

    async def _main() -> None:
        model_cand = _load_model(candidate_ckpt, cfg, device)
        model_champ = _load_model(champion_ckpt, cfg, device)
        ev_cand = BatchedGnnEvaluator(model=model_cand, device=device,
                                      max_batch=cfg.max_batch, window_ms=5.0,
                                      watchdog_windows=10)
        ev_champ = BatchedGnnEvaluator(model=model_champ, device=device,
                                       max_batch=cfg.max_batch, window_ms=5.0,
                                       watchdog_windows=10)
        ev_cand.start()
        ev_champ.start()
        game = CatanGame(vp_target=cfg.vp_target, bonuses=cfg.bonuses)
        sem = asyncio.Semaphore(n_concurrent)
        f = open(results_path, "a")
        # Live-game counter shared by both evaluators. Without it, each
        # evaluator keeps its 10**9 default, so the all-parked flush clause
        # (n >= active_game_count) NEVER fires — batches degrade to the 5ms
        # window only, throughput collapses, and the watchdog spams false
        # "stuck game" warnings every window. Each live game alternates leaf
        # requests between the two evaluators, so the right per-evaluator
        # bound is the live game count (mirrors self_play_async's `active`).
        active = {"n": 0}

        def _set_active(n: int) -> None:
            ev_cand.active_game_count = max(1, n)
            ev_champ.active_game_count = max(1, n)

        async def one(rot: int, seed: int) -> None:
            async with sem:
                active["n"] += 1
                _set_active(active["n"])
                try:
                    seating = seating_for_rotation(rot)
                    mcts_c = AsyncMcts(evaluator=ev_cand, c=1.4,
                                       rng=np.random.default_rng(seed + 11))
                    mcts_x = AsyncMcts(evaluator=ev_champ, c=1.4,
                                       rng=np.random.default_rng(seed + 13))
                    winner, timed_out = await _play_arena_game(
                        game=game, seed=seed, seating=seating,
                        mcts_cand=mcts_c, mcts_champ=mcts_x, sims=cfg.arena_sims)
                    rec = {"seed": seed, "rot": rot, "winner_seat": winner,
                           "winner_role": (seating[winner] if winner >= 0 else None),
                           "timed_out": timed_out}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    done[seed] = rec
                finally:
                    active["n"] -= 1
                    _set_active(active["n"])

        plan = [(rot, seed) for rot, seed in
                seed_plan(seed_base=seed_base, games=cfg.arena_games)
                if seed not in done]
        await asyncio.gather(*(one(rot, seed) for rot, seed in plan),
                             return_exceptions=False)
        await ev_cand.stop()
        await ev_champ.stop()
        f.close()

    if any(seed not in done for _, seed in
           seed_plan(seed_base=seed_base, games=cfg.arena_games)):
        asyncio.run(_main())

    result = ArenaResult()
    for rec in done.values():
        if rec["timed_out"]:
            result.timeouts += 1
        elif rec["winner_role"] == "cand":
            result.wins_cand += 1
            result.per_rotation[rec["rot"]] += 1
        elif rec["winner_role"] == "champ":
            result.wins_champ += 1
        else:
            result.draws += 1
    result.to_json(out_dir / "arena.json")
    return result
