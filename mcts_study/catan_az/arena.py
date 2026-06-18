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


def _read_arena_results(results_path: Path) -> dict[int, dict]:
    """Resume map seed -> result, tolerant of a torn final line.

    M3 (deep-inspect HIGH): a crash mid-write leaves a half-written final line in
    results.jsonl. A bare json.loads would raise JSONDecodeError and poison-pill
    the iteration forever (it never returns, ARENA never marks done, no human is
    present). Skip un-parseable lines, mirroring analytics.py's reader of the
    same file."""
    done: dict[int, dict] = {}
    if not Path(results_path).exists():
        return done
    for line in Path(results_path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue                      # torn/partial line from a crash
        done[rec["seed"]] = rec
    return done


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
        # Every game lands in exactly one of wins_cand / wins_champ / draws —
        # timed-out games are now credited to their VP-leader (a win) or, if
        # tied, a draw. So this sums to the full game count. (timeouts is a
        # parallel tally for observability, NOT a 4th bucket.)
        return self.wins_cand + self.wins_champ + self.draws

    @property
    def decisive(self) -> int:
        return self.wins_cand + self.wins_champ

    @property
    def winrate_cand(self) -> float:
        # Winrate over DECISIVE games (a VP-tie draw is no signal either way).
        return self.wins_cand / self.decisive if self.decisive else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games else 0.0

    @property
    def timeout_rate(self) -> float:
        # Fraction of games that went to the VP-leader tiebreak (stalled past
        # the wall-clock cap). High is expected for closely-matched nets and
        # is NO LONGER a validity problem (those games are now decided), but
        # we still surface it.
        return self.timeouts / self.games if self.games else 0.0

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "ArenaResult":
        return cls(**json.loads(Path(path).read_text()))


def should_promote(result: ArenaResult, cfg) -> str:
    """'promote' | 'hold' | 'invalid'.

    With VP-leader tiebreak, timed-out games ARE decisive, so the validity
    guard now keys on DRAWS (VP ties = genuine no-signal games) and on having
    enough decisive games to trust the winrate — not on the timeout rate.
    Promote iff the candidate's winrate over decisive games clears the bar.
    """
    if result.games == 0:
        return "invalid"
    if result.draw_rate > cfg.arena_max_draw_rate:
        return "invalid"
    if result.decisive < cfg.arena_min_decisive:
        return "invalid"
    if result.winrate_cand > cfg.promote_threshold:
        return "promote"
    return "hold"


# --------------------------------------------------------------------------
# Async game driver (GPU path; exercised by the integration test).
# --------------------------------------------------------------------------

async def _play_arena_game(*, game, seed: int, seating: list[str],
                           mcts_cand, mcts_champ, sims: int,
                           max_seconds: float | None = None) -> tuple[int, bool, int]:
    """Returns (winner_seat or -1, timed_out). Mirrors the e10g driver:
    chance fast-path, single-legal fast-path, step cap. A per-game
    wall-clock cap (max_seconds) bounds the rare pathological game that
    crawls toward the 200k step cap — without it, one slow game holds the
    whole arena's asyncio.gather hostage (2026-06-13: seed 30030029 ran
    45min+ alone while the 119-game verdict sat decided)."""
    import time as _t
    state = game.new_initial_state(seed=seed)
    chance_rng = random.Random(seed)
    steps, max_steps = 0, 200_000
    deadline = (_t.monotonic() + max_seconds) if max_seconds else None
    while not state.is_terminal() and steps < max_steps:
        if deadline is not None and _t.monotonic() > deadline:
            # wall-clock timeout -> VP tiebreak (winner seat + the VP margin)
            return _vp_leader_margin(state), True, _vp_margin(state)
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
        # Step-cap exit also uses the VP tiebreak.
        return _vp_leader_margin(state), True, _vp_margin(state)
    rs = state.returns()
    winner = rs.index(1.0) if 1.0 in rs else -1
    return winner, False, _vp_margin(state)


def _vp_leader_margin(state) -> int:
    """VP-leader tiebreak winner SEAT (spec 2026-06-13 §9). NOTE: despite the
    legacy name this returns a seat index, not a margin — `_vp_margin` returns
    the margin. Kept for the call sites that only need the winner.

    For timed-out games: top public VP wins; a VP tie is broken by a second
    signal (settlements+cities built), then -1 (draw). The margin fallback cuts
    the draw rate as champion and candidate converge — without it, near-identical
    nets produce mostly ties and the gate stalls on the draw-rate guard.
    """
    vps = [int(state._engine.vp(i)) for i in range(4)]
    top = max(vps)
    leaders = [i for i, v in enumerate(vps) if v == top]
    if len(leaders) == 1:
        return leaders[0]
    stats = state._engine.stats()["players"]

    def builds(i):
        return stats[i]["settlements_built"] + stats[i]["cities_built"]

    best = max(builds(i) for i in leaders)
    tied = [i for i in leaders if builds(i) == best]
    return tied[0] if len(tied) == 1 else -1


def _vp_margin(state) -> int:
    """VP gap between the leader and the runner-up (top - second).

    The only skill-vs-luck signal in a 100%-timeout arena: a 1-VP squeaker is
    near-noise, a 4-VP win is dominance (Opus critique 2026-06-15). 0 on a VP
    tie (the game was decided by the builds tiebreak or is a draw)."""
    vps = sorted((int(state._engine.vp(i)) for i in range(4)), reverse=True)
    return vps[0] - vps[1]


def _vp_leader(state) -> int:
    """Back-compat alias; delegates to the margin-aware tiebreak."""
    return _vp_leader_margin(state)


def _load_model(ckpt: Path, cfg, device: str):
    import torch
    from catan_gnn.gnn_model import GnnModel
    model = GnnModel(hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers)
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)
    return model.to(device).eval()


def _aggregate_arena(done: dict[int, dict], out_dir: Path) -> ArenaResult:
    """Shared aggregation of results.jsonl records into an ArenaResult."""
    result = ArenaResult()
    for rec in done.values():
        if rec["timed_out"]:
            result.timeouts += 1
        if rec["winner_role"] == "cand":
            result.wins_cand += 1
            result.per_rotation[rec["rot"]] += 1
        elif rec["winner_role"] == "champ":
            result.wins_champ += 1
        else:
            result.draws += 1
    result.to_json(out_dir / "arena.json")
    return result


def _run_arena_rust(*, candidate_ckpt: Path, champion_ckpt: Path, cfg,
                    out_dir: Path, seed_base: int, results_path: Path,
                    done: dict[int, dict]) -> ArenaResult:
    """Rust-engine arena: catan_mcts_rs.run_arena over the full plan, writing
    the SAME results.jsonl schema, then shared aggregation. Resumable: seeds
    already in results.jsonl are skipped (the Rust call replays the full plan;
    we only append/keep records for seeds not yet done)."""
    import time as _t
    import catan_mcts_rs
    from catan_gnn.export_torchscript import export

    def _ts(ckpt: Path) -> str:
        ckpt = Path(ckpt)
        ts = ckpt.with_suffix(".ts")
        if not ts.exists():
            export(checkpoint=ckpt, out_ts=ts,
                   hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers)
        return str(ts)

    plan_seeds = {s for _, s in seed_plan(seed_base=seed_base, games=cfg.arena_games)}
    if all(s in done for s in plan_seeds):
        return _aggregate_arena(done, out_dir)

    records = catan_mcts_rs.run_arena(
        _ts(candidate_ckpt), _ts(champion_ckpt), seed_base, cfg.arena_games,
        cfg.arena_sims, cfg.vp_target, cfg.bonuses)
    with open(results_path, "a") as f:
        for rec in records:
            if rec["seed"] in done:
                continue
            rec = {**rec, "ts": _t.time()}
            f.write(json.dumps(rec) + "\n")
            done[rec["seed"]] = rec
    return _aggregate_arena(done, out_dir)


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
    done = _read_arena_results(results_path)

    # Rust engine path (catan_mcts_rs): in-process MCTS + TorchScript GNN, zero
    # per-node PyO3. Games finish naturally (no wall-clock deadline). Writes the
    # SAME results.jsonl schema, then falls through to the shared aggregation.
    if getattr(cfg, "engine", "python") == "rust":
        return _run_arena_rust(
            candidate_ckpt=candidate_ckpt, champion_ckpt=champion_ckpt, cfg=cfg,
            out_dir=out_dir, seed_base=seed_base, results_path=results_path,
            done=done)

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
                    winner, timed_out, vp_margin = await _play_arena_game(
                        game=game, seed=seed, seating=seating,
                        mcts_cand=mcts_c, mcts_champ=mcts_x, sims=cfg.arena_sims,
                        max_seconds=getattr(cfg, "arena_game_max_seconds", None))
                    import time as _tt
                    rec = {"seed": seed, "rot": rot, "winner_seat": winner,
                           "winner_role": (seating[winner] if winner >= 0 else None),
                           "timed_out": timed_out, "vp_margin": vp_margin,
                           "ts": _tt.time()}   # wall-clock finish (live rate/ETA)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    done[seed] = rec
                finally:
                    active["n"] -= 1
                    _set_active(active["n"])

        plan = [(rot, seed) for rot, seed in
                seed_plan(seed_base=seed_base, games=cfg.arena_games)
                if seed not in done]
        # M3 (deep-inspect): return_exceptions=True so one game's deterministic
        # engine/MCTS exception can't abort the whole arena (losing all in-flight
        # GPU work) and — worse — poison-pill the iteration on every restart
        # (the failing seed is never written to `done`, so it's replanned
        # forever with no human present). Log failures; completed games are
        # already durable in results.jsonl.
        outcomes = await asyncio.gather(
            *(one(rot, seed) for rot, seed in plan), return_exceptions=True)
        failures = [(plan[i], o) for i, o in enumerate(outcomes)
                    if isinstance(o, BaseException)]
        for (rot, seed), exc in failures:
            print(f"[arena] game rot={rot} seed={seed} FAILED: "
                  f"{type(exc).__name__}: {exc}")
        if failures:
            print(f"[arena] WARNING: {len(failures)}/{len(plan)} games errored "
                  f"(excluded from the verdict; not retried)")
        await ev_cand.stop()
        await ev_champ.stop()
        f.close()

    if any(seed not in done for _, seed in
           seed_plan(seed_base=seed_base, games=cfg.arena_games)):
        asyncio.run(_main())

    # A timed-out game is credited to its VP-leader (winner_role set); only a
    # VP-tie timeout (winner_role None) is a draw. Shared with the Rust path.
    return _aggregate_arena(done, out_dir)
