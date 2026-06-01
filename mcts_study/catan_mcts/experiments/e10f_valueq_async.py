"""Experiment 10f-async: ValueQ (1-ply value-Q) vs raw-argmax PureGnn vs LookV3.

The gate test for the diagnosed PureGnn plateau fix
(docs/superpowers/journals/2026-06-01-puregnn-plateau-diagnosis.md). All three
GNN seats load the SAME checkpoint (Cell6 by default); the only difference is the
DEPLOYMENT:
  - Slot 0: ValueQA       — ValueQGnnBot (1-ply value-Q over the value head)
  - Slot 1: RawPureGnnB   — PureGnnBot (argmax of policy head)  [the baseline]
  - Slot 2: RawPureGnnC   — PureGnnBot (argmax of policy head)  [filler 3rd GNN]
  - Slot 3: LookaheadMctsV3 — synchronous heuristic MCTS (the target)

If ValueQA beats RawPureGnn (and ideally LookV3), the diagnosis is confirmed AND
we have a shippable search-free fix. All GNN evals (ValueQ's children) batch
across concurrent games via ONE shared BatchedGnnEvaluator.

Mirrors e10e_async's rotation/recording/concurrency design exactly.
"""
from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path

import numpy as np
import torch

from catan_gnn.gnn_model import GnnModel

from ..adapter import CatanGame
from ..batched_evaluator import BatchedGnnEvaluator
from ..bots_gnn import PureGnnBot
from ..value_q_bot import ValueQGnnBot
from ..players_v3 import build_lookahead_mcts_v3
from ..recorder import SelfPlayRecorder
from .common import make_run_dir


_BASE_SEATING = ["ValueQA", "RawPureGnnB", "RawPureGnnC", "LookaheadMctsV3"]


def _resolve_device(spec: str) -> str:
    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


def _load_model(checkpoint: Path, hidden_dim: int, num_layers: int,
                device: str = "cpu") -> GnnModel:
    model = GnnModel(hidden_dim=hidden_dim, num_layers=num_layers)
    obj = torch.load(checkpoint, map_location=device, weights_only=False)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


async def _play_one_tournament_game(
    *,
    game,
    seed: int,
    seating: list[str],
    model_raw: GnnModel,
    lookahead_bot,
    evaluator: BatchedGnnEvaluator,
    device: str,
):
    """Play one rotation-game. winner is the absolute SEAT index (0-3), or -1."""
    state = game.new_initial_state(seed=seed)
    chance_rng = random.Random(seed)

    value_q = ValueQGnnBot(evaluator=evaluator)        # async seat
    pure_raw = PureGnnBot(model=model_raw, device=device)  # sync seat (shared)

    steps = 0
    max_steps = 200_000

    while not state.is_terminal() and steps < max_steps:
        if state.is_chance_node():
            outs = state.chance_outcomes()
            r = chance_rng.random()
            cum = 0.0
            chosen = outs[-1][0]
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

        cp = state.current_player()
        role = seating[cp]

        if role == "ValueQA":
            action = await value_q.step(state)
        elif role in ("RawPureGnnB", "RawPureGnnC"):
            action = pure_raw.step(state)
        elif role == "LookaheadMctsV3":
            action = lookahead_bot.step(state)
        else:
            raise RuntimeError(f"unknown role {role}")

        state.apply_action(int(action))
        steps += 1

    timed_out = not state.is_terminal()
    winner = -1
    final_vp = [0, 0, 0, 0]
    if state.is_terminal():
        rs = state.returns()
        if 1.0 in rs:
            winner = rs.index(1.0)
        try:
            stats = state._engine.stats()
            final_vp = [int(stats["players"][i]["vp_final"]) for i in range(4)]
        except Exception:
            pass

    action_history = list(state._engine.action_history())
    return winner, final_vp, steps, action_history, timed_out


async def _play_and_record(
    *,
    game,
    seed: int,
    rot_idx: int,
    model_raw: GnnModel,
    lookahead_depth: int,
    base_sims_v3: int,
    evaluator: BatchedGnnEvaluator,
    device: str,
    rec: SelfPlayRecorder,
    sem: asyncio.Semaphore,
    active: dict,
):
    async with sem:
        active["n"] += 1
        evaluator.active_game_count = active["n"]
        try:
            seating = _BASE_SEATING[rot_idx:] + _BASE_SEATING[:rot_idx]
            lookahead_bot = build_lookahead_mcts_v3(
                game, lookahead_depth=lookahead_depth, seed=seed,
                base_sims=base_sims_v3,
            )
            winner, final_vp, length, action_history, timed_out = (
                await _play_one_tournament_game(
                    game=game, seed=seed, seating=seating,
                    model_raw=model_raw,
                    lookahead_bot=lookahead_bot,
                    evaluator=evaluator,
                    device=device,
                )
            )
            with rec.game(seed=seed) as g_rec:
                if timed_out:
                    rec.skip_game(
                        seed=seed,
                        reason="max-steps-timeout",
                        length_in_moves=length,
                        action_history=action_history,
                        moves_recorder=g_rec,
                    )
                else:
                    g_rec.finalize(
                        winner=winner,
                        final_vp=final_vp,
                        length_in_moves=length,
                        action_history=action_history,
                    )
                    rec.mark_done(seed)
        finally:
            active["n"] -= 1
            evaluator.active_game_count = max(1, active["n"])


async def _run_async(
    *,
    out: Path,
    checkpoint: Path,
    num_games_per_seating: int,
    lookahead_depth: int,
    base_sims_v3: int,
    hidden_dim: int,
    num_layers: int,
    seed_base: int,
    vp_target: int,
    bonuses: bool,
    device: str,
    max_batch: int,
    window_ms: float,
    n_concurrent: int,
    resume: bool,
):
    model = _load_model(checkpoint, hidden_dim, num_layers, device)
    evaluator = BatchedGnnEvaluator(
        model=model, device=device,
        max_batch=max_batch, window_ms=window_ms,
        watchdog_windows=10,
    )
    evaluator.start()

    rec = SelfPlayRecorder(out, config={
        "experiment": "e10f_valueq_async",
        "checkpoint": str(checkpoint),
        "seating": _BASE_SEATING,
        "lookahead_depth": lookahead_depth,
        "base_sims_v3": base_sims_v3,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_games_per_seating": num_games_per_seating,
        "vp_target": vp_target,
        "bonuses": bonuses,
        "max_batch": max_batch,
        "window_ms": window_ms,
        "n_concurrent": n_concurrent,
        "seed_base": seed_base,
    })

    done = rec.done_seeds() if resume else set()
    game = CatanGame(vp_target=vp_target, bonuses=bonuses)
    sem = asyncio.Semaphore(n_concurrent)
    active = {"n": 0}

    tasks = []
    seeds_flat = []
    for rot_idx in range(4):
        for i in range(num_games_per_seating):
            seed = seed_base + rot_idx * 10_000 + i
            if seed in done:
                continue
            seeds_flat.append(seed)
            tasks.append(_play_and_record(
                game=game, seed=seed, rot_idx=rot_idx,
                model_raw=model,
                lookahead_depth=lookahead_depth,
                base_sims_v3=base_sims_v3,
                evaluator=evaluator,
                device=device,
                rec=rec, sem=sem, active=active,
            ))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    failures = [(seeds_flat[i], r) for i, r in enumerate(results)
                if isinstance(r, BaseException)]
    for seed, exc in failures:
        print(f"[e10f] game seed={seed} FAILED: {type(exc).__name__}: {exc}")

    completed = len(tasks) - len(failures)
    if failures:
        print(f"[e10f] WARNING: {len(failures)}/{len(tasks)} games failed")
    print(f"[e10f] done: {completed}/{len(tasks)} games, "
          f"mean_batch={evaluator.mean_batch_size():.1f}, "
          f"total_batches={evaluator.total_batches}")

    await evaluator.stop()
    rec.checkpoint(rec.config_id[:8])


def run_e10f_valueq_async(
    *,
    out_root: Path,
    checkpoint: Path,
    num_games_per_seating: int = 30,
    lookahead_depth: int = 10,
    base_sims_v3: int = 200,
    hidden_dim: int = 128,
    num_layers: int = 4,
    seed_base: int = 20_000_000,
    vp_target: int = 10,
    bonuses: bool = True,
    device: str = "cpu",
    max_batch: int = 32,
    window_ms: float = 5.0,
    n_concurrent: int = 32,
    resume_dir: Path | None = None,
) -> Path:
    out = resume_dir if resume_dir is not None else make_run_dir(out_root, "e10f_valueq_async")
    asyncio.run(_run_async(
        out=out, checkpoint=checkpoint,
        num_games_per_seating=num_games_per_seating,
        lookahead_depth=lookahead_depth, base_sims_v3=base_sims_v3,
        hidden_dim=hidden_dim, num_layers=num_layers,
        seed_base=seed_base, vp_target=vp_target, bonuses=bonuses,
        device=device, max_batch=max_batch, window_ms=window_ms,
        n_concurrent=n_concurrent, resume=resume_dir is not None,
    ))
    return out


def cli_main():
    p = argparse.ArgumentParser(description="e10f: ValueQ vs raw-PureGnn vs LookV3")
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="single GNN checkpoint used by all three GNN seats")
    p.add_argument("--num-games-per-seating", type=int, default=30,
                   help="total = this x 4 rotations")
    p.add_argument("--lookahead-depth", type=int, default=10)
    p.add_argument("--base-sims-v3", type=int, default=200)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--seed-base", type=int, default=20_000_000)
    p.add_argument("--vp-target", type=int, default=10)
    p.add_argument("--bonuses", action="store_true", default=True)
    p.add_argument("--no-bonuses", dest="bonuses", action="store_false")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--window-ms", type=float, default=5.0)
    p.add_argument("--n-concurrent", type=int, default=32)
    p.add_argument("--resume-dir", type=Path, default=None)
    args = p.parse_args()
    device = _resolve_device(args.device)
    out = run_e10f_valueq_async(
        out_root=args.out_root, checkpoint=args.checkpoint,
        num_games_per_seating=args.num_games_per_seating,
        lookahead_depth=args.lookahead_depth, base_sims_v3=args.base_sims_v3,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        seed_base=args.seed_base, vp_target=args.vp_target, bonuses=args.bonuses,
        device=device, max_batch=args.max_batch, window_ms=args.window_ms,
        n_concurrent=args.n_concurrent, resume_dir=args.resume_dir,
    )
    print(f"e10f_valueq_async wrote to {out}")


if __name__ == "__main__":
    cli_main()
