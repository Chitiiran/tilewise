# catan_mcts/experiments/self_play_async.py
"""Single-process asyncio self-play data generator.

Runs N game-coroutines concurrently against one BatchedGnnEvaluator, writes
e9-schema parquets via SelfPlayRecorder, resumable via done.txt. See spec
2026-05-30-batched-gnn-evaluator.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import torch

from catan_gnn.gnn_model import GnnModel
from ..adapter import CatanGame
from ..batched_evaluator import BatchedGnnEvaluator
from ..async_mcts import play_one_async_game
from ..recorder import SelfPlayRecorder
from .common import make_run_dir


def _load_model(checkpoint: Path, hidden_dim: int, num_layers: int, device: str):
    model = GnnModel(hidden_dim=hidden_dim, num_layers=num_layers)
    obj = torch.load(checkpoint, map_location=device, weights_only=False)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)
    return model.to(device).eval()


async def _play_and_record(*, game, seed, evaluator, n_sims, rec, sem, active):
    async with sem:
        active["n"] += 1
        evaluator.active_game_count = active["n"]
        try:
            result = await play_one_async_game(
                game=game, seed=seed, evaluator=evaluator, n_sims=n_sims,
                rng=np.random.default_rng(seed))
            with rec.game(seed=seed) as g_rec:
                for m in result.moves:
                    g_rec.record_move(
                        current_player=m.current_player, move_index=m.move_index,
                        legal_action_mask=m.legal_mask,
                        mcts_visit_counts=m.visit_counts,
                        action_taken=m.action_taken, mcts_root_value=m.root_value)
                g_rec.finalize(winner=result.winner, final_vp=result.final_vp,
                               length_in_moves=result.length_in_moves,
                               action_history=result.action_history,
                               timed_out=not result.terminal)
            rec.mark_done(seed)
        finally:
            active["n"] -= 1
            evaluator.active_game_count = max(1, active["n"])


async def _run_async(*, out, checkpoint, num_games, n_sims, n_concurrent,
                     hidden_dim, num_layers, vp_target, bonuses, device,
                     max_batch, window_ms, seed_base, resume,
                     ram_budget_mb, per_game_mb):
    if ram_budget_mb is not None:
        cap = max(1, int(ram_budget_mb / per_game_mb))
        if cap < n_concurrent:
            print(f"[self_play] concurrency capped: {n_concurrent} -> {cap} "
                  f"(ram_budget={ram_budget_mb}MB / {per_game_mb}MB per game)")
            n_concurrent = cap
    model = _load_model(checkpoint, hidden_dim, num_layers, device)
    evaluator = BatchedGnnEvaluator(model=model, device=device,
                                    max_batch=max_batch, window_ms=window_ms)
    evaluator.start()
    rec = SelfPlayRecorder(out, config={
        "experiment": "self_play_async", "n_sims": n_sims,
        "n_concurrent": n_concurrent, "vp_target": vp_target, "bonuses": bonuses,
        "max_batch": max_batch, "window_ms": window_ms, "device": device,
        "seed_base": seed_base})
    done = rec.done_seeds() if resume else set()
    sem = asyncio.Semaphore(n_concurrent)
    active = {"n": 0}
    game = CatanGame(vp_target=vp_target, bonuses=bonuses)
    seeds = [seed_base + i for i in range(num_games) if (seed_base + i) not in done]
    tasks = [_play_and_record(game=game, seed=s, evaluator=evaluator,
                              n_sims=n_sims, rec=rec, sem=sem, active=active)
             for s in seeds]
    await asyncio.gather(*tasks, return_exceptions=True)
    mean_batch = evaluator.total_requests / max(evaluator.total_batches, 1)
    print(f"[self_play] done: {len(seeds)} games, "
          f"mean_batch={mean_batch:.1f}, "
          f"total_batches={evaluator.total_batches}")
    await evaluator.stop()
    # Use a labeled checkpoint (config_id prefix) so multiple resume-runs into
    # the same dir each produce a distinct games.<label>.parquet rather than
    # overwriting a shared games.parquet. rglob("games*.parquet") in tests
    # and downstream tooling finds all shards transparently.
    rec.checkpoint(rec.config_id[:8])


def run_self_play(*, out_root: Path, checkpoint: Path, num_games: int = 64,
                  n_sims: int = 200, n_concurrent: int = 64,
                  hidden_dim: int = 128, num_layers: int = 4,
                  vp_target: int = 10, bonuses: bool = True, device: str = "cpu",
                  max_batch: int = 64, window_ms: float = 5.0,
                  max_seconds: float = 900.0, seed_base: int = 20_000_000,
                  resume_dir: Path | None = None,
                  ram_budget_mb: float | None = None,
                  per_game_mb: float = 50.0) -> Path:
    out = resume_dir if resume_dir is not None else make_run_dir(out_root, "self_play_async")
    asyncio.run(_run_async(
        out=out, checkpoint=checkpoint, num_games=num_games, n_sims=n_sims,
        n_concurrent=n_concurrent, hidden_dim=hidden_dim, num_layers=num_layers,
        vp_target=vp_target, bonuses=bonuses, device=device, max_batch=max_batch,
        window_ms=window_ms, seed_base=seed_base, resume=resume_dir is not None,
        ram_budget_mb=ram_budget_mb, per_game_mb=per_game_mb))
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--num-games", type=int, default=64)
    p.add_argument("--n-sims", type=int, default=200)
    p.add_argument("--n-concurrent", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--vp-target", type=int, default=10)
    p.add_argument("--bonuses", action="store_true", default=True)
    p.add_argument("--no-bonuses", dest="bonuses", action="store_false")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--window-ms", type=float, default=5.0)
    p.add_argument("--seed-base", type=int, default=20_000_000)
    p.add_argument("--ram-budget-mb", type=float, default=None)
    args = p.parse_args()
    out = run_self_play(
        out_root=args.out_root, checkpoint=args.checkpoint, num_games=args.num_games,
        n_sims=args.n_sims, n_concurrent=args.n_concurrent, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, vp_target=args.vp_target, bonuses=args.bonuses,
        device=args.device, max_batch=args.max_batch, window_ms=args.window_ms,
        seed_base=args.seed_base, ram_budget_mb=args.ram_budget_mb)
    print(f"self_play_async wrote to {out}")


if __name__ == "__main__":
    cli_main()
