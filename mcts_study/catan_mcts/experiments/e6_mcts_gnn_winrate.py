"""Experiment 6: MCTS-with-GNN-evaluator winrate vs 3 random opponents.

Spec §7 Bench-3. Mirrors e1's structure (sims grid, 4-worker option,
done.txt resumability), but the evaluator is GnnEvaluator(checkpoint=...)
instead of RustRolloutEvaluator or LookaheadVpEvaluator.
"""
from __future__ import annotations

import argparse
import random
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from open_spiel.python.algorithms import mcts as os_mcts
from tqdm import tqdm

from catan_gnn.gnn_model import GnnModel
from ..adapter import CatanGame
from ..gnn_evaluator import GnnEvaluator
from ..recorder import SelfPlayRecorder
from .common import make_run_dir, play_one_game


class _RandomBot:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())


def _load_model(checkpoint: Path, hidden_dim: int, num_layers: int) -> GnnModel:
    model = GnnModel(hidden_dim=hidden_dim, num_layers=num_layers)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _build_mcts(game, sims: int, evaluator, seed: int):
    rng = np.random.default_rng(seed)
    return os_mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=sims,
        evaluator=evaluator, solve=False, random_state=rng,
    )


def _run_cell(rec, sims, seeds, done, max_seconds, model, progress_desc_prefix=""):
    game = CatanGame()
    for seed in tqdm(seeds, desc=f"{progress_desc_prefix}sims={sims}", leave=False):
        if seed in done:
            continue
        chance_rng = random.Random(seed)
        evaluator = GnnEvaluator(model=model)   # fresh evaluator per game (cache reset)
        mcts_bot = _build_mcts(game, sims=sims, evaluator=evaluator, seed=seed)
        bots = {0: mcts_bot, 1: _RandomBot(seed+1), 2: _RandomBot(seed+2), 3: _RandomBot(seed+3)}
        with rec.game(seed=seed) as g_rec:
            outcome = play_one_game(
                game=game, bots=bots, seed=seed, chance_rng=chance_rng,
                recorded_player=0, recorder_game=g_rec, mcts_bot=mcts_bot,
                max_seconds=max_seconds,
            )
            if outcome.timed_out:
                g_rec._moves.clear()
                g_rec._finalized = True
                rec.skip_game(
                    seed=seed, reason="wall-clock-timeout",
                    length_in_moves=outcome.length_in_moves,
                )
            else:
                g_rec.finalize(
                    winner=outcome.winner, final_vp=outcome.final_vp,
                    length_in_moves=outcome.length_in_moves,
                    action_history=outcome.action_history,
                )
                rec.mark_done(seed)


def main(
    *,
    out_root: Path,
    checkpoint: Path,
    num_games: int = 50,
    sims_grid: list[int] = (100, 400),
    hidden_dim: int = 32,
    num_layers: int = 2,
    seed_base: int = 6_000_000,
    max_seconds: float = 360.0,
    resume: bool = True,
    workers: int = 1,
) -> Path:
    out = make_run_dir(out_root, "e6_mcts_gnn_winrate")
    base_config = {
        "experiment": "e6_mcts_gnn_winrate",
        "checkpoint": str(checkpoint),
        "hidden_dim": hidden_dim, "num_layers": num_layers,
        "sims_grid": list(sims_grid), "num_games": num_games,
        "max_seconds": max_seconds, "workers": workers,
    }

    if workers <= 1:
        rec = SelfPlayRecorder(out, config=base_config)
        done = rec.done_seeds() if resume else set()
        model = _load_model(checkpoint, hidden_dim, num_layers)
        for sims in sims_grid:
            seeds = [seed_base + sims * 1_000 + i for i in range(num_games)]
            _run_cell(rec, sims, seeds, done, max_seconds, model)
            rec.checkpoint(f"sims={sims}")
        rec.flush()
        return out

    # Parallel mode (mirrors e1).
    seeds_per_cell_per_worker = [[[] for _ in range(workers)] for _ in sims_grid]
    for cell_idx, sims in enumerate(sims_grid):
        for i in range(num_games):
            seed = seed_base + sims * 1_000 + i
            seeds_per_cell_per_worker[cell_idx][i % workers].append(seed)
    args_list = []
    for w in range(workers):
        seeds_per_cell = [seeds_per_cell_per_worker[c][w] for c in range(len(sims_grid))]
        args_list.append((
            w, out, list(sims_grid), seeds_per_cell, max_seconds, base_config,
            checkpoint, hidden_dim, num_layers,
        ))
    ctx = get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        pool.map(_worker, args_list)
    return out


def _worker(args):
    worker_idx, parent_out, sims_grid, seeds_per_cell, max_seconds, base_config, ckpt, hd, nl = args
    worker_dir = parent_out / f"worker{worker_idx}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    rec = SelfPlayRecorder(worker_dir, config={**base_config, "worker_idx": worker_idx})
    done = rec.done_seeds()
    model = _load_model(ckpt, hd, nl)
    for sims, seeds in zip(sims_grid, seeds_per_cell):
        _run_cell(rec, sims, seeds, done, max_seconds, model,
                  progress_desc_prefix=f"w{worker_idx} ")
        rec.checkpoint(f"sims={sims}")
    rec.flush()


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--num-games", type=int, default=50)
    p.add_argument("--sims-grid", type=int, nargs="+", default=[100, 400])
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--seed-base", type=int, default=6_000_000)
    p.add_argument("--max-seconds", type=float, default=360.0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    out = main(
        out_root=args.out_root, checkpoint=args.checkpoint,
        num_games=args.num_games, sims_grid=args.sims_grid,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        seed_base=args.seed_base, max_seconds=args.max_seconds,
        resume=not args.no_resume, workers=args.workers,
    )
    print(f"e6 wrote to {out}")


if __name__ == "__main__":
    cli_main()
