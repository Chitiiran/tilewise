"""Experiment 8: head-to-head tournament with TWO GNN checkpoints, no MCTS-on-GNN.

Players (one per seat, rotated through all 4 cyclic rotations):
  - GnnA:          PureGnn (argmax of policy head, no search) using checkpoint A
  - GnnB:          PureGnn (argmax of policy head, no search) using checkpoint B
  - LookaheadMcts: MCTSBot at sims=N with LookaheadVpEvaluator (existing strong baseline)
  - Random:        uniform legal action (control)

Why these four and why no MCTS-on-GNN: e7's findings showed MCTS amplifies a
noisy GNN's mistakes, so for THIS comparison we want the raw policy head only —
i.e. "does GNN-A's policy beat GNN-B's policy at choosing the next move?"
The MCTS+GNN combination remains in e7 for the orthogonal question of whether
search helps a given evaluator.

Intended uses:
  - v1 (older) vs v2 (newer) policy quality, using LookaheadMcts + Random as anchors.
  - epoch 5 vs epoch 20 of the same training run, once per-epoch checkpoints exist.

Wishlist §4b.3 ("Two-GNN tournament mode"). Sister script to e7_gnn_tournament.
"""
from __future__ import annotations

import argparse
import itertools
import random
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from open_spiel.python.algorithms import mcts as os_mcts
from tqdm import tqdm

from catan_gnn.gnn_model import GnnModel

from ..adapter import CatanGame
from ..bots_gnn import PureGnnBot
from ..evaluator import LookaheadVpEvaluator
from ..recorder import SelfPlayRecorder
from .common import make_run_dir, play_one_game


class _RandomBot:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())


_BASE_SEATING = ["GnnA", "GnnB", "LookaheadMcts", "Random"]


def _load_model(checkpoint: Path, hidden_dim: int, num_layers: int) -> GnnModel:
    model = GnnModel(hidden_dim=hidden_dim, num_layers=num_layers)
    state = torch.load(checkpoint, map_location="cpu")
    # Per-epoch bundles store a dict; weights-only files store the state_dict directly.
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model


def _build_lookahead_mcts(game, sims: int, lookahead_depth: int, seed: int):
    rng = np.random.default_rng(seed)
    evaluator = LookaheadVpEvaluator(depth=lookahead_depth, base_seed=seed)
    return os_mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=sims,
        evaluator=evaluator, solve=False, random_state=rng,
    )


def _run_cell(rec: SelfPlayRecorder, perm_idx: int, seating: list[str],
              model_a: GnnModel, model_b: GnnModel,
              sims: int, lookahead_depth: int,
              seeds: list[int], done: set[int],
              max_seconds: float, progress_desc_prefix: str = "",
              flush_every: int = 5) -> None:
    """Run a slice of games for one seating permutation.

    Salvageability (kill-midway safety):
    - `done.txt` is appended after every finished game (durable).
    - Buffered rows are flushed to a unique parquet shard every `flush_every`
      games — so a kill loses at most flush_every-1 games of in-memory state.
    - Each shard's filename includes perm_idx + a chunk counter so we don't
      overwrite earlier flushes within the same permutation.
    """
    game = CatanGame()
    desc = f"{progress_desc_prefix}p{perm_idx:02d} {'-'.join(seating)}"
    games_since_flush = 0
    # Resume safety: don't overwrite chunk files from a prior run. Seed
    # chunk_idx past the highest existing chunk for this permutation.
    import re as _re
    existing_chunks = list(rec._out_dir.glob(f"games.perm={perm_idx:02d}.chunk*.parquet"))
    chunk_idx = 0
    for f in existing_chunks:
        m = _re.search(r"chunk(\d+)", f.name)
        if m:
            chunk_idx = max(chunk_idx, int(m.group(1)) + 1)
    for seed in tqdm(seeds, desc=desc, leave=False):
        if seed in done:
            continue
        chance_rng = random.Random(seed)
        lookahead_mcts_bot = _build_lookahead_mcts(
            game, sims=sims, lookahead_depth=lookahead_depth, seed=seed,
        )
        bots: dict = {}
        for slot, role in enumerate(seating):
            if role == "GnnA":
                bots[slot] = PureGnnBot(model=model_a)
            elif role == "GnnB":
                bots[slot] = PureGnnBot(model=model_b)
            elif role == "LookaheadMcts":
                bots[slot] = lookahead_mcts_bot
            else:
                bots[slot] = _RandomBot(seed + 200 + slot)
        # No recorded_player — we don't need MCTS visit counts for this
        # tournament (no GnnMcts seat). Just play and record outcome.
        with rec.game(seed=seed) as g_rec:
            outcome = play_one_game(
                game=game, bots=bots, seed=seed, chance_rng=chance_rng,
                recorded_player=None, recorder_game=g_rec,
                mcts_bot=None, max_seconds=max_seconds,
            )
            if outcome.timed_out:
                rec.skip_game(
                    seed=seed, reason="wall-clock-timeout",
                    length_in_moves=outcome.length_in_moves,
                    action_history=outcome.action_history,
                    moves_recorder=g_rec,
                )
            else:
                g_rec.finalize(
                    winner=outcome.winner,
                    final_vp=outcome.final_vp,
                    length_in_moves=outcome.length_in_moves,
                    action_history=outcome.action_history,
                )
                rec.mark_done(seed)
        games_since_flush += 1
        if games_since_flush >= flush_every:
            rec.checkpoint(f"perm={perm_idx:02d}.chunk{chunk_idx:03d}")
            games_since_flush = 0
            chunk_idx += 1
    # Flush any tail rows for this permutation.
    if games_since_flush > 0:
        rec.checkpoint(f"perm={perm_idx:02d}.chunk{chunk_idx:03d}")


def _all_permutations() -> list[list[str]]:
    """All 24 orderings of the 4 player roles. Used in --mode permutations
    so each role-pair adjacency (e.g., LookaheadMcts immediately followed by
    GnnB) appears the same number of times — no role pair is privileged.
    """
    return [list(p) for p in itertools.permutations(_BASE_SEATING)]


def _seating_for(mode: str, idx: int) -> list[str]:
    """Return the seating for a given seating-index. mode='cyclic' (idx ∈ 0..3)
    or mode='permutations' (idx ∈ 0..23)."""
    if mode == "cyclic":
        return _BASE_SEATING[idx:] + _BASE_SEATING[:idx]
    elif mode == "permutations":
        return _all_permutations()[idx]
    else:
        raise ValueError(f"unknown mode {mode!r}")


def _num_seatings(mode: str) -> int:
    return 4 if mode == "cyclic" else 24


# Seed encoding: seed = seed_base + perm_idx * 1000 + game_idx_within_seating.
# perm_idx 0..23 (or 0..3 in cyclic), game_idx 0..999. Aggregator uses
# (seed - seed_base) // 1000 to recover perm_idx. Caps games-per-seating at
# 1000, which is far above what we need.
_SEED_PERM_STRIDE = 1000


def _worker(args) -> None:
    (worker_idx, parent_out, ckpt_a, ckpt_b, sims, lookahead_depth,
     mode, seeds_per_perm, max_seconds, base_config,
     hidden_dim_a, num_layers_a, hidden_dim_b, num_layers_b) = args
    worker_dir = parent_out / f"worker{worker_idx}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    rec = SelfPlayRecorder(worker_dir, config={**base_config, "worker_idx": worker_idx})
    done = rec.done_seeds()
    model_a = _load_model(ckpt_a, hidden_dim_a, num_layers_a)
    model_b = _load_model(ckpt_b, hidden_dim_b, num_layers_b)
    for perm_idx in range(_num_seatings(mode)):
        seeds = seeds_per_perm[perm_idx]
        if not seeds:
            continue
        seating = _seating_for(mode, perm_idx)
        _run_cell(rec, perm_idx, seating, model_a, model_b, sims, lookahead_depth,
                  seeds, done, max_seconds, progress_desc_prefix=f"w{worker_idx} ")
        # _run_cell already flushed in chunks; no outer checkpoint needed.
    rec.flush()


def main(
    *,
    out_root: Path,
    checkpoint_a: Path,
    checkpoint_b: Path,
    label_a: str = "GnnA",
    label_b: str = "GnnB",
    num_games_per_seating: int = 3,
    sims: int = 50,
    lookahead_depth: int = 25,
    hidden_dim_a: int = 32,
    num_layers_a: int = 2,
    hidden_dim_b: int = 32,
    num_layers_b: int = 2,
    seed_base: int = 14_000_000,
    max_seconds: float = 600.0,
    resume: bool = True,
    workers: int = 1,
    mode: str = "cyclic",
) -> Path:
    """Run the 4-player tournament.

    mode='cyclic': 4 cyclic rotations × num_games_per_seating games each.
        Total = 4 × num_games_per_seating.
    mode='permutations': all 24 orderings × num_games_per_seating games each.
        Total = 24 × num_games_per_seating. Avoids the "role X is always next
        to role Y" artifact that cyclic has — every role-pair adjacency
        appears in the same proportion.
    """
    out = make_run_dir(out_root, "e8_gnn_vs_gnn")
    n_seatings = _num_seatings(mode)
    base_config = {
        "experiment": "e8_gnn_vs_gnn",
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "label_a": label_a,
        "label_b": label_b,
        "sims": sims,
        "lookahead_depth": lookahead_depth,
        "hidden_dim_a": hidden_dim_a, "num_layers_a": num_layers_a,
        "hidden_dim_b": hidden_dim_b, "num_layers_b": num_layers_b,
        "num_games_per_seating": num_games_per_seating,
        "max_seconds": max_seconds,
        "workers": workers,
        "seating": _BASE_SEATING,           # canonical role list
        "seating_mode": mode,
        "n_seatings": n_seatings,
        "seed_base": seed_base,
        "seed_perm_stride": _SEED_PERM_STRIDE,
    }
    if num_games_per_seating > _SEED_PERM_STRIDE:
        raise ValueError(
            f"num_games_per_seating ({num_games_per_seating}) exceeds "
            f"_SEED_PERM_STRIDE ({_SEED_PERM_STRIDE}); seed encoding would collide"
        )

    if workers <= 1:
        rec = SelfPlayRecorder(out, config=base_config)
        done = rec.done_seeds() if resume else set()
        model_a = _load_model(checkpoint_a, hidden_dim_a, num_layers_a)
        model_b = _load_model(checkpoint_b, hidden_dim_b, num_layers_b)
        for perm_idx in range(n_seatings):
            seating = _seating_for(mode, perm_idx)
            seeds = [seed_base + perm_idx * _SEED_PERM_STRIDE + i
                     for i in range(num_games_per_seating)]
            _run_cell(rec, perm_idx, seating, model_a, model_b, sims, lookahead_depth,
                      seeds, done, max_seconds)
            # _run_cell already flushed in chunks; no outer checkpoint needed.
        rec.flush()
        return out

    # Parallel: each worker gets a slice of seeds for each permutation.
    seeds_per_perm_per_worker: list[list[list[int]]] = [
        [[] for _ in range(workers)] for _ in range(n_seatings)
    ]
    for perm_idx in range(n_seatings):
        for i in range(num_games_per_seating):
            seed = seed_base + perm_idx * _SEED_PERM_STRIDE + i
            seeds_per_perm_per_worker[perm_idx][i % workers].append(seed)

    args_list = []
    for w in range(workers):
        seeds_per_perm = [seeds_per_perm_per_worker[p][w] for p in range(n_seatings)]
        args_list.append((
            w, out, checkpoint_a, checkpoint_b, sims, lookahead_depth,
            mode, seeds_per_perm, max_seconds, base_config,
            hidden_dim_a, num_layers_a, hidden_dim_b, num_layers_b,
        ))

    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        for _ in pool.imap_unordered(_worker, args_list):
            pass
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--checkpoint-a", type=Path, required=True,
                   help="GNN-A checkpoint (e.g. v1_d15)")
    p.add_argument("--checkpoint-b", type=Path, required=True,
                   help="GNN-B checkpoint (e.g. v2_d25)")
    p.add_argument("--label-a", type=str, default="GnnA")
    p.add_argument("--label-b", type=str, default="GnnB")
    p.add_argument("--num-games-per-seating", type=int, default=3,
                   help="Games per cyclic rotation (×4 rotations = total games)")
    p.add_argument("--sims", type=int, default=50,
                   help="MCTS simulation count for LookaheadMcts seat only")
    p.add_argument("--lookahead-depth", type=int, default=25)
    p.add_argument("--hidden-dim-a", type=int, default=32)
    p.add_argument("--num-layers-a", type=int, default=2)
    p.add_argument("--hidden-dim-b", type=int, default=32)
    p.add_argument("--num-layers-b", type=int, default=2)
    p.add_argument("--seed-base", type=int, default=14_000_000)
    p.add_argument("--max-seconds", type=float, default=600.0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--mode", type=str, default="cyclic",
                   choices=["cyclic", "permutations"],
                   help="cyclic = 4 rotations; permutations = all 24 orderings "
                        "(no fixed-order role-pair adjacency)")
    args = p.parse_args()
    out = main(
        out_root=args.out_root,
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        label_a=args.label_a,
        label_b=args.label_b,
        num_games_per_seating=args.num_games_per_seating,
        sims=args.sims,
        lookahead_depth=args.lookahead_depth,
        hidden_dim_a=args.hidden_dim_a, num_layers_a=args.num_layers_a,
        hidden_dim_b=args.hidden_dim_b, num_layers_b=args.num_layers_b,
        seed_base=args.seed_base,
        max_seconds=args.max_seconds,
        resume=not args.no_resume,
        workers=args.workers,
        mode=args.mode,
    )
    print(f"e8 wrote to {out}")


if __name__ == "__main__":
    cli_main()
