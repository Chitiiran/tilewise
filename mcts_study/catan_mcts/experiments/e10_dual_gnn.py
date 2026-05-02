"""Experiment 10b: dual-PureGnn vs LookaheadV3 vs Random tournament.

Variant of e10 that swaps GnnMcts for a SECOND PureGnn slot, so we can
A/B two trained checkpoints head-to-head (e.g. PureGnn-10k-epoch3 vs
PureGnn-d500) under identical conditions.

Seating (per rotation):
  - Slot 0: PureGnnA (--checkpoint-a)
  - Slot 1: PureGnnB (--checkpoint-b)
  - Slot 2: LookaheadMctsV3
  - Slot 3: Random

The checkpoint labels (`label_a`, `label_b`) are recorded in the run
config so analysis can recover which seat was which.

Args mirror e10 but split GnnMcts/PureGnn into two PureGnn checkpoint
slots. No MCTS-on-GNN player (GnnMcts) since we're isolating the
policy-head quality of the two checkpoints, not MCTS+GNN performance.
"""
from __future__ import annotations

import argparse
import random
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from catan_gnn.gnn_model import GnnModel

from ..adapter import CatanGame
from ..bots_gnn import PureGnnBot
from ..players_v3 import build_lookahead_mcts_v3
from ..recorder import SelfPlayRecorder
from .common import make_run_dir, play_one_game


class _RandomBot:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())


# Slot order. Rotation k cycles by k positions.
_BASE_SEATING = ["PureGnnA", "PureGnnB", "LookaheadMctsV3", "Random"]


def _resolve_device(spec: str) -> str:
    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


def _load_model(checkpoint: Path, hidden_dim: int, num_layers: int,
                device: str = "cpu") -> GnnModel:
    model = GnnModel(hidden_dim=hidden_dim, num_layers=num_layers)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def _run_cell(rec: SelfPlayRecorder, rot_idx: int,
              model_a: GnnModel, model_b: GnnModel,
              lookahead_depth: int, base_sims_v3: int,
              vp_target: int, bonuses: bool,
              seeds: list[int], done: set[int],
              max_seconds: float, progress_desc_prefix: str = "",
              device: str = "cpu") -> None:
    game = CatanGame(vp_target=vp_target, bonuses=bonuses)
    seating = _BASE_SEATING[rot_idx:] + _BASE_SEATING[:rot_idx]
    desc = f"{progress_desc_prefix}rot={rot_idx} {'-'.join(seating)}"
    for seed in tqdm(seeds, desc=desc, leave=False):
        if seed in done:
            continue
        chance_rng = random.Random(seed)
        lookahead_v3_bot = build_lookahead_mcts_v3(
            game, lookahead_depth=lookahead_depth, seed=seed,
            base_sims=base_sims_v3,
        )
        bots = {}
        for slot, role in enumerate(seating):
            if role == "PureGnnA":
                bots[slot] = PureGnnBot(model=model_a, device=device)
            elif role == "PureGnnB":
                bots[slot] = PureGnnBot(model=model_b, device=device)
            elif role == "LookaheadMctsV3":
                bots[slot] = lookahead_v3_bot
            else:
                bots[slot] = _RandomBot(seed + 200 + slot)
        with rec.game(seed=seed) as g_rec:
            outcome = play_one_game(
                game=game, bots=bots, seed=seed, chance_rng=chance_rng,
                recorded_player=None, recorder_game=None, mcts_bot=None,
                max_seconds=max_seconds,
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


def _worker(args) -> None:
    (worker_idx, parent_out, checkpoint_a, checkpoint_b, lookahead_depth,
     base_sims_v3, vp_target, bonuses, seeds_per_rot, max_seconds,
     base_config, hidden_dim, num_layers, device) = args
    resolved_device = _resolve_device(device)
    worker_dir = parent_out / f"worker{worker_idx}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    rec = SelfPlayRecorder(
        worker_dir,
        config={**base_config, "worker_idx": worker_idx, "device": resolved_device},
    )
    done = rec.done_seeds()
    model_a = _load_model(checkpoint_a, hidden_dim, num_layers, device=resolved_device)
    model_b = _load_model(checkpoint_b, hidden_dim, num_layers, device=resolved_device)
    for rot_idx in range(4):
        seeds = seeds_per_rot[rot_idx]
        _run_cell(
            rec, rot_idx, model_a, model_b, lookahead_depth, base_sims_v3,
            vp_target, bonuses, seeds, done, max_seconds,
            progress_desc_prefix=f"w{worker_idx} ", device=resolved_device,
        )
        rec.checkpoint(f"rot={rot_idx}")
    rec.flush()


def main(
    *,
    out_root: Path,
    checkpoint_a: Path,
    checkpoint_b: Path,
    label_a: str = "PureGnnA",
    label_b: str = "PureGnnB",
    num_games_per_seating: int = 12,
    lookahead_depth: int = 10,
    base_sims_v3: int = 200,
    hidden_dim: int = 32,
    num_layers: int = 2,
    seed_base: int = 16_000_000,
    max_seconds: float = 600.0,
    vp_target: int = 5,
    bonuses: bool = False,
    resume: bool = True,
    workers: int = 1,
    device: str = "auto",
) -> Path:
    out = make_run_dir(out_root, "e10b_dual_gnn")
    base_config = {
        "experiment": "e10b_dual_gnn",
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "label_a": label_a,
        "label_b": label_b,
        "lookahead_depth": lookahead_depth,
        "base_sims_v3": base_sims_v3,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_games_per_seating": num_games_per_seating,
        "max_seconds": max_seconds,
        "vp_target": vp_target,
        "bonuses": bonuses,
        "workers": workers,
        "seating": _BASE_SEATING,
        "seed_base": seed_base,
    }

    if workers <= 1:
        resolved_device = _resolve_device(device)
        rec = SelfPlayRecorder(out, config={**base_config, "device": resolved_device})
        done = rec.done_seeds() if resume else set()
        model_a = _load_model(checkpoint_a, hidden_dim, num_layers, device=resolved_device)
        model_b = _load_model(checkpoint_b, hidden_dim, num_layers, device=resolved_device)
        for rot_idx in range(4):
            seeds = [seed_base + rot_idx * 10_000 + i for i in range(num_games_per_seating)]
            _run_cell(
                rec, rot_idx, model_a, model_b, lookahead_depth, base_sims_v3,
                vp_target, bonuses, seeds, done, max_seconds,
                device=resolved_device,
            )
            rec.checkpoint(f"rot={rot_idx}")
        rec.flush()
        return out

    seeds_per_rot_per_worker: list[list[list[int]]] = [
        [[] for _ in range(workers)] for _ in range(4)
    ]
    for rot_idx in range(4):
        for i in range(num_games_per_seating):
            seed = seed_base + rot_idx * 10_000 + i
            seeds_per_rot_per_worker[rot_idx][i % workers].append(seed)

    args_list = []
    for w in range(workers):
        seeds_per_rot = [seeds_per_rot_per_worker[r][w] for r in range(4)]
        args_list.append((
            w, out, checkpoint_a, checkpoint_b, lookahead_depth, base_sims_v3,
            vp_target, bonuses, seeds_per_rot, max_seconds, base_config,
            hidden_dim, num_layers, device,
        ))
    ctx = get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        pool.map(_worker, args_list)
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--checkpoint-a", type=Path, required=True)
    p.add_argument("--checkpoint-b", type=Path, required=True)
    p.add_argument("--label-a", type=str, default="PureGnnA")
    p.add_argument("--label-b", type=str, default="PureGnnB")
    p.add_argument("--num-games-per-seating", type=int, default=12,
                   help="48 total games at default (12 per rotation x 4 rotations)")
    p.add_argument("--lookahead-depth", type=int, default=10)
    p.add_argument("--base-sims-v3", type=int, default=200)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--seed-base", type=int, default=16_000_000)
    p.add_argument("--max-seconds", type=float, default=600.0)
    p.add_argument("--vp-target", type=int, default=5)
    p.add_argument("--bonuses", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    out = main(
        out_root=args.out_root,
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        label_a=args.label_a,
        label_b=args.label_b,
        num_games_per_seating=args.num_games_per_seating,
        lookahead_depth=args.lookahead_depth,
        base_sims_v3=args.base_sims_v3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seed_base=args.seed_base,
        max_seconds=args.max_seconds,
        vp_target=args.vp_target,
        bonuses=args.bonuses,
        resume=not args.no_resume,
        workers=args.workers,
        device=args.device,
    )
    print(f"e10b wrote to {out}")


if __name__ == "__main__":
    cli_main()
