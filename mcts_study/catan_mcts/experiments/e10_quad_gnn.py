"""Experiment 10d: 4-PureGnn tournament (no LookaheadV3, no Random).

Variant of e10_triple_gnn that uses 4 PureGnn slots instead of 3 PureGnn
+ 1 LookaheadV3. Used to isolate GNN-vs-GNN comparison without
LookaheadV3 dominating the result.

Seating (per rotation):
  - Slot 0: PureGnnA (--checkpoint-a)
  - Slot 1: PureGnnB (--checkpoint-b)
  - Slot 2: PureGnnC (--checkpoint-c)
  - Slot 3: PureGnnD (--checkpoint-d)

All 4 checkpoints are PureGnnBot wrappers — no MCTS, no LookaheadV3.
Every game is decided by which GNN's argmax-policy wins outright.
"""
from __future__ import annotations

import argparse
import random
from multiprocessing import get_context
from pathlib import Path

import torch
from tqdm import tqdm

from catan_gnn.gnn_model import GnnModel

from ..adapter import CatanGame
from ..bots_gnn import PureGnnBot
from ..recorder import SelfPlayRecorder
from .common import make_run_dir, play_one_game


_BASE_SEATING = ["PureGnnA", "PureGnnB", "PureGnnC", "PureGnnD"]


def _resolve_device(spec: str) -> str:
    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


def _load_model(checkpoint: Path, hidden_dim: int, num_layers: int,
                device: str = "cpu") -> GnnModel:
    model = GnnModel(hidden_dim=hidden_dim, num_layers=num_layers)
    obj = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model_state" in obj:
        state = obj["model_state"]
    else:
        state = obj
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def _run_cell(rec: SelfPlayRecorder, rot_idx: int,
              model_a: GnnModel, model_b: GnnModel,
              model_c: GnnModel, model_d: GnnModel,
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
        bots = {}
        for slot, role in enumerate(seating):
            if role == "PureGnnA":
                bots[slot] = PureGnnBot(model=model_a, device=device)
            elif role == "PureGnnB":
                bots[slot] = PureGnnBot(model=model_b, device=device)
            elif role == "PureGnnC":
                bots[slot] = PureGnnBot(model=model_c, device=device)
            elif role == "PureGnnD":
                bots[slot] = PureGnnBot(model=model_d, device=device)
            else:
                raise RuntimeError(f"unknown role: {role}")
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
    (worker_idx, parent_out,
     checkpoint_a, checkpoint_b, checkpoint_c, checkpoint_d,
     vp_target, bonuses, seeds_per_rot, max_seconds,
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
    model_c = _load_model(checkpoint_c, hidden_dim, num_layers, device=resolved_device)
    model_d = _load_model(checkpoint_d, hidden_dim, num_layers, device=resolved_device)
    for rot_idx in range(4):
        seeds = seeds_per_rot[rot_idx]
        _run_cell(
            rec, rot_idx, model_a, model_b, model_c, model_d,
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
    checkpoint_c: Path,
    checkpoint_d: Path,
    label_a: str = "PureGnnA",
    label_b: str = "PureGnnB",
    label_c: str = "PureGnnC",
    label_d: str = "PureGnnD",
    num_games_per_seating: int = 30,
    hidden_dim: int = 128,
    num_layers: int = 4,
    seed_base: int = 19_000_000,
    max_seconds: float = 600.0,
    vp_target: int = 5,
    bonuses: bool = False,
    resume: bool = True,
    workers: int = 10,
    device: str = "auto",
) -> Path:
    out = make_run_dir(out_root, "e10d_quad_gnn")
    base_config = {
        "experiment": "e10d_quad_gnn",
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "checkpoint_c": str(checkpoint_c),
        "checkpoint_d": str(checkpoint_d),
        "label_a": label_a,
        "label_b": label_b,
        "label_c": label_c,
        "label_d": label_d,
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
        model_c = _load_model(checkpoint_c, hidden_dim, num_layers, device=resolved_device)
        model_d = _load_model(checkpoint_d, hidden_dim, num_layers, device=resolved_device)
        for rot_idx in range(4):
            seeds = [seed_base + rot_idx * 10_000 + i for i in range(num_games_per_seating)]
            _run_cell(
                rec, rot_idx, model_a, model_b, model_c, model_d,
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
            w, out,
            checkpoint_a, checkpoint_b, checkpoint_c, checkpoint_d,
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
    p.add_argument("--checkpoint-c", type=Path, required=True)
    p.add_argument("--checkpoint-d", type=Path, required=True)
    p.add_argument("--label-a", type=str, default="PureGnnA")
    p.add_argument("--label-b", type=str, default="PureGnnB")
    p.add_argument("--label-c", type=str, default="PureGnnC")
    p.add_argument("--label-d", type=str, default="PureGnnD")
    p.add_argument("--num-games-per-seating", type=int, default=30,
                   help="120 total games at default (30 per rotation x 4 rotations)")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--seed-base", type=int, default=19_000_000)
    p.add_argument("--max-seconds", type=float, default=600.0)
    p.add_argument("--vp-target", type=int, default=5)
    p.add_argument("--bonuses", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    out = main(
        out_root=args.out_root,
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        checkpoint_c=args.checkpoint_c,
        checkpoint_d=args.checkpoint_d,
        label_a=args.label_a,
        label_b=args.label_b,
        label_c=args.label_c,
        label_d=args.label_d,
        num_games_per_seating=args.num_games_per_seating,
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
    print(f"e10d wrote to {out}")


if __name__ == "__main__":
    cli_main()
