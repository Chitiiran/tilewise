"""Experiment 10: 4-player tournament under v3 rules (Catan-Lite).

Direct port of e7 with two changes:
  1. Engine constructed with `vp_target=5, bonuses=False`.
  2. The "LookaheadMcts" baseline is replaced with `LookaheadMctsV3`
     (per-player VP-aware exponential sim schedule per spec §5).

Players (one per seat, rotated through all 4 cyclic rotations):
  - GnnMcts:        MCTSBot at sims=N with the trained GnnEvaluator
  - PureGnn:        argmax of GNN policy head, no search
  - LookaheadMctsV3: LookaheadMctsV3 — the v3 strong baseline that the
                    GNN must beat (currently 0/4 in v2)
  - Random:         uniform legal action

Per spec §12 success criterion #3: PureGnn or GnnMcts must win >= 1/4 seats
in this tournament for v3.5/v3.6 to pass.
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
from ..bots_gnn import PureGnnBot
from ..gnn_evaluator import GnnEvaluator
from ..players_v3 import build_lookahead_mcts_v3
from ..recorder import SelfPlayRecorder
from .common import make_run_dir, play_one_game


class _RandomBot:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())


_BASE_SEATING = ["GnnMcts", "PureGnn", "LookaheadMctsV3", "Random"]


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


def _build_gnn_mcts(game, model: GnnModel, sims: int, seed: int, device: str = "cpu"):
    rng = np.random.default_rng(seed)
    evaluator = GnnEvaluator(model=model, device=device)
    return os_mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=sims,
        evaluator=evaluator, solve=False, random_state=rng,
    )


def _run_cell(rec: SelfPlayRecorder, rot_idx: int, model: GnnModel, sims: int,
              lookahead_depth: int, base_sims_v3: int, vp_target: int,
              bonuses: bool, seeds: list[int], done: set[int],
              max_seconds: float, progress_desc_prefix: str = "",
              device: str = "cpu") -> None:
    game = CatanGame(vp_target=vp_target, bonuses=bonuses)
    seating = _BASE_SEATING[rot_idx:] + _BASE_SEATING[:rot_idx]
    desc = f"{progress_desc_prefix}rot={rot_idx} {'-'.join(seating)}"
    for seed in tqdm(seeds, desc=desc, leave=False):
        if seed in done:
            continue
        chance_rng = random.Random(seed)
        gnn_mcts_bot = _build_gnn_mcts(game, model=model, sims=sims, seed=seed,
                                       device=device)
        lookahead_v3_bot = build_lookahead_mcts_v3(
            game, lookahead_depth=lookahead_depth, seed=seed,
            base_sims=base_sims_v3,
        )
        bots = {}
        gnn_mcts_slot = -1
        for slot, role in enumerate(seating):
            if role == "GnnMcts":
                bots[slot] = gnn_mcts_bot
                gnn_mcts_slot = slot
            elif role == "PureGnn":
                bots[slot] = PureGnnBot(model=model, device=device)
            elif role == "LookaheadMctsV3":
                bots[slot] = lookahead_v3_bot
            else:
                bots[slot] = _RandomBot(seed + 200 + slot)
        with rec.game(seed=seed) as g_rec:
            outcome = play_one_game(
                game=game, bots=bots, seed=seed, chance_rng=chance_rng,
                recorded_player=gnn_mcts_slot, recorder_game=g_rec,
                mcts_bot=gnn_mcts_bot, max_seconds=max_seconds,
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
    (worker_idx, parent_out, checkpoint, sims, lookahead_depth, base_sims_v3,
     vp_target, bonuses, seeds_per_rot, max_seconds, base_config,
     hidden_dim, num_layers, device) = args
    resolved_device = _resolve_device(device)
    worker_dir = parent_out / f"worker{worker_idx}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    rec = SelfPlayRecorder(worker_dir, config={**base_config, "worker_idx": worker_idx,
                                                 "device": resolved_device})
    done = rec.done_seeds()
    model = _load_model(checkpoint, hidden_dim, num_layers, device=resolved_device)
    for rot_idx in range(4):
        seeds = seeds_per_rot[rot_idx]
        _run_cell(rec, rot_idx, model, sims, lookahead_depth, base_sims_v3,
                  vp_target, bonuses, seeds, done, max_seconds,
                  progress_desc_prefix=f"w{worker_idx} ", device=resolved_device)
        rec.checkpoint(f"rot={rot_idx}")
    rec.flush()


def main(
    *,
    out_root: Path,
    checkpoint: Path,
    num_games_per_seating: int = 5,
    sims: int = 100,
    lookahead_depth: int = 10,
    base_sims_v3: int = 200,
    hidden_dim: int = 32,
    num_layers: int = 2,
    seed_base: int = 10_000_000,
    max_seconds: float = 600.0,
    vp_target: int = 5,
    bonuses: bool = False,
    resume: bool = True,
    workers: int = 1,
    device: str = "auto",
) -> Path:
    out = make_run_dir(out_root, "e10_v3_tournament")
    base_config = {
        "experiment": "e10_v3_tournament",
        "checkpoint": str(checkpoint),
        "sims": sims,
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
    }

    if workers <= 1:
        resolved_device = _resolve_device(device)
        rec = SelfPlayRecorder(out, config={**base_config, "device": resolved_device})
        done = rec.done_seeds() if resume else set()
        model = _load_model(checkpoint, hidden_dim, num_layers, device=resolved_device)
        for rot_idx in range(4):
            seeds = [seed_base + rot_idx * 10_000 + i for i in range(num_games_per_seating)]
            _run_cell(rec, rot_idx, model, sims, lookahead_depth, base_sims_v3,
                      vp_target, bonuses, seeds, done, max_seconds,
                      device=resolved_device)
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
            w, out, checkpoint, sims, lookahead_depth, base_sims_v3,
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
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--num-games-per-seating", type=int, default=5)
    p.add_argument("--sims", type=int, default=100,
                   help="Sims for GnnMcts (the AlphaZero-style player)")
    p.add_argument("--lookahead-depth", type=int, default=10,
                   help="Lookahead depth for LookaheadMctsV3 leaf evaluator")
    p.add_argument("--base-sims-v3", type=int, default=200,
                   help="Sim budget at acting_vp=0 for LookaheadMctsV3 (decays per spec §5)")
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--seed-base", type=int, default=10_000_000)
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
        checkpoint=args.checkpoint,
        num_games_per_seating=args.num_games_per_seating,
        sims=args.sims,
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
    print(f"e10 wrote to {out}")


if __name__ == "__main__":
    cli_main()
