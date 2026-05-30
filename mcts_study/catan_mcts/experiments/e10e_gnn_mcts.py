"""Experiment 10e: GNN+MCTS diagnostic — does search on top of the GNN help?

Forked from e10_triple_gnn. The open question (never measured): is the trained
GNN's policy+value head, plugged into MCTS as an evaluator (the AlphaZero-style
player), stronger or weaker than the SAME net used policy-only (PureGnn)?

Every prior tournament only ever ran PureGnn (argmax policy, no search) against
LookaheadV3 (a hand-coded VP heuristic + MCTS). No GNN-evaluator-backed MCTS bot
has ever been on the board. This experiment isolates exactly that comparison.

Seating (per rotation):
  - Slot 0: PureGnn  (--checkpoint-a, the "top" net; policy-only)
  - Slot 1: GnnMcts  (--checkpoint-b, the "top" net via GnnEvaluator -> MCTSBot)
  - Slot 2: PureGnn  (--checkpoint-c, the "second" net; policy-only)
  - Slot 3: LookaheadMctsV3

Apples-to-apples knob: --gnn-mcts-sims defaults to 200 to match LookV3's base
sims, so slot 1 vs slot 3 is "GNN evaluator vs heuristic evaluator at equal
search depth."

PERF WARNING: the GnnEvaluator runs one batch=1 forward pass per MCTS leaf
(see gnn_evaluator.py notes). At sims=200 in full Catan this slot is ~100-500x
slower per move than the PureGnn/LookV3 slots and gates wall-clock. Keep game
counts small for diagnostics. A BatchedGnnEvaluator is the deferred speedup.
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


_BASE_SEATING = ["PureGnnA", "GnnMctsB", "PureGnnC", "LookaheadMctsV3"]


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


def build_gnn_mcts_bot(game: CatanGame, model: GnnModel, sims: int,
                       seed: int, device: str = "cpu"):
    """GNN-as-MCTS-evaluator bot (the AlphaZero-style player).

    A fresh GnnEvaluator per call: it caches forward results by id(state), so
    it must not be shared across games (mirrors e6's per-game construction).
    """
    evaluator = GnnEvaluator(model=model, device=device)
    rng = np.random.default_rng(seed)
    return os_mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=sims,
        evaluator=evaluator, solve=False, random_state=rng,
    )


def _run_cell(rec: SelfPlayRecorder, rot_idx: int,
              model_a: GnnModel, model_b: GnnModel, model_c: GnnModel,
              gnn_mcts_sims: int,
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
            elif role == "GnnMctsB":
                bots[slot] = build_gnn_mcts_bot(
                    game, model=model_b, sims=gnn_mcts_sims,
                    seed=seed + 1000, device=device,
                )
            elif role == "PureGnnC":
                bots[slot] = PureGnnBot(model=model_c, device=device)
            elif role == "LookaheadMctsV3":
                bots[slot] = lookahead_v3_bot
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
     checkpoint_a, checkpoint_b, checkpoint_c,
     gnn_mcts_sims, lookahead_depth, base_sims_v3,
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
    for rot_idx in range(4):
        seeds = seeds_per_rot[rot_idx]
        _run_cell(
            rec, rot_idx, model_a, model_b, model_c,
            gnn_mcts_sims, lookahead_depth, base_sims_v3,
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
    label_a: str = "PureGnn_top",
    label_b: str = "GnnMcts_top",
    label_c: str = "PureGnn_second",
    num_games_per_seating: int = 30,
    gnn_mcts_sims: int = 200,
    lookahead_depth: int = 10,
    base_sims_v3: int = 200,
    hidden_dim: int = 128,
    num_layers: int = 4,
    seed_base: int = 19_000_000,
    max_seconds: float = 600.0,
    vp_target: int = 10,
    bonuses: bool = True,
    resume: bool = True,
    workers: int = 10,
    device: str = "auto",
) -> Path:
    out = make_run_dir(out_root, "e10e_gnn_mcts")
    base_config = {
        "experiment": "e10e_gnn_mcts",
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "checkpoint_c": str(checkpoint_c),
        "label_a": label_a,
        "label_b": label_b,
        "label_c": label_c,
        "gnn_mcts_sims": gnn_mcts_sims,
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
        model_c = _load_model(checkpoint_c, hidden_dim, num_layers, device=resolved_device)
        for rot_idx in range(4):
            seeds = [seed_base + rot_idx * 10_000 + i for i in range(num_games_per_seating)]
            _run_cell(
                rec, rot_idx, model_a, model_b, model_c,
                gnn_mcts_sims, lookahead_depth, base_sims_v3,
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
            checkpoint_a, checkpoint_b, checkpoint_c,
            gnn_mcts_sims, lookahead_depth, base_sims_v3,
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
    p.add_argument("--checkpoint-a", type=Path, required=True,
                   help="top net, used policy-only (PureGnn)")
    p.add_argument("--checkpoint-b", type=Path, required=True,
                   help="top net, used as MCTS evaluator (GnnMcts)")
    p.add_argument("--checkpoint-c", type=Path, required=True,
                   help="second net, used policy-only (PureGnn)")
    p.add_argument("--label-a", type=str, default="PureGnn_top")
    p.add_argument("--label-b", type=str, default="GnnMcts_top")
    p.add_argument("--label-c", type=str, default="PureGnn_second")
    p.add_argument("--num-games-per-seating", type=int, default=30,
                   help="120 total games at default (30 per rotation x 4 rotations)")
    p.add_argument("--gnn-mcts-sims", type=int, default=200,
                   help="MCTS sims for the GnnMcts slot; 200 matches LookV3 base")
    p.add_argument("--lookahead-depth", type=int, default=10)
    p.add_argument("--base-sims-v3", type=int, default=200)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--seed-base", type=int, default=19_000_000)
    p.add_argument("--max-seconds", type=float, default=600.0)
    p.add_argument("--vp-target", type=int, default=10)
    p.add_argument("--bonuses", action="store_true", default=True)
    p.add_argument("--no-bonuses", dest="bonuses", action="store_false")
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
        label_a=args.label_a,
        label_b=args.label_b,
        label_c=args.label_c,
        num_games_per_seating=args.num_games_per_seating,
        gnn_mcts_sims=args.gnn_mcts_sims,
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
    print(f"e10e wrote to {out}")


if __name__ == "__main__":
    cli_main()
