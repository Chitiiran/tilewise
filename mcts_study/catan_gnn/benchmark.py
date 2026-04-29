"""Bench-2: static-position evaluator comparison.

Sample n positions from the val split, run both GnnEvaluator and
LookaheadVpEvaluator(depth=D), record per-position value MAE and policy KL,
write summary JSON.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from catan_bot import _engine
from catan_mcts.evaluator import LookaheadVpEvaluator

from .dataset import CatanReplayDataset
from .gnn_model import GnnModel
from .state_to_pyg import state_to_pyg


def bench2_main(
    *,
    checkpoint: Path,
    run_dirs: list[Path],
    out_path: Path,
    n_positions: int = 1000,
    lookahead_depth: int = 25,
    seed: int = 0,
) -> Path:
    rng = random.Random(seed)
    ds = CatanReplayDataset([Path(p) for p in run_dirs])
    n_total = len(ds)
    indices = rng.sample(range(n_total), min(n_positions, n_total))

    model = GnnModel(hidden_dim=32, num_layers=2)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    look = LookaheadVpEvaluator(depth=lookahead_depth, base_seed=seed)

    value_diffs = []
    policy_kls = []
    with torch.no_grad():
        for i in indices:
            data, _, _, legal = ds[i]
            batch = Batch.from_data_list([data])
            v_gnn, p_logits = model(batch)
            v_gnn_np = v_gnn.squeeze(0).numpy()
            # Reconstruct an OpenSpiel state by replaying the engine. Use ds
            # internals: same seed + walk to move_index. We re-call the
            # dataset's own logic by accessing engine state directly.
            row = ds._index.iloc[i]
            seed_g = int(row["seed"])
            engine = _engine.Engine(seed_g)
            move_index = int(row["move_index"])
            history = ds._history_by_seed[seed_g]
            mcts_dec = 0
            for action_id in history:
                if engine.is_terminal():
                    break
                if engine.is_chance_pending():
                    a = int(action_id)
                    engine.apply_chance_outcome(a & 0x7FFF_FFFF)
                    continue
                legal_now = engine.legal_actions()
                if len(legal_now) > 1 and engine.current_player() == int(row["current_player"]):
                    if mcts_dec == move_index:
                        break
                    mcts_dec += 1
                engine.step(int(action_id))
            # Lookahead acts on a clone (mirror of LookaheadVpEvaluator behavior).
            v_look = engine.clone().lookahead_vp_value(lookahead_depth, int(seed) + i)
            v_look = np.asarray(v_look, dtype=np.float32)
            value_diffs.append(float(np.mean(np.abs(v_gnn_np - v_look))))

            # Policy KL between GNN softmaxed-over-legal and visit-count target.
            mask = legal.numpy()
            masked = np.where(mask, p_logits.squeeze(0).numpy(), -np.inf)
            z = masked - np.nanmax(masked)
            ez = np.exp(z); gnn_p = ez / ez.sum()
            visits = np.array(row["mcts_visit_counts"], dtype=np.float32)
            target_p = visits / max(1.0, visits.sum())
            # KL(target || gnn_p), summed over legal.
            eps = 1e-9
            kl = float(np.sum(target_p * (np.log(target_p + eps) - np.log(gnn_p + eps))))
            policy_kls.append(kl)

    summary = {
        "checkpoint": str(checkpoint),
        "n_positions": len(indices),
        "lookahead_depth": lookahead_depth,
        "bench2_value_mae": float(np.mean(value_diffs)),
        "bench2_policy_kl": float(np.mean(policy_kls)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    return out_path


def cli_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--out-path", type=Path, required=True)
    p.add_argument("--n-positions", type=int, default=1000)
    p.add_argument("--lookahead-depth", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    bench2_main(
        checkpoint=args.checkpoint, run_dirs=args.run_dirs,
        out_path=args.out_path, n_positions=args.n_positions,
        lookahead_depth=args.lookahead_depth, seed=args.seed,
    )


if __name__ == "__main__":
    cli_main()
