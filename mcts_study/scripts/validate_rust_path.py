"""Phase 9 production validation: confirm Rust self-play + arena are bit-exact
to the Python oracle on the PRODUCTION net (GnnModel 128x4, az_iter_1.pt) — not
just the toy net the gates used — and report wall-clock per game.

Run a SMALL batch (default 4 seeds) on BOTH paths from the same checkpoint/.ts
and diff the records. This is the spec §6 step-9 cross-check on the real net.

GPU-util / throughput-vs-Python measurement is reported separately by the
caller (nvidia-smi during a real run); this script does the correctness half +
timing.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg
from catan_gnn.export_torchscript import export
from catan_mcts.adapter import CatanGame
from catan_mcts.async_mcts import play_one_async_game
import catan_mcts_rs

CKPT = Path("/home/chitii/catan_data/runs/v3/az_loop/checkpoints/az_iter_1.pt")
HIDDEN, LAYERS = 128, 4


def _softmax(x):
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


class SyncEval:
    def __init__(self, model):
        self.model = model.eval()

    async def eval_leaf(self, state):
        if state.is_terminal():
            return np.asarray(state.returns(), dtype=np.float32), None
        obs = state._engine.observation()
        with torch.no_grad():
            v, logits = self.model(Batch.from_data_list([state_to_pyg(obs)]))
        value = v.squeeze(0).numpy().astype(np.float32)
        logits = logits.squeeze(0).numpy().astype(np.float32)
        legal = state.legal_actions()
        la = np.asarray(legal, dtype=np.int64)
        return value, [(int(a), float(p)) for a, p in zip(legal, _softmax(logits[la]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--n-sims", type=int, default=200)
    ap.add_argument("--self-play", action="store_true", default=True)
    ap.add_argument("--no-self-play", dest="self_play", action="store_false")
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    args = ap.parse_args()

    model = GnnModel(hidden_dim=HIDDEN, num_layers=LAYERS)
    obj = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)
    model.eval()
    ts = args.checkpoint.with_suffix(".val.ts")
    export(checkpoint=args.checkpoint, out_ts=ts, hidden_dim=HIDDEN, num_layers=LAYERS)

    ok = True
    py_total = rs_total = 0.0
    for seed in range(args.seeds):
        t0 = time.monotonic()
        py = asyncio.run(play_one_async_game(
            game=CatanGame(), seed=seed, evaluator=SyncEval(model),
            n_sims=args.n_sims, rng=np.random.default_rng(seed),
            self_play=args.self_play))
        py_total += time.monotonic() - t0

        t0 = time.monotonic()
        rs = catan_mcts_rs.run_selfplay(
            str(ts), [seed], args.n_sims, args.self_play, 10, True,
            30, 0.8, 0.25, 200_000)[0]
        rs_total += time.monotonic() - t0

        same = (rs["length_in_moves"] == py.length_in_moves
                and rs["winner"] == py.winner
                and list(rs["action_history"]) == list(py.action_history)
                and len(rs["moves"]) == len(py.moves)
                and all(list(mr["visit_counts"]) == list(mp.visit_counts)
                        and mr["action_taken"] == mp.action_taken
                        for mr, mp in zip(rs["moves"], py.moves)))
        print(f"seed {seed}: identical={same} len={py.length_in_moves} "
              f"winner={py.winner} py={py_total:.1f}s rs={rs_total:.1f}s")
        ok = ok and same

    print(f"\nPRODUCTION-NET CROSS-CHECK: {'BIT-EXACT' if ok else 'DIVERGED'} "
          f"over {args.seeds} seeds (n_sims={args.n_sims}, self_play={args.self_play})")
    print(f"wall-clock: Python {py_total:.1f}s, Rust {rs_total:.1f}s "
          f"(speedup {py_total/max(rs_total,1e-9):.1f}x, single-threaded per-state eval)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
