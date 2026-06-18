"""Phase-0 spike, step 1: can torch.jit.script(GnnModel) even succeed?

This is the make-or-break gate. GnnModel uses PyG HeteroConv + SAGEConv, whose
Python-level dict iteration is historically NOT scriptable by torch.jit.script.

We try, in order of preference:
  (A) torch.jit.script(model)         -- cleanest; identical graph
  (B) torch.jit.trace(model, example) -- fallback; fixed-topology trace

If both fail, the spec says STOP and bring the fixed-topology-net pivot to the
user. This script ONLY reports what works; it does not decide the pivot.

Run in WSL venv. Loads the real az_iter_1 checkpoint (hidden_dim=128, layers=4).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import torch

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg
from torch_geometric.data import Batch

CKPT = Path("/home/chitii/catan_data/runs/v3/az_loop/checkpoints/az_iter_1.pt")
HIDDEN_DIM, NUM_LAYERS = 128, 4


def load_model() -> GnnModel:
    model = GnnModel(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)
    return model.eval()


def make_example_batch():
    """One real state, batched (B=1), as the model.forward expects."""
    from catan_mcts.adapter import CatanGame
    game = CatanGame()
    state = game.new_initial_state()
    obs = state._engine.observation()
    print("obs shapes:", {k: getattr(v, "shape", v) for k, v in obs.items()
                          if hasattr(v, "shape")})
    data = state_to_pyg(obs)
    return Batch.from_data_list([data])


def main() -> int:
    model = load_model()
    example = make_example_batch()

    with torch.no_grad():
        ref_v, ref_logits = model(example)
    print(f"eager forward OK: value {ref_v.shape}, logits {ref_logits.shape}")

    print("\n=== (A) torch.jit.script ===")
    try:
        scripted = torch.jit.script(model)
        with torch.no_grad():
            sv, sl = scripted(example)
        dv = (sv - ref_v).abs().max().item()
        dl = (sl - ref_logits).abs().max().item()
        print(f"SCRIPT SUCCEEDED. max|dv|={dv:.3e} max|dl|={dl:.3e}")
        return 0
    except Exception:
        print("SCRIPT FAILED:")
        traceback.print_exc()

    print("\n=== (B) torch.jit.trace ===")
    try:
        traced = torch.jit.trace(model, (example,), strict=False)
        with torch.no_grad():
            tv, tl = traced(example)
        dv = (tv - ref_v).abs().max().item()
        dl = (tl - ref_logits).abs().max().item()
        print(f"TRACE SUCCEEDED. max|dv|={dv:.3e} max|dl|={dl:.3e}")
        print("NOTE: trace fixes topology; OK here since Catan graph is fixed.")
        return 0
    except Exception:
        print("TRACE FAILED:")
        traceback.print_exc()

    print("\nBOTH FAILED -> pivot decision required (see spec §5.2).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
