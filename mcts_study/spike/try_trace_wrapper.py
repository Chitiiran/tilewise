"""Phase-0 spike, step 2: trace a plain-tensor WRAPPER around GnnModel.

torch.jit.script fails (PyG HeteroConv varargs). torch.jit.trace on the raw
model fails because its input is a PyG HeteroDataBatch (tracer can't infer the
container type).

Catan's graph is FIXED-topology (19 hex / 54 vert / 72 edge, fixed edge_index).
So we can wrap the model in an nn.Module whose forward takes PLAIN TENSORS
(the three node-feature tensors + the scalar vector), rebuilds the single-graph
HeteroData internally from the constant edge_index, and returns plain tensors.

trace records the concrete ops executed for the fixed topology -> the PyG
dict-iteration unrolls into a static op graph. If THIS traces and round-trips
bit-exact, we avoid the full fixed-topology rewrite: the seam becomes
"plain tensors in, plain tensors out", which is exactly what tch-rs wants.

B=1 only (one leaf at a time) is the minimum viable seam; batching is a
follow-up once the seam is proven. We test B=1 here.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import (
    _H2V_EI, _V2H_EI, _V2E_EI, _E2V_EI, state_to_pyg,
)
from torch_geometric.data import HeteroData, Batch

CKPT = Path("/home/chitii/catan_data/runs/v3/az_loop/checkpoints/az_iter_1.pt")
HIDDEN_DIM, NUM_LAYERS = 128, 4


def load_model() -> GnnModel:
    model = GnnModel(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)
    return model.eval()


class TensorWrapper(nn.Module):
    """Plain-tensor seam over GnnModel for a SINGLE graph (B=1).

    forward(hex_x[19,8], vertex_x[54,13], edge_x[72,6], scalars[59])
        -> (value[1,4], policy_logits[1,280])

    The fixed edge_index tensors are registered as buffers so they travel with
    the traced module (and so trace records them as constants).
    """

    def __init__(self, model: GnnModel) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("h2v", _H2V_EI.clone())
        self.register_buffer("v2h", _V2H_EI.clone())
        self.register_buffer("v2e", _V2E_EI.clone())
        self.register_buffer("e2v", _E2V_EI.clone())

    def forward(self, hex_x, vertex_x, edge_x, scalars):
        data = HeteroData()
        data["hex"].x = hex_x
        data["vertex"].x = vertex_x
        data["edge"].x = edge_x
        data["hex", "to", "vertex"].edge_index = self.h2v
        data["vertex", "to", "hex"].edge_index = self.v2h
        data["vertex", "to", "edge"].edge_index = self.v2e
        data["edge", "to", "vertex"].edge_index = self.e2v
        data.scalars = scalars.view(1, -1)
        # B=1 batch vector: every node maps to graph 0.
        data["hex"].batch = torch.zeros(hex_x.size(0), dtype=torch.long)
        data["vertex"].batch = torch.zeros(vertex_x.size(0), dtype=torch.long)
        data["edge"].batch = torch.zeros(edge_x.size(0), dtype=torch.long)
        v, logits = self.model(data)
        return v, logits


def real_state_tensors():
    from catan_mcts.adapter import CatanGame
    game = CatanGame()
    state = game.new_initial_state()
    obs = state._engine.observation()
    hx = torch.from_numpy(np.ascontiguousarray(obs["hex_features"], dtype=np.float32))
    vx = torch.from_numpy(np.ascontiguousarray(obs["vertex_features"], dtype=np.float32))
    ex = torch.from_numpy(np.ascontiguousarray(obs["edge_features"], dtype=np.float32))
    sc = torch.from_numpy(np.ascontiguousarray(obs["scalars"], dtype=np.float32))
    return hx, vx, ex, sc, obs


def main() -> int:
    model = load_model()
    wrapper = TensorWrapper(model).eval()
    hx, vx, ex, sc, obs = real_state_tensors()

    # Reference via the ORIGINAL path (state_to_pyg + Batch), to prove the
    # wrapper itself is faithful before we even trace.
    with torch.no_grad():
        ref_v, ref_logits = model(Batch.from_data_list([state_to_pyg(obs)]))
        wv, wl = wrapper(hx, vx, ex, sc)
    print(f"wrapper vs original eager: max|dv|={(wv-ref_v).abs().max():.3e} "
          f"max|dl|={(wl-ref_logits).abs().max():.3e}")

    print("\n=== trace the wrapper ===")
    try:
        traced = torch.jit.trace(wrapper, (hx, vx, ex, sc), strict=True)
    except Exception:
        print("TRACE FAILED:")
        traceback.print_exc()
        return 1

    with torch.no_grad():
        tv, tl = traced(hx, vx, ex, sc)
    dv = (tv - ref_v).abs().max().item()
    dl = (tl - ref_logits).abs().max().item()
    print(f"TRACE SUCCEEDED. max|dv|={dv:.3e} max|dl|={dl:.3e}")

    # Critical: does the traced graph generalize to OTHER states (different
    # features, SAME topology)? Trace bakes in tensor *values* it saw as
    # constants -- we must prove features still flow through, not get frozen.
    from catan_mcts.adapter import CatanGame
    game = CatanGame()
    st = game.new_initial_state()
    # advance a few chance/early moves to get a different observation
    import random
    rng = random.Random(123)
    for _ in range(8):
        if st.is_terminal():
            break
        la = st.legal_actions()
        st.apply_action(la[rng.randrange(len(la))])
    obs2 = st._engine.observation()
    hx2 = torch.from_numpy(np.ascontiguousarray(obs2["hex_features"], dtype=np.float32))
    vx2 = torch.from_numpy(np.ascontiguousarray(obs2["vertex_features"], dtype=np.float32))
    ex2 = torch.from_numpy(np.ascontiguousarray(obs2["edge_features"], dtype=np.float32))
    sc2 = torch.from_numpy(np.ascontiguousarray(obs2["scalars"], dtype=np.float32))
    with torch.no_grad():
        ref_v2, ref_l2 = model(Batch.from_data_list([state_to_pyg(obs2)]))
        tv2, tl2 = traced(hx2, vx2, ex2, sc2)
    dv2 = (tv2 - ref_v2).abs().max().item()
    dl2 = (tl2 - ref_l2).abs().max().item()
    print(f"GENERALIZATION (different state, same topo): "
          f"max|dv|={dv2:.3e} max|dl|={dl2:.3e}")
    if dv2 > 1e-5 or dl2 > 1e-5:
        print("!! traced graph does NOT generalize -- features were frozen. "
              "Fixed-topology rewrite needed.")
        return 1

    out = Path("spike/wrapper_traced.ts")
    traced.save(str(out))
    print(f"\nsaved traced module -> {out}")
    print("SPIKE STEP 2 PASS (Python side). Next: load in tch-rs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
