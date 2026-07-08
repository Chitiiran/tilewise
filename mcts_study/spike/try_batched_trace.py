"""Task-10 spike: can we trace a BATCHED plain-tensor wrapper that generalizes
across batch sizes B, bit-exact to PyG Batch.from_data_list?

The wrapper takes the standard PyG-batched tensors as inputs:
  hex_x [B*19, 8], vertex_x [B*54,13], edge_x [B*72,6], scalars [B,59],
  + the four batched edge_index tensors and the three per-type batch vectors.
Rust will build these from B observations (fixed topology => deterministic
offsets). Trace once at B=2, verify bit-exact at B=1,3,5 vs eager.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg, _H2V_EI, _V2H_EI, _V2E_EI, _E2V_EI
from catan_mcts.adapter import CatanGame

HIDDEN, LAYERS = 32, 2
NH, NV, NE = 19, 54, 72


class BatchWrapper(nn.Module):
    """forward takes fully-batched PyG tensors -> (value[B,4], logits[B,280])."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hex_x, vertex_x, edge_x, scalars,
                h2v, v2h, v2e, e2v, hb, vb, eb):
        data = HeteroData()
        data["hex"].x = hex_x
        data["vertex"].x = vertex_x
        data["edge"].x = edge_x
        data["hex", "to", "vertex"].edge_index = h2v
        data["vertex", "to", "hex"].edge_index = v2h
        data["vertex", "to", "edge"].edge_index = v2e
        data["edge", "to", "vertex"].edge_index = e2v
        data["hex"].batch = hb
        data["vertex"].batch = vb
        data["edge"].batch = eb
        data.scalars = scalars
        return self.model(data)


def batched_inputs(obss):
    """Build the batched tensors for a list of observations (fixed topology)."""
    B = len(obss)
    hx = torch.cat([torch.from_numpy(np.ascontiguousarray(o["hex_features"], np.float32)) for o in obss])
    vx = torch.cat([torch.from_numpy(np.ascontiguousarray(o["vertex_features"], np.float32)) for o in obss])
    ex = torch.cat([torch.from_numpy(np.ascontiguousarray(o["edge_features"], np.float32)) for o in obss])
    sc = torch.stack([torch.from_numpy(np.ascontiguousarray(o["scalars"], np.float32)) for o in obss])
    # offset edge_index per graph
    def off(ei, src_n, dst_n, count_dim):
        cols = []
        for g in range(B):
            e = ei.clone()
            e[0] += g * src_n
            e[1] += g * dst_n
            cols.append(e)
        return torch.cat(cols, dim=1)
    h2v = off(_H2V_EI, NH, NV, 0)
    v2h = off(_V2H_EI, NV, NH, 0)
    v2e = off(_V2E_EI, NV, NE, 0)
    e2v = off(_E2V_EI, NE, NV, 0)
    hb = torch.repeat_interleave(torch.arange(B), NH)
    vb = torch.repeat_interleave(torch.arange(B), NV)
    eb = torch.repeat_interleave(torch.arange(B), NE)
    return (hx, vx, ex, sc, h2v, v2h, v2e, e2v, hb, vb, eb)


def states(n, seed0=0):
    import random
    out, s = [], seed0
    while len(out) < n:
        g = CatanGame(); st = g.new_initial_state(); rng = random.Random(s); s += 1
        for _ in range(rng.randrange(1, 60)):
            if st.is_terminal(): break
            la = st.legal_actions(); st.apply_action(la[rng.randrange(len(la))])
        if not st.is_terminal(): out.append(st._engine.observation())
    return out


def main():
    torch.manual_seed(7)
    model = GnnModel(hidden_dim=HIDDEN, num_layers=LAYERS).eval()
    wrap = BatchWrapper(model).eval()

    # Trace at B=2.
    ex2 = batched_inputs(states(2, 0))
    traced = torch.jit.trace(wrap, ex2, strict=True)

    # Approach A FAILED (variable-B trace freezes dim_size). Approach B: FIXED
    # B_MAX. Trace at B_MAX; to eval k<=B_MAX real graphs, pad with (B_MAX-k)
    # copies of the first graph, take the first k outputs. dim_size is then
    # always B_MAX so nothing is frozen wrong.
    B_MAX = 4
    pad_traced = torch.jit.trace(wrap, batched_inputs(states(B_MAX, 0)), strict=True)
    worst = 0.0
    for k in (1, 2, 3, 4):
        obss = states(k, 200 + k)
        padded = obss + [obss[0]] * (B_MAX - k)  # pad to B_MAX
        inp = batched_inputs(padded)
        with torch.no_grad():
            ev, el = model(Batch.from_data_list([state_to_pyg(o) for o in obss]))
            tv, tl = pad_traced(*inp)
        # compare only the first k outputs
        dv = (tv[:k] - ev).abs().max().item()
        dl = (tl[:k] - el).abs().max().item()
        worst = max(worst, dv, dl)
        print(f"k={k} (padded to {B_MAX}): max|dv|={dv:.3e} max|dl|={dl:.3e}")
    print("FIXED-Bmax PADDED TRACE",
          "GENERALIZES BIT-EXACT" if worst == 0.0 else f"FAILS (worst {worst})")

    # THE REAL QUESTION: does a B=k batched forward equal k separate B=1
    # forwards (the path the gates/records used)? If not bit-exact, batching
    # changes decisions and is unsafe without mitigation.
    print("\n=== batched-B=k vs k separate B=1 forwards ===")
    from spike.try_trace_wrapper import TensorWrapper  # the B=1 wrapper
    w1 = TensorWrapper(model).eval()
    worst_b1 = 0.0
    for B in (2, 3, 4):
        obss = states(B, 300 + B)
        with torch.no_grad():
            bv, bl = traced(*batched_inputs(obss)) if B == 2 else pad_traced(
                *batched_inputs(obss + [obss[0]] * (B_MAX - B)))
            for i, o in enumerate(obss):
                f = lambda k: torch.from_numpy(np.ascontiguousarray(o[k], np.float32))
                sv, sl = w1(f("hex_features"), f("vertex_features"),
                            f("edge_features"), f("scalars"))
                dv = (bv[i] - sv[0]).abs().max().item()
                dl = (bl[i] - sl[0]).abs().max().item()
                worst_b1 = max(worst_b1, dv, dl)
        print(f"B={B}: worst vs-B=1 so far max={worst_b1:.3e}")
    print("BATCH vs B=1:",
          "BIT-EXACT" if worst_b1 == 0.0 else f"DIFFERS by {worst_b1:.3e} (decisions may flip)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
