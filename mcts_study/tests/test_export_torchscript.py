"""Exported .ts must reproduce eager GnnModel bit-exact (spec §5.1).

The train→infer seam: torch.jit.trace of a plain-tensor wrapper around the PyG
GnnModel. Proven in the Phase-0 spike; this is the productionized, TDD'd version.
"""
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg
from catan_gnn.export_torchscript import export
from catan_mcts.adapter import CatanGame


def _states(n):
    """n distinct non-terminal observations across seeded random playouts."""
    import random
    out, seed = [], 0
    while len(out) < n:
        g = CatanGame()
        st = g.new_initial_state()
        rng = random.Random(seed)
        seed += 1
        for _ in range(rng.randrange(1, 60)):
            if st.is_terminal():
                break
            la = st.legal_actions()
            st.apply_action(la[rng.randrange(len(la))])
        if not st.is_terminal():
            out.append(st._engine.observation())
    return out


def _obs_tensors(o):
    f = lambda k: torch.from_numpy(np.ascontiguousarray(o[k], dtype=np.float32))
    return (f("hex_features"), f("vertex_features"),
            f("edge_features"), f("scalars"))


def test_exported_ts_bit_exact(tmp_path):
    # Small net for speed; the seam is architecture-agnostic.
    model = GnnModel(hidden_dim=32, num_layers=2).eval()
    ckpt = tmp_path / "m.pt"
    torch.save({"model_state": model.state_dict()}, ckpt)
    ts = tmp_path / "m.ts"
    export(checkpoint=ckpt, out_ts=ts, hidden_dim=32, num_layers=2)

    loaded = torch.jit.load(str(ts)).eval()
    max_dv = max_dl = 0.0
    for o in _states(50):
        hx, vx, ex, sc = _obs_tensors(o)
        with torch.no_grad():
            rv, rl = model(Batch.from_data_list([state_to_pyg(o)]))
            tv, tl = loaded(hx, vx, ex, sc)
        max_dv = max(max_dv, (tv - rv).abs().max().item())
        max_dl = max(max_dl, (tl - rl).abs().max().item())
    assert max_dv == 0.0 and max_dl == 0.0, f"dv={max_dv} dl={max_dl}"


def test_export_loads_plain_state_dict(tmp_path):
    """Checkpoints saved as a bare state_dict (not wrapped) also export."""
    model = GnnModel(hidden_dim=32, num_layers=2).eval()
    ckpt = tmp_path / "bare.pt"
    torch.save(model.state_dict(), ckpt)
    ts = tmp_path / "bare.ts"
    out = export(checkpoint=ckpt, out_ts=ts, hidden_dim=32, num_layers=2)
    assert out.exists()
    loaded = torch.jit.load(str(ts)).eval()
    o = _states(1)[0]
    hx, vx, ex, sc = _obs_tensors(o)
    with torch.no_grad():
        rv, rl = model(Batch.from_data_list([state_to_pyg(o)]))
        tv, tl = loaded(hx, vx, ex, sc)
    assert (tv - rv).abs().max().item() == 0.0
    assert (tl - rl).abs().max().item() == 0.0
