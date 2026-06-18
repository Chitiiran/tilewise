"""Phase 8 (Task 9): _default_train emits a TorchScript .ts beside the best
checkpoint, loadable + bit-exact to the trained net.

train_main is monkeypatched to write a checkpoint quickly (the export wiring is
what's under test, not training).
"""
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg
from catan_mcts.adapter import CatanGame
from catan_az import loop as az_loop
from catan_az.config import AzConfig


def test_default_train_emits_loadable_ts(tmp_path, monkeypatch):
    cfg = AzConfig(hidden_dim=32, num_layers=2)
    iter_dir = tmp_path / "iter_1"

    # Fake train_main: just write checkpoint_best.pt with a small net.
    torch.manual_seed(7)
    model = GnnModel(hidden_dim=32, num_layers=2).eval()

    def fake_train_main(*, run_dirs, out_dir, **kwargs):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict()}, out_dir / "checkpoint_best.pt")

    monkeypatch.setattr("catan_gnn.train.train_main", fake_train_main)

    best = az_loop._default_train(cfg, run_dirs=[tmp_path / "sp"], iter_dir=iter_dir,
                                  init_ckpt=str(tmp_path / "init.pt"))
    ts = best.with_suffix(".ts")
    assert ts.exists(), "TorchScript .ts not emitted beside checkpoint_best.pt"

    # Bit-exact vs eager on a real state.
    loaded = torch.jit.load(str(ts)).eval()
    o = CatanGame().new_initial_state()._engine.observation()
    f = lambda k: torch.from_numpy(np.ascontiguousarray(o[k], dtype=np.float32))
    with torch.no_grad():
        rv, rl = model(Batch.from_data_list([state_to_pyg(o)]))
        tv, tl = loaded(f("hex_features"), f("vertex_features"),
                        f("edge_features"), f("scalars"))
    assert (tv - rv).abs().max().item() == 0.0
    assert (tl - rl).abs().max().item() == 0.0
