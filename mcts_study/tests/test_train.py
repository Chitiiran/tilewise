"""Tests for catan_gnn.train: produces a checkpoint and training_log.json."""
import json
from pathlib import Path

import pytest
import torch

from catan_gnn.train import train_main


def _make_minimal_run(tmp_path: Path):
    from catan_mcts.experiments.e1_winrate_vs_random import main
    out = main(
        out_root=tmp_path, num_games=2,
        sims_per_move_grid=[2], seed_base=12345,
        max_seconds=300.0,
    )
    return out


def test_train_produces_artifacts(tmp_path: Path):
    """End-to-end: run a tiny training job, get a checkpoint and a log."""
    run_dir = _make_minimal_run(tmp_path / "run")
    out_dir = tmp_path / "gnn_v0"
    train_main(
        run_dirs=[run_dir],
        out_dir=out_dir,
        hidden_dim=32, num_layers=2,
        epochs=2, batch_size=4, lr=1e-3,
        w_value=1.0, w_policy=1.0,
        seed=0,
    )
    assert (out_dir / "checkpoint.pt").exists()
    assert (out_dir / "training_log.json").exists()
    assert (out_dir / "config.json").exists()
    log = json.loads((out_dir / "training_log.json").read_text())
    assert "epochs" in log
    assert len(log["epochs"]) == 2
    for ep in log["epochs"]:
        for k in ("epoch", "train_loss_total", "val_loss_total",
                 "val_value_mae", "val_policy_top1_acc"):
            assert k in ep, f"missing key {k} in epoch row"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_checkpoint_is_loadable(tmp_path: Path, device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    run_dir = _make_minimal_run(tmp_path / "run")
    out_dir = tmp_path / "gnn_v0"
    train_main(
        run_dirs=[run_dir], out_dir=out_dir,
        hidden_dim=32, num_layers=2,
        epochs=1, batch_size=4, lr=1e-3,
        w_value=1.0, w_policy=1.0, seed=0,
        device=device,
    )
    from catan_gnn.gnn_model import GnnModel
    model = GnnModel(hidden_dim=32, num_layers=2)
    state = torch.load(out_dir / "checkpoint.pt", map_location="cpu")
    model.load_state_dict(state)
