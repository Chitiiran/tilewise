# tests/test_self_play_async.py
import numpy as np
import torch
import pandas as pd
from catan_gnn.gnn_model import GnnModel
from catan_mcts.experiments.self_play_async import run_self_play


def _save_ckpt(tmp_path):
    torch.manual_seed(0)
    m = GnnModel(hidden_dim=8, num_layers=2)
    p = tmp_path / "ckpt.pt"
    torch.save({"model_state": m.state_dict()}, p)
    return p


def test_self_play_writes_valid_parquet(tmp_path):
    ckpt = _save_ckpt(tmp_path)
    out = run_self_play(
        out_root=tmp_path / "runs", checkpoint=ckpt, num_games=4, n_sims=4,
        n_concurrent=4, hidden_dim=8, num_layers=2, vp_target=10, bonuses=True,
        device="cpu", max_batch=4, window_ms=5, seed_base=5_000_000)
    assert out.exists()
    parquets = list(out.rglob("games*.parquet"))
    assert parquets
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    assert len(df) == 4
    assert {"seed", "winner", "length_in_moves"}.issubset(df.columns)


def test_resume_skips_done_seeds(tmp_path):
    ckpt = _save_ckpt(tmp_path)
    common = dict(checkpoint=ckpt, n_sims=4, n_concurrent=2, hidden_dim=8,
                  num_layers=2, vp_target=10, bonuses=True, device="cpu",
                  max_batch=2, window_ms=5, seed_base=5_100_000)
    out_root = tmp_path / "runs"
    out = run_self_play(out_root=out_root, num_games=2, **common)
    out2 = run_self_play(out_root=out_root, num_games=4, resume_dir=out, **common)
    df = pd.concat([pd.read_parquet(p) for p in out2.rglob("games*.parquet")],
                   ignore_index=True)
    assert df["seed"].nunique() == 4
    assert len(df) == 4
