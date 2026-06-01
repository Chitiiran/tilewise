"""Smoke test for e10e_async: one game per seating rotation, CPU, tiny model."""
import torch
from catan_gnn.gnn_model import GnnModel
from catan_mcts.experiments.e10e_async import run_e10e_async


def test_e10e_async_one_game_per_seating_runs(tmp_path):
    ckpts = []
    for i in range(3):
        torch.manual_seed(i)
        m = GnnModel(hidden_dim=8, num_layers=2)
        p = tmp_path / f"ckpt_{i}.pt"
        torch.save({"model_state": m.state_dict()}, p)
        ckpts.append(p)

    out = run_e10e_async(
        out_root=tmp_path / "runs",
        checkpoint_a=ckpts[0],
        checkpoint_b=ckpts[1],
        checkpoint_c=ckpts[2],
        num_games_per_seating=1,
        gnn_mcts_sims=4,
        base_sims_v3=50,
        lookahead_depth=2,
        hidden_dim=8,
        num_layers=2,
        vp_target=10,
        bonuses=True,
        device="cpu",
        max_batch=4,
        window_ms=5,
        n_concurrent=4,
        seed_base=21_000_000,
    )

    assert out.exists(), f"run dir not created: {out}"
    parquets = list(out.rglob("games*.parquet"))
    assert parquets, f"no games parquet written under {out}"
