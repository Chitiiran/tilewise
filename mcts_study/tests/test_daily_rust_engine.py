"""Phase 8 (Task 9): the cfg.engine switch routes self-play + arena to the Rust
path, and the Rust run_arena (production entry) agrees with the Python arena on
a small plan.
"""
import json
from pathlib import Path

import pytest
import torch

from catan_gnn.gnn_model import GnnModel
from catan_gnn.export_torchscript import export
from catan_az.config import AzConfig
from catan_az import arena as az_arena

catan_mcts_rs = pytest.importorskip("catan_mcts_rs")


def test_config_engine_defaults_rust():
    # Flipped to "rust" 2026-06-18 after the Phase-9 production cross-check
    # passed bit-exact. "python" remains a valid fallback.
    assert AzConfig().engine == "rust"


def test_daily_selfplay_module_switch(monkeypatch):
    """_launch_selfplay_procs uses the rust module iff cfg.engine == 'rust'."""
    from catan_az import daily

    captured = {}

    class _FakePopen:
        def __init__(self, cmd, **kw):
            captured["cmd"] = cmd

        def wait(self, timeout=None):
            return 0

        @property
        def returncode(self):
            return 0

    monkeypatch.setattr(daily.subprocess, "Popen", _FakePopen)
    # Avoid the post-run dir glob doing real work.
    monkeypatch.setattr(daily, "make_run_dir", lambda *a, **k: Path("."), raising=False)

    for engine, expect in (("python", "self_play_async"), ("rust", "self_play_rust")):
        cfg = AzConfig(engine=engine, worker_nice=10, sims=8, n_concurrent=4,
                       max_batch=8, hidden_dim=32, num_layers=2, vp_target=10)
        try:
            daily._launch_selfplay_procs(
                cfg, out_dir=Path("/tmp/sp_switch"), checkpoint=Path("x.pt"),
                n_games=1, n_procs=1, generator_name="g", gen_iter=0,
                rules_id="v3-full")
        except Exception:
            pass  # we only care about the captured cmd
        assert any(expect in str(c) for c in captured["cmd"]), \
            f"engine={engine} should invoke {expect}, got {captured['cmd']}"


def test_run_arena_rust_matches_python(tmp_path):
    """run_arena(engine='rust') vs run_arena(engine='python') on the same small
    plan: identical winrate + wins (the production-path version of the gate)."""
    # Two small nets.
    paths = {}
    for label, seed in (("cand", 111), ("champ", 222)):
        torch.manual_seed(seed)
        m = GnnModel(hidden_dim=32, num_layers=2).eval()
        ck = tmp_path / f"{label}.pt"
        torch.save({"model_state": m.state_dict()}, ck)
        export(checkpoint=ck, out_ts=ck.with_suffix(".ts"),
               hidden_dim=32, num_layers=2)
        paths[label] = ck

    base = dict(arena_games=8, arena_sims=8, hidden_dim=32, num_layers=2,
                vp_target=10, bonuses=True, arena_max_draw_rate=1.0,
                arena_min_decisive=0, promote_threshold=0.55)
    cfg_rust = AzConfig(engine="rust", **base)
    cfg_py = AzConfig(engine="python", **base)

    res_rust = az_arena.run_arena(
        candidate_ckpt=paths["cand"], champion_ckpt=paths["champ"],
        cfg=cfg_rust, out_dir=tmp_path / "ar_rust", seed_base=30_000_000)
    res_py = az_arena.run_arena(
        candidate_ckpt=paths["cand"], champion_ckpt=paths["champ"],
        cfg=cfg_py, out_dir=tmp_path / "ar_py", seed_base=30_000_000,
        n_concurrent=4)

    assert res_rust.wins_cand == res_py.wins_cand
    assert res_rust.wins_champ == res_py.wins_champ
    assert res_rust.draws == res_py.draws
    assert res_rust.winrate_cand == res_py.winrate_cand
