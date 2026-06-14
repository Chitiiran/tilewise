"""Micro daily cycle end-to-end through the REAL stages (spec §10).

Proves the daily plumbing: preflight -> generate_fresh (real self_play_async)
-> run_iteration (train+arena+publish) -> archive. Tiny everything; needs CUDA
(self_play_async defaults --device cuda)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


def test_micro_daily_cycle(tmp_path):
    import torch

    from catan_az import daily
    from catan_az.config import AzConfig
    from catan_az.ladder import Ladder
    from catan_gnn.gnn_model import GnnModel

    if not torch.cuda.is_available():
        pytest.skip("micro daily needs CUDA (self_play_async --device cuda)")

    champ = tmp_path / "scratch.pt"
    torch.save(GnnModel(hidden_dim=32, num_layers=2).state_dict(), champ)
    Ladder(tmp_path, champion_checkpoint=str(champ), champion_name="scratch")

    cfg = AzConfig(window_games=4, fresh_ratio=0.5, sims=4, n_concurrent=4,
                   max_batch=4, max_epochs=1, arena_games=4, arena_sims=4,
                   vp_target=5, bonuses=False, hidden_dim=32, num_layers=2,
                   arena_min_decisive=1, arena_max_draw_rate=1.0,
                   archive_root=str(tmp_path / "hdd"))
    (tmp_path / "hdd").mkdir()

    daily.run_day(cfg, loop_root=tmp_path, capped_procs=1,
                  cycle_fn=daily.run_cycle, max_iters=1)

    assert (tmp_path / "journal.csv").exists()
    assert (tmp_path / "daily_state.json").exists()
    manifest = daily.DailyManifest.load(tmp_path)
    assert manifest.stage == "done"
