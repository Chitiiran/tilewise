"""2-iteration micro run proving candidate self-play (spec §4). iter-1 uses the
champion; iter-2 uses iter-1's candidate. Needs CUDA (self_play_async)."""
from __future__ import annotations

import glob
import json

import pytest

pytestmark = pytest.mark.slow


def test_two_iter_micro_candidate_selfplay(tmp_path):
    import torch

    from catan_az import daily
    from catan_az.config import AzConfig
    from catan_az.ladder import Ladder
    from catan_gnn.gnn_model import GnnModel

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA (self_play_async --device cuda)")

    champ = tmp_path / "scratch.pt"
    torch.save(GnnModel(hidden_dim=32, num_layers=2).state_dict(), champ)
    Ladder(tmp_path, champion_checkpoint=str(champ), champion_name="scratch")

    cfg = AzConfig(games_per_iter=4, sims=4, n_concurrent=4, max_batch=4,
                   max_epochs=1, arena_games=4, arena_sims=4, arena_min_decisive=1,
                   arena_max_draw_rate=1.0, vp_target=5, bonuses=False,
                   hidden_dim=32, num_layers=2, archive_root=str(tmp_path / "hdd"),
                   max_iters_per_model=10)
    (tmp_path / "hdd").mkdir()

    daily.run_day(cfg, loop_root=tmp_path, capped_procs=1,
                  cycle_fn=daily.run_cycle, max_iters=2)

    metas = [json.loads(open(m).read()) for m in
             glob.glob(str(tmp_path / "iter_*/selfplay/*/meta.json"))]
    gens = {m["generator_name"] for m in metas}
    assert "scratch" in gens                              # iter-1 used champion
    assert any(g.startswith("cand_iter_1") for g in gens)  # iter-2 used candidate
    # every dir is gen_iter-tagged (the field whose absence was the bug)
    assert all("gen_iter" in m for m in metas)
    # PROGRESS.md shows the generators + new_games (not 0)
    prog = (tmp_path / "PROGRESS.md").read_text()
    assert "cand_iter_1" in prog
