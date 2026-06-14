"""gen_iter-tagged self-play + own-iter quota (spec §4 step 2)."""
from __future__ import annotations

import json

import pandas as pd


def test_generate_iter_games_tags_gen_iter_and_quota(tmp_path, monkeypatch):
    import catan_az.daily as daily
    from catan_az.config import AzConfig
    launched = {}

    def fake_launch(cfg, out_dir, checkpoint, n_games, n_procs,
                    generator_name, gen_iter, rules_id):
        launched["n"] = n_games
        launched["gi"] = gen_iter
        d = out_dir / "run1"
        d.mkdir(parents=True)
        pd.DataFrame({"seed": range(n_games), "winner": [0] * n_games}
                     ).to_parquet(d / "games.x.parquet")
        (d / "meta.json").write_text(json.dumps(
            {"rules_id": rules_id, "generator_name": generator_name,
             "gen_iter": gen_iter}))
        return [d]

    monkeypatch.setattr(daily, "_launch_selfplay_procs", fake_launch)
    cfg = AzConfig(games_per_iter=1000)
    dirs = daily.generate_iter_games(cfg, iter_dir=tmp_path / "iter_5",
                                     generator=("cand_iter_4", tmp_path / "c.pt"),
                                     gen_iter=5, capped_procs=5, prior_dirs=[])
    assert launched["n"] == 1000 and launched["gi"] == 5
    assert len(dirs) == 1


def test_generate_iter_resumes_only_own_deficit(tmp_path, monkeypatch):
    """Existing games from a DIFFERENT gen_iter do NOT reduce this iter's quota
    (the exact bug). Only this iter's own gen_iter games do."""
    import catan_az.daily as daily
    from catan_az.config import AzConfig
    p = tmp_path / "old"
    p.mkdir()
    pd.DataFrame({"seed": range(900), "winner": [0] * 900}).to_parquet(p / "games.x.parquet")
    (p / "meta.json").write_text(json.dumps(
        {"rules_id": "v3-full", "generator_name": "x", "gen_iter": 3}))
    asked = {}

    def fake_launch(cfg, out_dir, checkpoint, n_games, n_procs,
                    generator_name, gen_iter, rules_id):
        asked["n"] = n_games
        d = out_dir / "r"
        d.mkdir(parents=True)
        pd.DataFrame({"seed": range(n_games), "winner": [0] * n_games}).to_parquet(d / "games.g.parquet")
        (d / "meta.json").write_text(json.dumps(
            {"rules_id": rules_id, "generator_name": generator_name, "gen_iter": gen_iter}))
        return [d]

    monkeypatch.setattr(daily, "_launch_selfplay_procs", fake_launch)
    cfg = AzConfig(games_per_iter=1000)
    daily.generate_iter_games(cfg, iter_dir=tmp_path / "iter_5",
                              generator=("cand_iter_4", tmp_path / "c.pt"),
                              gen_iter=5, capped_procs=5, prior_dirs=[p])
    assert asked["n"] == 1000   # full quota, NOT 100 (old gen_iter=3 ignored)
