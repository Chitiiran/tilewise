"""Daily driver: manifest, run_day loop, fresh-ratio launcher, cycle, stagnation
(spec 2026-06-13 §4-5, §7). Stage fns injected/monkeypatched — no GPU."""
from __future__ import annotations


def test_manifest_round_trip(tmp_path):
    from catan_az.daily import DailyManifest
    m = DailyManifest(iter=3, stage="selfplay", champion="az_iter_1",
                      fresh_target=700, fresh_done=120, rules_id="v3-full")
    m.save(tmp_path)
    back = DailyManifest.load(tmp_path)
    assert back == m


def test_run_day_runs_until_stop(tmp_path):
    from catan_az.daily import run_day
    from catan_az.config import AzConfig
    calls = []

    def fake_cycle(cfg, loop_root, iter_n, capped_procs):
        calls.append(iter_n)
        if iter_n == 2:
            (loop_root / "STOP").write_text("")
        return "promote"

    run_day(AzConfig(), loop_root=tmp_path, capped_procs=5,
            cycle_fn=fake_cycle, max_iters=10)
    assert calls == [1, 2]


def test_run_day_respects_max_iters(tmp_path):
    from catan_az.daily import run_day
    from catan_az.config import AzConfig
    calls = []
    run_day(AzConfig(), loop_root=tmp_path, capped_procs=5,
            cycle_fn=lambda cfg, root, n, procs: calls.append(n) or "hold",
            max_iters=3)
    assert len(calls) == 3


def test_generate_fresh_computes_deficit_and_launches(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import catan_az.daily as daily
    from catan_az.config import AzConfig
    launched = {}

    def fake_launch(cfg, out_dir, checkpoint, n_games, n_procs, champion, rules_id):
        launched["n_games"] = n_games
        d = out_dir / "run1"
        d.mkdir(parents=True)
        pd.DataFrame({"seed": range(n_games), "winner": [0] * n_games}
                     ).to_parquet(d / "games.x.parquet")
        (d / "meta.json").write_text(json.dumps({"rules_id": rules_id,
                                                "champion": champion}))
        return [d]

    monkeypatch.setattr(daily, "_launch_selfplay_procs", fake_launch)
    cfg = AzConfig(window_games=1000, fresh_ratio=0.70)
    dirs = daily.generate_fresh(cfg, iter_dir=tmp_path / "iter_1",
                                champion="az_iter_1",
                                champion_ckpt=tmp_path / "c.pt",
                                capped_procs=5, prior_dirs=[])
    assert launched["n_games"] == 700
    assert len(dirs) == 1


def test_generate_fresh_skips_when_target_met(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import catan_az.daily as daily
    from catan_az.config import AzConfig
    # prior dir already has 800 fresh az_iter_1 games
    p = tmp_path / "prior"
    p.mkdir()
    pd.DataFrame({"seed": range(800), "winner": [0] * 800}).to_parquet(p / "games.x.parquet")
    (p / "meta.json").write_text(json.dumps({"rules_id": "v3-full", "champion": "az_iter_1"}))
    called = {"launched": False}
    monkeypatch.setattr(daily, "_launch_selfplay_procs",
                        lambda *a, **k: called.__setitem__("launched", True) or [])
    cfg = AzConfig(window_games=1000, fresh_ratio=0.70)
    dirs = daily.generate_fresh(cfg, iter_dir=tmp_path / "iter_2",
                                champion="az_iter_1", champion_ckpt=tmp_path / "c.pt",
                                capped_procs=5, prior_dirs=[p])
    assert dirs == [] and called["launched"] is False
