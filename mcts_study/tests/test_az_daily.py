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


def test_run_cycle_generates_then_runs_iteration(tmp_path, monkeypatch):
    import catan_az.daily as daily
    from catan_az.config import AzConfig
    seen = {}

    def fake_gen(cfg, **k):
        seen["gen"] = True
        return [tmp_path / "sp"]

    monkeypatch.setattr(daily, "generate_fresh", fake_gen)
    monkeypatch.setattr(daily, "_all_selfplay_dirs", lambda root: [])

    def fake_run_iteration(cfg, loop_root, iter_n, *, existing_selfplay_dirs):
        seen["iter"] = iter_n
        seen["dirs"] = existing_selfplay_dirs
        return "promote"

    monkeypatch.setattr(daily, "run_iteration", fake_run_iteration)
    monkeypatch.setattr(daily, "_champion_from_ladder",
                        lambda root: ("az_iter_1", str(tmp_path / "c.pt")))
    monkeypatch.setattr(daily, "archive_out_of_window", lambda **k: 0)
    monkeypatch.setattr(daily, "select_window", lambda *a, **k: [])
    v = daily.run_cycle(AzConfig(), tmp_path, 3, capped_procs=5)
    assert v == "promote" and seen["iter"] == 3 and seen["gen"] is True


def test_run_cycle_archives_after_publish(tmp_path, monkeypatch):
    import catan_az.daily as daily
    from catan_az.config import AzConfig
    called = {}
    monkeypatch.setattr(daily, "generate_fresh", lambda cfg, **k: [])
    monkeypatch.setattr(daily, "run_iteration", lambda *a, **k: "promote")
    monkeypatch.setattr(daily, "_champion_from_ladder", lambda r: ("c", "/c.pt"))
    monkeypatch.setattr(daily, "_all_selfplay_dirs", lambda r: [])
    monkeypatch.setattr(daily, "select_window", lambda *a, **k: [])
    monkeypatch.setattr(daily, "archive_out_of_window",
                        lambda **k: called.setdefault("archived", True) or 0)
    daily.run_cycle(AzConfig(), tmp_path, 1, capped_procs=5)
    assert called.get("archived") is True


def test_stagnation_holds_from_journal(tmp_path):
    from catan_az.daily import stagnation_holds_from_journal
    import csv
    p = tmp_path / "journal.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "verdict"])
        w.writeheader()
        for i, v in enumerate(["promote", "hold", "hold", "hold"], 1):
            w.writerow({"iter": i, "verdict": v})
    assert stagnation_holds_from_journal(p) == 3


def test_run_day_stops_on_stagnation(tmp_path):
    import csv
    from catan_az.daily import run_day
    from catan_az.config import AzConfig
    # pre-seed journal with 4 holds so the FIRST cycle trips the 5-hold guard
    # after writing its own hold.
    jp = tmp_path / "journal.csv"
    with open(jp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "verdict"]); w.writeheader()
        for i in range(1, 5):
            w.writerow({"iter": i, "verdict": "hold"})
    calls = []

    def fake_cycle(cfg, loop_root, iter_n, capped_procs):
        calls.append(iter_n)
        with open(jp, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=["iter", "verdict"]).writerow(
                {"iter": iter_n, "verdict": "hold"})
        return "hold"

    run_day(AzConfig(), loop_root=tmp_path, capped_procs=5,
            cycle_fn=fake_cycle, max_iters=10, next_iter=5)
    assert len(calls) == 1   # stopped after first cycle (5 trailing holds)


def test_generate_fresh_raises_on_zero_games(tmp_path, monkeypatch):
    """A crashed self-play proc (empty dir) fails loud + located, not a
    cryptic downstream 'no games in window'."""
    import catan_az.daily as daily
    from catan_az.config import AzConfig
    import pytest

    def empty_launch(cfg, out_dir, checkpoint, n_games, n_procs, champion, rules_id):
        d = out_dir / "crashed_self_play_async"
        d.mkdir(parents=True)   # empty: proc died before writing games
        return [d]

    monkeypatch.setattr(daily, "_launch_selfplay_procs", empty_launch)
    cfg = AzConfig(window_games=4, fresh_ratio=0.5)
    with pytest.raises(RuntimeError, match="0 games"):
        daily.generate_fresh(cfg, iter_dir=tmp_path / "iter_1",
                             champion="c", champion_ckpt=tmp_path / "c.pt",
                             capped_procs=1, prior_dirs=[])


def test_resumes_incomplete_iteration(tmp_path):
    """An iteration with a dir but no PUBLISH.done must be RESUMED (its
    self-play + train are salvaged via done-markers), not skipped to the next
    number (2026-06-14: iter-3 training crash wasted 6h when resume jumped to
    iter-4)."""
    from catan_az.daily import _next_iter_number
    # iter_1 fully done (PUBLISH.done), iter_3 started but incomplete.
    (tmp_path / "iter_1").mkdir()
    (tmp_path / "iter_1" / "PUBLISH.done").write_text("{}")
    (tmp_path / "iter_3").mkdir()
    (tmp_path / "iter_3" / "SELFPLAY.done").write_text("{}")  # no PUBLISH.done
    assert _next_iter_number(tmp_path) == 3   # resume iter_3, not 4


def test_next_iter_after_all_complete(tmp_path):
    from catan_az.daily import _next_iter_number
    (tmp_path / "iter_1").mkdir()
    (tmp_path / "iter_1" / "PUBLISH.done").write_text("{}")
    (tmp_path / "iter_2").mkdir()
    (tmp_path / "iter_2" / "PUBLISH.done").write_text("{}")
    assert _next_iter_number(tmp_path) == 3   # all done -> next is 3


def test_next_iter_empty(tmp_path):
    from catan_az.daily import _next_iter_number
    assert _next_iter_number(tmp_path) == 1
