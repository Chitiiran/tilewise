"""run_cycle redesign + max-iters terminal (spec §4 steps 3-7)."""
from __future__ import annotations

import csv


def test_run_cycle_uses_generator_and_gen_window(tmp_path, monkeypatch):
    import catan_az.daily as daily
    from catan_az.config import AzConfig
    from catan_az.ladder import Ladder
    Ladder(tmp_path, champion_checkpoint=str(tmp_path / "c.pt"), champion_name="seed")
    (tmp_path / "c.pt").write_bytes(b"x")
    seen = {}

    monkeypatch.setattr(daily, "select_generator",
                        lambda root, iter_n, champion: ("seed", "/c.pt"))
    def fake_gen(cfg, **k):
        seen["gi"] = k["gen_iter"]
        return [tmp_path / "sp"]

    def fake_window(dirs, **k):
        seen["win_min"] = k["gen_iter_min"]
        return list(dirs)

    monkeypatch.setattr(daily, "generate_iter_games", fake_gen)
    monkeypatch.setattr(daily, "_all_selfplay_dirs", lambda r: [])
    monkeypatch.setattr(daily, "gen_window", fake_window)
    monkeypatch.setattr(daily, "run_iteration",
                        lambda cfg, root, n, *, existing_selfplay_dirs: "hold")
    monkeypatch.setattr(daily, "archive_out_of_window", lambda **k: 0)
    monkeypatch.setattr(daily, "_append_progress_row", lambda *a, **k: None)

    v = daily.run_cycle(AzConfig(), tmp_path, 3, capped_procs=5)
    assert v == "hold"
    assert seen["gi"] == 3
    assert seen["win_min"] == 0   # last_promotion_iter starts at 0


def test_run_cycle_sets_reset_boundary_on_promote(tmp_path, monkeypatch):
    import catan_az.daily as daily
    from catan_az.config import AzConfig
    from catan_az.ladder import Ladder
    Ladder(tmp_path, champion_checkpoint=str(tmp_path / "c.pt"), champion_name="seed")
    (tmp_path / "c.pt").write_bytes(b"x")

    monkeypatch.setattr(daily, "select_generator", lambda root, n, champion: ("seed", "/c.pt"))
    monkeypatch.setattr(daily, "generate_iter_games", lambda cfg, **k: [tmp_path / "sp"])
    monkeypatch.setattr(daily, "_all_selfplay_dirs", lambda r: [])
    monkeypatch.setattr(daily, "gen_window", lambda dirs, **k: list(dirs))
    monkeypatch.setattr(daily, "_append_progress_row", lambda *a, **k: None)
    monkeypatch.setattr(daily, "archive_out_of_window", lambda **k: 0)

    def fake_iter(cfg, root, n, *, existing_selfplay_dirs):
        # simulate run_iteration's PUBLISH promoting az_iter_4
        Ladder(root).register_candidate("az_iter_4", "/4.pt", created_iter=4)
        Ladder(root).promote("az_iter_4")
        return "promote"

    monkeypatch.setattr(daily, "run_iteration", fake_iter)
    daily.run_cycle(AzConfig(), tmp_path, 4, capped_procs=5)
    assert Ladder(tmp_path).last_promotion_iter() == 4   # boundary set


def test_run_day_stops_after_max_iters_without_promotion(tmp_path):
    from catan_az.daily import run_day
    from catan_az.config import AzConfig
    jp = tmp_path / "journal.csv"
    with open(jp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "verdict"]); w.writeheader()
        for i in range(1, 10):
            w.writerow({"iter": i, "verdict": "hold"})   # 9 holds pre-seeded
    calls = []

    def fake_cycle(cfg, root, n, procs):
        calls.append(n)
        with open(jp, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=["iter", "verdict"]).writerow(
                {"iter": n, "verdict": "hold"})
        return "hold"

    run_day(AzConfig(max_iters_per_model=10), loop_root=tmp_path, capped_procs=5,
            cycle_fn=fake_cycle, max_iters=50, next_iter=10)
    assert len(calls) == 1   # one cycle -> 10 holds since promotion -> STOP


def test_holds_since_promotion(tmp_path):
    from catan_az.daily import _holds_since_promotion
    jp = tmp_path / "journal.csv"
    with open(jp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "verdict"]); w.writeheader()
        for v in ["hold", "promote", "hold", "invalid", "hold"]:
            w.writerow({"iter": 0, "verdict": v})
    assert _holds_since_promotion(jp) == 3   # since the 'promote'
