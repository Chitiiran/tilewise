"""Analytics layer for the AZ dashboard — derived metrics + failure-mode
detectors computed from the loop's real files. Pure functions, unit-tested
without a server."""
from __future__ import annotations

import csv
import json


def _write_journal(p, rows):
    cols = ["iter", "champion", "selfplay_dirs", "window_dirs",
            "arena_wins_cand", "arena_wins_champ", "arena_draws",
            "arena_timeouts", "arena_winrate", "verdict", "champion_elo_after"]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def test_iteration_metrics_trends(tmp_path):
    from catan_az.analytics import iteration_metrics
    _write_journal(tmp_path / "journal.csv", [
        {"iter": 1, "arena_wins_cand": 78, "arena_wins_champ": 42,
         "arena_draws": 0, "arena_timeouts": 0, "arena_winrate": 0.65,
         "verdict": "promote", "champion_elo_after": 1003.6},
        {"iter": 2, "arena_wins_cand": 23, "arena_wins_champ": 39,
         "arena_draws": 58, "arena_timeouts": 120, "arena_winrate": 0.37,
         "verdict": "invalid", "champion_elo_after": 1003.6},
    ])
    m = iteration_metrics(tmp_path)
    assert len(m) == 2
    assert m[0]["winrate"] == 0.65 and m[0]["draw_rate"] == 0.0
    assert m[1]["draw_rate"] == 58 / 120         # 0.483
    assert m[1]["timeout_rate"] == 120 / 120     # all timed out


def test_holds_since_promotion_metric(tmp_path):
    from catan_az.analytics import holds_since_promotion
    _write_journal(tmp_path / "journal.csv", [
        {"iter": 1, "verdict": "promote"},
        {"iter": 2, "verdict": "hold"},
        {"iter": 3, "verdict": "invalid"},
    ])
    assert holds_since_promotion(tmp_path) == 2


def test_seat_bias_from_arena_results(tmp_path):
    """Per-seat cand winrate — detects board-luck vs skill in the gate."""
    from catan_az.analytics import seat_bias
    d = tmp_path / "iter_3" / "arena"
    d.mkdir(parents=True)
    rows = [
        {"seed": 1, "rot": 0, "winner_seat": 0, "winner_role": "cand", "timed_out": False},
        {"seed": 2, "rot": 0, "winner_seat": 1, "winner_role": "champ", "timed_out": False},
        {"seed": 3, "rot": 1, "winner_seat": 0, "winner_role": "champ", "timed_out": False},
    ]
    (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    sb = seat_bias(tmp_path, iter_n=3)
    assert sb["by_seat"][0]["cand_wins"] == 1   # seat 0 cand won once
    assert sb["total_games"] == 3


def test_detect_failure_modes(tmp_path):
    """The headline: surface failure modes analytics can prevent."""
    from catan_az.analytics import detect_failure_modes
    _write_journal(tmp_path / "journal.csv", [
        {"iter": 1, "verdict": "promote", "arena_draws": 0, "arena_timeouts": 0,
         "arena_wins_cand": 78, "arena_wins_champ": 42},
        {"iter": 2, "verdict": "invalid", "arena_draws": 58, "arena_timeouts": 120,
         "arena_wins_cand": 23, "arena_wins_champ": 39},
        {"iter": 3, "verdict": "hold", "arena_draws": 33, "arena_timeouts": 120,
         "arena_wins_cand": 38, "arena_wins_champ": 49},
    ])
    flags = detect_failure_modes(tmp_path, stagnation_threshold=2)
    names = {f["id"] for f in flags}
    assert "stagnation" in names           # 2 holds/invalids since promote
    assert "high_timeout_rate" in names    # iter-2/3 100% timeouts
    assert all("severity" in f and "message" in f for f in flags)


def test_no_failure_modes_when_healthy(tmp_path):
    from catan_az.analytics import detect_failure_modes
    _write_journal(tmp_path / "journal.csv", [
        {"iter": 1, "verdict": "promote", "arena_draws": 5, "arena_timeouts": 2,
         "arena_wins_cand": 70, "arena_wins_champ": 45},
    ])
    flags = detect_failure_modes(tmp_path, stagnation_threshold=5)
    assert flags == []   # healthy -> no flags
