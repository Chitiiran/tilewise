"""Iteration 2: liveness, training-health, stale-data, and the cross-cutting
correctness fixes the Opus critic flagged (config-driven thresholds, history
scan, utf-8, crash-safe parse)."""
from __future__ import annotations

import csv
import json
import time


def _journal(p, rows):
    cols = ["iter", "champion", "selfplay_dirs", "window_dirs",
            "arena_wins_cand", "arena_wins_champ", "arena_draws",
            "arena_timeouts", "arena_winrate", "verdict", "champion_elo_after"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


# ---- liveness ----

def test_liveness_fresh_heartbeat(tmp_path):
    from catan_az.analytics import liveness
    (tmp_path / "status.json").write_text(json.dumps(
        {"iter": 5, "stage": "arena", "ts": time.time()}))
    (tmp_path / "daily_state.json").write_text(json.dumps(
        {"iter": 5, "stage": "iterate", "fresh_target": 1000, "fresh_done": 430}))
    L = liveness(tmp_path, now=time.time())
    assert L["alive"] is True
    assert L["stage"] == "arena"
    assert L["age_seconds"] < 5
    assert L["progress"]["fresh_done"] == 430
    assert L["progress"]["fresh_target"] == 1000


def test_liveness_stale_heartbeat_flags_dead(tmp_path):
    from catan_az.analytics import liveness
    old = time.time() - 4 * 3600   # 4h ago
    (tmp_path / "status.json").write_text(json.dumps(
        {"iter": 5, "stage": "arena", "ts": old}))
    L = liveness(tmp_path, now=time.time(), stale_after_seconds=1800)
    assert L["alive"] is False
    assert L["age_seconds"] > 1800


# ---- training health ----

def test_training_health_reads_log(tmp_path):
    from catan_az.analytics import training_health
    d = tmp_path / "iter_3" / "training"
    d.mkdir(parents=True)
    (d / "training_log.json").write_text(json.dumps({"epochs": [
        {"epoch": 1, "train_loss_total": 2.3, "val_loss_total": 2.4, "val_policy_top1_acc": 0.46},
        {"epoch": 2, "train_loss_total": 2.0, "val_loss_total": 2.5, "val_policy_top1_acc": 0.45},
    ]}))
    h = training_health(tmp_path, iter_n=3)
    assert h["epochs_trained"] == 2
    assert h["final_val_top1"] == 0.45
    assert h["best_val_top1"] == 0.46
    assert h["nan_loss"] is False


def test_training_health_detects_nan(tmp_path):
    from catan_az.analytics import training_health
    d = tmp_path / "iter_3" / "training"
    d.mkdir(parents=True)
    (d / "training_log.json").write_text(json.dumps({"epochs": [
        {"epoch": 1, "train_loss_total": float("nan"), "val_loss_total": 2.4, "val_policy_top1_acc": 0.1},
    ]}))
    h = training_health(tmp_path, iter_n=3)
    assert h["nan_loss"] is True


# ---- stale-data detector (THE project's worst bug, no alarm until now) ----

def test_detect_stale_data_from_progress(tmp_path):
    from catan_az.analytics import detect_failure_modes
    _journal(tmp_path / "journal.csv", [
        {"iter": 4, "verdict": "hold", "arena_draws": 5, "arena_timeouts": 0,
         "arena_wins_cand": 50, "arena_wins_champ": 50},
    ])
    # PROGRESS.md row showing new_games=0 (the stale-data signature)
    (tmp_path / "PROGRESS.md").write_text(
        "| iter | champion | generator | new_games | window_dirs | window_iters | verdict | winrate | draws |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
        "| 4 | az_iter_1 | az_iter_1 | 0 | 21 | STALE 3 | hold | 50% | 5% |\n")
    flags = detect_failure_modes(tmp_path)
    assert any(f["id"] == "stale_data" for f in flags)


# ---- cross-cutting correctness ----

def test_detect_scans_history_not_just_last_row(tmp_path):
    """A 100% timeout in iter-2 must still surface even if iter-3 is cleaner —
    the critic's 'metrics[-1] under-reports systemic conditions'."""
    from catan_az.analytics import detect_failure_modes
    _journal(tmp_path / "journal.csv", [
        {"iter": 2, "verdict": "invalid", "arena_draws": 58, "arena_timeouts": 120,
         "arena_wins_cand": 23, "arena_wins_champ": 39},
        {"iter": 3, "verdict": "hold", "arena_draws": 5, "arena_timeouts": 0,
         "arena_wins_cand": 60, "arena_wins_champ": 55},
    ])
    flags = detect_failure_modes(tmp_path)
    # iter-2's invalid verdict is in history, not the last row
    assert any(f["id"] == "invalid_verdict" for f in flags)


def test_seat_bias_crash_safe_on_partial_line(tmp_path):
    from catan_az.analytics import seat_bias
    d = tmp_path / "iter_3" / "arena"
    d.mkdir(parents=True)
    (d / "results.jsonl").write_text(
        '{"seed": 1, "rot": 0, "winner_seat": 0, "winner_role": "cand", "timed_out": false}\n'
        '{"seed": 2, "rot": 0, "winner')   # partial final line (kill mid-write)
    sb = seat_bias(tmp_path, iter_n=3)
    assert sb["total_games"] == 1   # partial line skipped, didn't crash


# ---- iter-3: fixes for bugs the Opus critic caught ----

def test_progress_parse_by_header_legacy_8col(tmp_path):
    """BUG-1: the REAL on-disk PROGRESS.md is the OLD 8-col schema where
    new_games is at a DIFFERENT index. Parse by HEADER NAME, not position, so
    stale_data fires on real data (not the new-col fixture that masked it)."""
    from catan_az.analytics import _progress_rows
    # legacy 8-col: iter | champion | new_games | window_dirs | trained_on | verdict | winrate | draws
    (tmp_path / "PROGRESS.md").write_text(
        "| iter | champion | new_games | window_dirs | trained_on | verdict | winrate | draws |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 4 | az_iter_1 | 0 | 21 | STALE iter 3 | hold | 51% | 38% |\n")
    rows = _progress_rows(tmp_path)
    assert rows[0]["new_games"] == 0    # the 0, NOT window_dirs=21


def test_progress_parse_by_header_new_9col(tmp_path):
    from catan_az.analytics import _progress_rows
    (tmp_path / "PROGRESS.md").write_text(
        "| iter | champion | generator | new_games | window_dirs | window_iters | verdict | winrate | draws |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
        "| 5 | az_iter_1 | cand_iter_4 | 1000 | 25 | 3,4,5 | hold | 51% | 38% |\n")
    rows = _progress_rows(tmp_path)
    assert rows[0]["new_games"] == 1000
    assert rows[0]["generator"] == "cand_iter_4"


def test_stale_data_fires_on_legacy_schema(tmp_path):
    """End-to-end: the detector must flag stale_data on the REAL legacy file."""
    from catan_az.analytics import detect_failure_modes
    _journal(tmp_path / "journal.csv", [
        {"iter": 4, "verdict": "hold", "arena_draws": 5, "arena_timeouts": 0,
         "arena_wins_cand": 50, "arena_wins_champ": 50}])
    (tmp_path / "PROGRESS.md").write_text(
        "| iter | champion | new_games | window_dirs | trained_on | verdict | winrate | draws |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 4 | az_iter_1 | 0 | 21 | STALE iter 3 | hold | 51% | 38% |\n")
    flags = detect_failure_modes(tmp_path)
    assert any(f["id"] == "stale_data" for f in flags)


def test_liveness_stage_keyed_threshold(tmp_path):
    """BUG-2: self-play stages run for HOURS legitimately; a 30-min stale
    threshold false-positives them. The threshold must be stage-aware."""
    import json, time
    from catan_az.analytics import liveness
    sp_age = time.time() - 2 * 3600   # 2h into self-play (healthy)
    (tmp_path / "status.json").write_text(json.dumps(
        {"iter": 5, "stage": "selfplay", "ts": sp_age}))
    L = liveness(tmp_path, now=time.time())
    assert L["alive"] is True   # 2h self-play is NOT dead
    # but 2h in the arena/train... still ok (arena also long); test a clearly-dead case
    dead_age = time.time() - 12 * 3600
    (tmp_path / "status.json").write_text(json.dumps(
        {"iter": 5, "stage": "selfplay", "ts": dead_age}))
    assert liveness(tmp_path, now=time.time())["alive"] is False   # 12h = dead even for selfplay


def test_training_collapse_flag(tmp_path):
    """training_health folded into detect_failure_modes as a real FLAG."""
    from catan_az.analytics import detect_failure_modes
    _journal(tmp_path / "journal.csv", [
        {"iter": 3, "verdict": "hold", "arena_draws": 5, "arena_timeouts": 0,
         "arena_wins_cand": 50, "arena_wins_champ": 50}])
    d = tmp_path / "iter_3" / "training"
    d.mkdir(parents=True)
    (d / "training_log.json").write_text(json.dumps({"epochs": [
        {"epoch": 1, "train_loss_total": float("nan"), "val_policy_top1_acc": 0.1}]}))
    flags = detect_failure_modes(tmp_path)
    assert any(f["id"] == "training_collapse" for f in flags)


def test_timeout_change_point_flag(tmp_path):
    """Replaces the deleted static timeout flag: flag a SHARP jump in timeout
    rate (steady 100% is fine; 40%->100% means something changed)."""
    from catan_az.analytics import detect_failure_modes
    _journal(tmp_path / "journal.csv", [
        {"iter": 2, "verdict": "hold", "arena_draws": 5, "arena_timeouts": 48,
         "arena_wins_cand": 50, "arena_wins_champ": 50},   # ~44% timeout
        {"iter": 3, "verdict": "hold", "arena_draws": 5, "arena_timeouts": 120,
         "arena_wins_cand": 50, "arena_wins_champ": 50}])   # ~100% timeout -> jump
    flags = detect_failure_modes(tmp_path)
    assert any(f["id"] == "timeout_jump" for f in flags)


def test_selfplay_health_from_parquet(tmp_path):
    """One self-play degeneracy metric: mean game length + step-cap fraction."""
    import pandas as pd
    from catan_az.analytics import selfplay_health
    d = tmp_path / "iter_5" / "selfplay" / "run1"
    d.mkdir(parents=True)
    pd.DataFrame({"seed": [1, 2, 3], "winner": [0, 1, -1],
                  "length_in_moves": [600, 800, 200000],   # one hit the cap
                  "timed_out": [False, False, True]}).to_parquet(d / "games.x.parquet")
    h = selfplay_health(tmp_path, iter_n=5)
    assert h["n_games"] == 3
    assert h["mean_length"] == (600 + 800 + 200000) / 3
    assert h["step_cap_fraction"] > 0   # one game hit the cap
