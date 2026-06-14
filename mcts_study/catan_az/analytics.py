"""Analytics layer for the AZ dashboard: derived metrics + failure-mode
detectors computed from the loop's real files (journal.csv, ladder.json,
per-iteration arena results.jsonl). Pure functions — the server is a thin
shell over these. The point is to turn raw run data into signals that catch
failure modes EARLY (stale data, stagnation, censored verdicts, seat bias)
instead of after a multi-day run wastes compute."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def _read_journal(loop_root) -> list[dict]:
    p = Path(loop_root) / "journal.csv"
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _f(row, key, default=0.0):
    v = row.get(key, "")
    try:
        return float(v) if v != "" else default
    except (TypeError, ValueError):
        return default


def _i(row, key, default=0):
    return int(_f(row, key, default))


def iteration_metrics(loop_root) -> list[dict]:
    """Per-iteration derived metrics: winrate, draw_rate, timeout_rate, decisive
    count, Elo. One dict per journal row, in order."""
    out = []
    for r in _read_journal(loop_root):
        c, ch, d = (_i(r, "arena_wins_cand"), _i(r, "arena_wins_champ"),
                    _i(r, "arena_draws"))
        to = _i(r, "arena_timeouts")
        g = c + ch + d
        out.append({
            "iter": _i(r, "iter"),
            "verdict": r.get("verdict", "?"),
            "winrate": _f(r, "arena_winrate"),
            "draw_rate": (d / g) if g else 0.0,
            "timeout_rate": (to / g) if g else 0.0,
            "decisive": c + ch,
            "games": g,
            "elo": _f(r, "champion_elo_after"),
            "champion": r.get("champion", ""),
        })
    return out


def holds_since_promotion(loop_root) -> int:
    """Iterations since the last promotion — the stagnation counter."""
    rows = _read_journal(loop_root)
    n = 0
    for r in reversed(rows):
        if r.get("verdict") == "promote":
            break
        n += 1
    return n


def seat_bias(loop_root, *, iter_n: int) -> dict:
    """Per-seat candidate winrate from an iteration's arena results — detects
    whether the gate is measuring skill or board-luck (a candidate that only
    wins from certain seats is a seating artifact, not an improvement)."""
    p = Path(loop_root) / f"iter_{iter_n}" / "arena" / "results.jsonl"
    by_seat = {s: {"cand_wins": 0, "champ_wins": 0, "games": 0} for s in range(4)}
    total = timeouts = 0
    if p.exists():
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            total += 1
            timeouts += 1 if r.get("timed_out") else 0
            seat = r.get("winner_seat", -1)
            if seat in by_seat:
                if r.get("winner_role") == "cand":
                    by_seat[seat]["cand_wins"] += 1
                elif r.get("winner_role") == "champ":
                    by_seat[seat]["champ_wins"] += 1
    return {"by_seat": by_seat, "total_games": total, "timeouts": timeouts}


def detect_failure_modes(loop_root, *, stagnation_threshold: int = 5) -> list[dict]:
    """The headline: failure modes analytics can catch EARLY. Returns a list of
    {id, severity, message} flags — empty when healthy. Each guards a real way
    a multi-day AZ run silently wastes compute."""
    flags = []
    metrics = iteration_metrics(loop_root)
    if not metrics:
        return flags

    # 1. Stagnation — N iterations with no promotion.
    holds = holds_since_promotion(loop_root)
    if holds >= stagnation_threshold:
        flags.append({"id": "stagnation", "severity": "warn",
                      "message": f"{holds} iterations since last promotion "
                                 f"(>= {stagnation_threshold})"})

    last = metrics[-1]

    # 2. High timeout rate — verdict is wall-clock-censored, winrate unreliable.
    if last["timeout_rate"] >= 0.50:
        flags.append({"id": "high_timeout_rate", "severity": "warn",
                      "message": f"iter {last['iter']}: "
                                 f"{last['timeout_rate']:.0%} of games timed out "
                                 f"(VP-tiebreak decided) — verdict less reliable"})

    # 3. High draw rate — nets converging / stalling; the gate may invalidate.
    if last["draw_rate"] >= 0.40:
        flags.append({"id": "high_draw_rate", "severity": "warn",
                      "message": f"iter {last['iter']}: {last['draw_rate']:.0%} "
                                 f"draws — candidate ≈ champion (low signal)"})

    # 4. Invalid verdict — the last gate produced no trustworthy result.
    if last["verdict"] == "invalid":
        flags.append({"id": "invalid_verdict", "severity": "error",
                      "message": f"iter {last['iter']} verdict INVALID — "
                                 f"too many draws / too few decisive games"})

    return flags
