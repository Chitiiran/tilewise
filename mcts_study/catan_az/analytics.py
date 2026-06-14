"""Analytics layer for the AZ dashboard: derived metrics + failure-mode
detectors computed from the loop's real files (journal.csv, ladder.json,
per-iteration arena results.jsonl). Pure functions — the server is a thin
shell over these. The point is to turn raw run data into signals that catch
failure modes EARLY (stale data, stagnation, censored verdicts, seat bias)
instead of after a multi-day run wastes compute."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path


def _read_journal(loop_root) -> list[dict]:
    p = Path(loop_root) / "journal.csv"
    if not p.exists():
        return []
    # utf-8: the loop writes from WSL (utf-8); the dashboard may read on a
    # Windows host (cp1252 default would mis-decode). Sort by iter so
    # "last row" == latest iteration even after a resume/re-publish.
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: _i(r, "iter"))
    return rows


def _read_json_safe(p, default):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


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
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)   # crash-safe: skip a partial final line
            except json.JSONDecodeError:
                continue
            total += 1
            timeouts += 1 if r.get("timed_out") else 0
            seat = r.get("winner_seat", -1)
            if seat in by_seat:
                if r.get("winner_role") == "cand":
                    by_seat[seat]["cand_wins"] += 1
                elif r.get("winner_role") == "champ":
                    by_seat[seat]["champ_wins"] += 1
    return {"by_seat": by_seat, "total_games": total, "timeouts": timeouts}


def liveness(loop_root, *, now=None, stale_after_seconds: int = 1800) -> dict:
    """LIVE run health from status.json's heartbeat + daily_state.json progress.
    The critic's #1 gap: a run that died 6h ago looked identical to a healthy
    one. `alive` = heartbeat age < stale_after_seconds. Includes in-stage
    progress (fresh_done/fresh_target) for an ETA."""
    now = now if now is not None else time.time()
    status = _read_json_safe(Path(loop_root) / "status.json", {})
    ds = _read_json_safe(Path(loop_root) / "daily_state.json", {})
    ts = status.get("ts")
    age = (now - ts) if isinstance(ts, (int, float)) else None
    return {
        "alive": (age is not None and age < stale_after_seconds),
        "stage": status.get("stage"),
        "iter": status.get("iter"),
        "age_seconds": age,
        "progress": {"fresh_done": ds.get("fresh_done"),
                     "fresh_target": ds.get("fresh_target")},
    }


def training_health(loop_root, *, iter_n: int) -> dict:
    """Per-iteration training health from training_log.json — catches a broken
    candidate BEFORE its ~14h arena runs (critic's highest-value miss).
    Flags NaN loss + reports val_top1 trend + epochs trained (early-stop-at-1
    = barely trained)."""
    p = Path(loop_root) / f"iter_{iter_n}" / "training" / "training_log.json"
    log = _read_json_safe(p, {})
    epochs = log.get("epochs", []) if isinstance(log, dict) else []
    if not epochs:
        return {"available": False, "epochs_trained": 0}
    val = [e.get("val_policy_top1_acc") for e in epochs
           if e.get("val_policy_top1_acc") is not None]
    losses = [e.get("train_loss_total", 0.0) for e in epochs]
    nan = any(isinstance(x, float) and math.isnan(x) for x in losses)
    return {
        "available": True,
        "epochs_trained": len(epochs),
        "final_val_top1": val[-1] if val else None,
        "best_val_top1": max(val) if val else None,
        "nan_loss": nan,
    }


def _progress_rows(loop_root) -> list[dict]:
    """Parse PROGRESS.md table rows -> {iter, generator, new_games, ...}."""
    p = Path(loop_root) / "PROGRESS.md"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) >= 4 and cells[0].isdigit():   # data row
            out.append({"iter": int(cells[0]), "champion": cells[1],
                        "generator": cells[2], "new_games": _safe_int(cells[3])})
    return out


def _safe_int(s, default=-1):
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def detect_failure_modes(loop_root, *, cfg=None) -> list[dict]:
    """Failure modes analytics catches EARLY. Returns {id, severity, message}
    flags. Thresholds come from AzConfig when given (critic: stop hard-coding).
    Scans HISTORY, not just the last row, so systemic conditions (e.g. every
    iter timing out) surface as one flag instead of vanishing on a clean row."""
    flags = []
    metrics = iteration_metrics(loop_root)
    if not metrics:
        return flags

    # thresholds: prefer config, fall back to sensible defaults.
    stag = getattr(cfg, "max_iters_per_model", 5) if cfg else 5
    draw_cap = getattr(cfg, "arena_max_draw_rate", 0.40) if cfg else 0.40

    # 1. Stagnation — N iterations with no promotion (the slow silent death).
    holds = holds_since_promotion(loop_root)
    if holds >= stag:
        flags.append({"id": "stagnation", "severity": "warn",
                      "message": f"{holds} iterations since last promotion "
                                 f"(>= {stag})"})

    # 2. Stale data — new_games==0 (the project's most expensive bug; had NO
    #    alarm before). Read from PROGRESS.md.
    for pr in _progress_rows(loop_root):
        if pr["new_games"] == 0:
            flags.append({"id": "stale_data", "severity": "error",
                          "message": f"iter {pr['iter']}: 0 new games generated "
                                     f"— retrained on existing data (stale)"})
            break   # one flag is enough

    # 3. Invalid verdicts ANYWHERE in history (not just last row).
    invalid_iters = [m["iter"] for m in metrics if m["verdict"] == "invalid"]
    if invalid_iters:
        flags.append({"id": "invalid_verdict", "severity": "error",
                      "message": f"invalid (untrustworthy) verdict at iter(s) "
                                 f"{invalid_iters} — too many draws / too few decisive"})

    # 4. High draw rate on the LATEST iter — nets converging (low signal).
    #    (NOTE: timeout_rate is intentionally NOT flagged — under the VP-tiebreak
    #    redesign a 100% timeout that's still DECISIVE is the expected steady
    #    state, not a fault. Flagging it caused alarm fatigue. — critic fix.)
    last = metrics[-1]
    if last["draw_rate"] >= draw_cap:
        flags.append({"id": "high_draw_rate", "severity": "warn",
                      "message": f"iter {last['iter']}: {last['draw_rate']:.0%} "
                                 f"draws (>= {draw_cap:.0%}) — candidate ≈ champion"})

    return flags
