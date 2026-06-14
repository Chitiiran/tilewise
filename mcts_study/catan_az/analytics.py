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
        it = _i(r, "iter")
        th = training_health(loop_root, iter_n=it)   # folded in (NEW-2: no N+1)
        out.append({
            "iter": it,
            "verdict": r.get("verdict", "?"),
            "winrate": _f(r, "arena_winrate"),
            "draw_rate": (d / g) if g else 0.0,
            "timeout_rate": (to / g) if g else 0.0,
            "decisive": c + ch,
            "games": g,
            "elo": _f(r, "champion_elo_after"),
            "champion": r.get("champion", ""),
            "epochs_trained": th.get("epochs_trained", 0),
            "val_top1": th.get("final_val_top1"),
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


# Stage-keyed staleness budgets (BUG-2 fix): the heartbeat is only rewritten at
# stage transitions, so a healthy multi-hour self-play stage legitimately has an
# old ts. A flat 30-min threshold false-positives it. self-play/arena get hours;
# train is short.
_STAGE_STALE_SECONDS = {
    "selfplay": 8 * 3600,    # 1000 games can take ~7-8h
    "iterate": 8 * 3600,     # iterate covers train+arena which is long
    "arena": 8 * 3600,
    "train": 1800,
    "done": 1800,
}
_DEFAULT_STALE_SECONDS = 1800


def liveness(loop_root, *, now=None, stale_after_seconds=None) -> dict:
    """LIVE run health from status.json's heartbeat + daily_state.json progress.
    `alive` = heartbeat age < a STAGE-KEYED budget (BUG-2: self-play runs for
    hours, so a flat 30-min threshold false-positives a healthy run). Pass
    stale_after_seconds to override."""
    now = now if now is not None else time.time()
    status = _read_json_safe(Path(loop_root) / "status.json", {})
    ds_path = Path(loop_root) / "daily_state.json"
    ds = _read_json_safe(ds_path, {})
    ts = status.get("ts")
    stage = status.get("stage")
    iter_n = status.get("iter")
    # status.json is only written at stage TRANSITIONS; run_cycle rewrites
    # daily_state.json (atomically) at every stage, so its mtime is the live
    # heartbeat during the hours-long self-play stage. Fall back to it when
    # status.json has no usable ts (pilot 2026-06-14: dashboard read "NOT
    # RUNNING" for the whole self-play stage because only daily_state existed).
    if not isinstance(ts, (int, float)) and ds_path.exists():
        ts = ds_path.stat().st_mtime
        stage = ds.get("stage", stage)
        iter_n = ds.get("iter", iter_n)
    age = (now - ts) if isinstance(ts, (int, float)) else None
    budget = (stale_after_seconds if stale_after_seconds is not None
              else _STAGE_STALE_SECONDS.get(stage, _DEFAULT_STALE_SECONDS))
    return {
        "alive": (age is not None and age < budget),
        "stage": stage,
        "iter": iter_n,
        "age_seconds": age,
        "stale_budget_seconds": budget,
        "progress": {"fresh_done": ds.get("fresh_done"),
                     "fresh_target": ds.get("fresh_target")},
    }


def selfplay_health(loop_root, *, iter_n: int) -> dict:
    """Self-play degeneracy signal from games.parquet: mean game length + the
    fraction of games hitting the step cap (200k). Collapsing/exploding game
    length is the earliest sign self-play data is degenerate — and it poisons
    everything downstream. (critic iter-3 #3)"""
    import glob
    import pandas as pd
    dirs = glob.glob(str(Path(loop_root) / f"iter_{iter_n}" / "selfplay" /
                         "*" / "games*.parquet"))
    if not dirs:
        return {"available": False, "n_games": 0}
    lengths, timed = [], 0
    n = 0
    for f in dirs:
        try:
            df = pd.read_parquet(f, columns=["length_in_moves", "timed_out"])
        except Exception:
            continue
        n += len(df)
        lengths.extend(df["length_in_moves"].tolist())
        timed += int(df["timed_out"].sum())
    if n == 0:
        return {"available": False, "n_games": 0}
    cap_hits = sum(1 for x in lengths if x >= 200_000)
    return {
        "available": True,
        "n_games": n,
        "mean_length": sum(lengths) / len(lengths),
        "step_cap_fraction": cap_hits / n,
        "timed_out_fraction": timed / n,
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
    """Parse PROGRESS.md table rows by HEADER NAME (BUG-1 fix: the real on-disk
    file is the legacy 8-col schema where new_games sits at a different index
    than the new 9-col one; positional parsing read window_dirs as new_games and
    the stale_data detector never fired). Header-name parsing survives both
    schemas + any future reorder."""
    p = Path(loop_root) / "PROGRESS.md"
    if not p.exists():
        return []
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines()
             if ln.lstrip().startswith("|")]
    header = None
    out = []
    for ln in lines:
        cells = [c.strip() for c in ln.split("|")[1:-1]]
        if not cells:
            continue
        if header is None and "iter" in cells:
            header = {name: i for i, name in enumerate(cells)}
            continue
        if header is None or not cells[header.get("iter", 0)].isdigit():
            continue   # separator row or pre-header noise
        get = lambda key, d="": (cells[header[key]] if key in header
                                 and header[key] < len(cells) else d)
        out.append({"iter": int(get("iter")), "champion": get("champion"),
                    "generator": get("generator", get("champion")),
                    "new_games": _safe_int(get("new_games"))})
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

    last = metrics[-1]

    # 4. High draw rate on the LATEST iter — nets converging (low signal).
    if last["draw_rate"] >= draw_cap:
        flags.append({"id": "high_draw_rate", "severity": "warn",
                      "message": f"iter {last['iter']}: {last['draw_rate']:.0%} "
                                 f"draws (>= {draw_cap:.0%}) — candidate ≈ champion"})

    # 5. Training health of the LATEST iter — catch a broken candidate BEFORE
    #    its ~14h arena (critic: the only detector that saves a run pre-arena).
    th = training_health(loop_root, iter_n=last["iter"])
    if th.get("nan_loss"):
        flags.append({"id": "training_collapse", "severity": "error",
                      "message": f"iter {last['iter']}: NaN training loss — "
                                 f"candidate is broken"})
    elif th.get("available") and th.get("epochs_trained", 0) <= 1:
        flags.append({"id": "undertrained", "severity": "warn",
                      "message": f"iter {last['iter']}: trained only "
                                 f"{th['epochs_trained']} epoch(s) — early-stop "
                                 f"fired immediately"})

    # 5b. Elo PLATEAU — fires EARLIER than holds_since_promotion: a flattening
    #     Elo slope over recent iters is the slow-failure signal (the expensive
    #     one on a 6-day run). |Δelo| over last 3 iters < epsilon -> warn.
    if len(metrics) >= 3:
        recent = [m["elo"] for m in metrics[-3:]]
        if max(recent) - min(recent) < 0.5:   # essentially no Elo movement
            flags.append({"id": "plateau", "severity": "warn",
                          "message": f"Elo flat (~{recent[-1]:.0f}) over last 3 "
                                     f"iters — likely plateaued, consider stopping"})

    # 6. Timeout change-point — a SHARP jump in timeout rate (steady 100% is the
    #    expected VP-tiebreak steady state; a 40%->100% swing means something
    #    changed). Replaces the deleted static timeout flag (critic §4).
    if len(metrics) >= 2:
        prev, cur = metrics[-2]["timeout_rate"], metrics[-1]["timeout_rate"]
        if cur - prev >= 0.30:
            flags.append({"id": "timeout_jump", "severity": "warn",
                          "message": f"timeout rate jumped {prev:.0%}->{cur:.0%} "
                                     f"(iter {metrics[-2]['iter']}->{last['iter']}) "
                                     f"— behavior changed, inspect"})

    return flags
