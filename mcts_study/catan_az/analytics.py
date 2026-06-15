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


def arena_live(loop_root, *, iter_n: int, cfg) -> dict:
    """LIVE arena view for the running gate — the most suspenseful stage (a
    multi-hour 300-game series that decides promote/hold/invalid). Reads
    results.jsonl as it grows and reports the running winrate over DECISIVE
    games, progress toward the target, the PROJECTED verdict against the bar,
    draw/timeout rates, and a recent-games stream for the live feed."""
    p = Path(loop_root) / f"iter_{iter_n}" / "arena" / "results.jsonl"
    if not p.exists():
        return {"available": False, "games_played": 0,
                "target": getattr(cfg, "arena_games", 0)}
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue                       # crash-safe: skip torn final line
    if not rows:
        return {"available": False, "games_played": 0,
                "target": getattr(cfg, "arena_games", 0)}
    cand = sum(1 for r in rows if r.get("winner_role") == "cand")
    champ = sum(1 for r in rows if r.get("winner_role") == "champ")
    draws = sum(1 for r in rows if r.get("winner_role") not in ("cand", "champ"))
    timeouts = sum(1 for r in rows if r.get("timed_out"))
    n = len(rows)
    decisive = cand + champ
    winrate = (cand / decisive) if decisive else 0.0
    draw_rate = draws / n
    target = int(getattr(cfg, "arena_games", 0))
    bar = float(getattr(cfg, "promote_threshold", 0.65))
    max_draw = float(getattr(cfg, "arena_max_draw_rate", 0.40))
    min_dec = int(getattr(cfg, "arena_min_decisive", 40))

    # Wilson 95% CI on the decisive winrate (Opus #2: an honest band, not a
    # point estimate that flips). z=1.96.
    ci = _wilson_ci(cand, decisive, z=1.96)
    too_close = ci["lo"] <= bar <= ci["hi"]   # CI straddles the promote bar

    # Clinch/eliminate (Opus #2/#4): of the decisive games left, how many must
    # cand win to reach the bar? Remaining decisive estimated via the live draw
    # rate (at 40% draws only ~60% of remaining games even count).
    remaining = max(0, target - n)
    rem_dec = int(round(remaining * (1.0 - draw_rate)))
    final_dec = decisive + rem_dec
    need = (int(math.ceil(bar * final_dec)) - cand) if final_dec else 0
    clinch = max(0, need)
    clinch_possible = need <= rem_dec
    # Already clinched: cand reaches the bar even if it loses every remaining
    # decisive game (need <= 0). The most exciting state the board can show.
    clinched = (need <= 0 and decisive >= min_dec)
    # Required winrate from here vs actual (Opus #4: the suspense is the pace gap)
    required_from_here = (need / rem_dec) if rem_dec > 0 else (0.0 if clinched else 1.0)
    actual_winrate = winrate

    # VP-margin aggregates (Opus #1: skill vs luck in a 100%-timeout arena).
    cand_m = [r["vp_margin"] for r in rows
              if r.get("winner_role") == "cand" and r.get("vp_margin") is not None]
    champ_m = [r["vp_margin"] for r in rows
               if r.get("winner_role") == "champ" and r.get("vp_margin") is not None]
    mean_cand_margin = (sum(cand_m) / len(cand_m)) if cand_m else None
    mean_champ_margin = (sum(champ_m) / len(champ_m)) if champ_m else None
    # margin delta (Opus #5: one signed number — cand wins bigger than it loses?)
    margin_delta = ((mean_cand_margin - mean_champ_margin)
                    if (mean_cand_margin is not None
                        and mean_champ_margin is not None) else None)

    def _hist(ms):
        h = {"1": 0, "2": 0, "3": 0, "4+": 0}
        for m in ms:
            h["4+" if m >= 4 else str(m)] = h.get("4+" if m >= 4 else str(m), 0) + 1
        return h
    margin_hist = {"cand": _hist([m for m in cand_m if m >= 1]),
                   "champ": _hist([m for m in champ_m if m >= 1])}
    margins_available = bool(cand_m or champ_m)

    # Live rate / ETA from per-game timestamps (Opus #3: the arena lacked any).
    ts = [r["ts"] for r in rows if isinstance(r.get("ts"), (int, float))]
    games_per_hour, eta_hours, last_game_ts = 0.0, None, None
    if len(ts) >= 2:
        span_h = max((max(ts) - min(ts)) / 3600.0, 1e-9)
        games_per_hour = round(len(ts) / span_h, 1)
        last_game_ts = max(ts)
        if games_per_hour > 0:
            eta_hours = round(max(0, target - n) / games_per_hour, 2)
    else:
        # Opus #3: no per-game ts (iter_6 predates the producer fix). Fall back
        # to the file mtime — a valid "last game written" stamp for ANY iter, so
        # the dashboard shows recency instead of looking frozen.
        try:
            last_game_ts = p.stat().st_mtime
        except OSError:
            last_game_ts = None

    # Momentum (Opus #3): rolling winrate over the last K decisive games.
    K = 10
    dec_recent = [r for r in rows if r.get("winner_role") in ("cand", "champ")][-K:]
    m_cand = sum(1 for r in dec_recent if r.get("winner_role") == "cand")
    momentum = {"window": len(dec_recent), "cand": m_cand,
                "winrate": (m_cand / len(dec_recent)) if dec_recent else 0.0}

    # Draw-rate cliff (Opus #4): how close to the invalid threshold.
    cliff_ratio = (draw_rate / max_draw) if max_draw else 0.0

    if draw_rate > max_draw and n >= min_dec:
        projected = "invalid"
    elif clinched:
        projected = "promote"          # mathematically locked, even if CI is wide
    elif decisive < min_dec or too_close:
        projected = "pending"          # honest: don't call it while the CI straddles
    elif winrate > bar:
        projected = "promote"
    else:
        projected = "hold"

    recent = [{"seed": r.get("seed"), "winner_role": r.get("winner_role"),
               "winner_seat": r.get("winner_seat"),
               "vp_margin": r.get("vp_margin"),
               "timed_out": bool(r.get("timed_out"))}
              for r in rows[-16:][::-1]]
    return {
        "available": True,
        "games_played": n,
        "target": target,
        "cand_wins": cand, "champ_wins": champ, "draws": draws,
        "decisive": decisive,
        "winrate_decisive": winrate,
        "winrate_ci": ci,
        "too_close": too_close,
        "draw_rate": draw_rate,
        "draw_rate_cliff_ratio": cliff_ratio,
        "timeout_rate": timeouts / n,
        "promote_bar": bar,
        "min_decisive": min_dec,
        "remaining_decisive_est": rem_dec,
        "clinch_wins_needed": clinch,
        "clinch_possible": clinch_possible,
        "clinched": clinched,
        "required_winrate_from_here": required_from_here,
        "actual_winrate": actual_winrate,
        "mean_cand_margin": mean_cand_margin,
        "mean_champ_margin": mean_champ_margin,
        "margin_delta": margin_delta,
        "margin_hist": margin_hist,
        "margins_available": margins_available,
        "games_per_hour": games_per_hour,
        "eta_hours": eta_hours,
        "last_game_ts": last_game_ts,
        "momentum": momentum,
        "projected_verdict": projected,
        "recent": recent,
    }


def _wilson_ci(wins: int, n: int, *, z: float = 1.96) -> dict:
    """Wilson score 95% interval for a binomial proportion — well-behaved at
    small n / extreme p (unlike normal approx). Returns {lo, hi, center}."""
    if n == 0:
        return {"lo": 0.0, "hi": 1.0, "center": 0.0}
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return {"lo": max(0.0, center - half), "hi": min(1.0, center + half),
            "center": center}


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b) if b else 0


def seat_bias(loop_root, *, iter_n: int) -> dict:
    """Per-seat candidate winrate from an iteration's arena results — detects
    whether the gate is measuring skill or board-luck (a candidate that only
    wins from certain seats is a seating artifact, not an improvement)."""
    p = Path(loop_root) / f"iter_{iter_n}" / "arena" / "results.jsonl"
    by_seat = {s: {"cand_wins": 0, "champ_wins": 0, "games": 0} for s in range(4)}
    # by_rotation is the HONEST seating-fairness cut (Opus UX 2026-06-15):
    # per-board-seat winrate is misleading because cand only occupies seats
    # {0,2} on rot 0/2 and {1,3} on rot 1/3, so by_seat conflates seat-assignment
    # with winning. Grouping by rotation removes that confound.
    by_rot = {r: {"cand_wins": 0, "champ_wins": 0} for r in range(4)}
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
            role = r.get("winner_role")
            seat = r.get("winner_seat", -1)
            rot = r.get("rot")
            if seat in by_seat and role in ("cand", "champ"):
                by_seat[seat][f"{role}_wins"] += 1
            if rot in by_rot and role in ("cand", "champ"):
                by_rot[rot][f"{role}_wins"] += 1
    return {"by_seat": by_seat, "by_rotation": by_rot,
            "total_games": total, "timeouts": timeouts}


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
    status_ts = status.get("ts")
    ds_ts = ds_path.stat().st_mtime if ds_path.exists() else None
    # Pick the FRESHER heartbeat. status.json is written only at stage
    # TRANSITIONS; run_cycle rewrites daily_state.json (atomically) at every
    # stage, so its mtime tracks the live stage. Two cases this must handle:
    #  - self-play stage: no recent status.json -> daily_state mtime is the
    #    heartbeat (pilot 2026-06-14: dashboard read "NOT RUNNING" otherwise).
    #  - a NEW run resuming a root whose OLD status.json is hours stale: the
    #    fresh daily_state must WIN, else the dashboard shows the dead old
    #    iteration while a new one runs (production 2026-06-14).
    # Favor status.json on a near-tie (both written at the same stage
    # transition); daily_state wins only when it is MEANINGFULLY fresher (the
    # stale-old-status resume case), guarded by a margin so simultaneous writes
    # don't flip non-deterministically on filesystem mtime jitter.
    _FRESHER_MARGIN = 60.0
    use_ds = (ds_ts is not None and
              (not isinstance(status_ts, (int, float))
               or ds_ts > status_ts + _FRESHER_MARGIN))
    if use_ds:
        ts, stage, iter_n = ds_ts, ds.get("stage"), ds.get("iter")
    else:
        ts, stage, iter_n = status_ts, status.get("stage"), status.get("iter")
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


def selfplay_live(loop_root, *, iter_n: int, target: int, now=None) -> dict:
    """LIVE data-generation view for the RUNNING self-play stage.

    Reads the per-seed shards (`games.seed=*.parquet`) as they land — the
    pre-compaction form that exists DURING the stage — so the dashboard has a
    window into the hours-long self-play instead of going dark until it's done.
    Reports progress toward target, generation rate (from shard mtimes), data
    quality (game length, timeouts), and winner-seat balance (a board-fairness
    sanity check). Also matches the compacted form so it keeps working after the
    stage ends (glob is `games*.parquet`)."""
    import glob

    import pandas as pd
    now = now if now is not None else time.time()
    shards = glob.glob(str(Path(loop_root) / f"iter_{iter_n}" / "selfplay" /
                           "*" / "games*.parquet"))
    if not shards:
        return {"available": False, "games_done": 0, "target": target}
    lengths, timed, mtimes = [], 0, []
    seat_wins = {"0": 0, "1": 0, "2": 0, "3": 0}
    n = 0
    for f in shards:
        try:
            df = pd.read_parquet(f, columns=["winner", "length_in_moves",
                                             "timed_out"])
        except Exception:
            continue
        df = df[df["winner"] != -1]              # drop phantom aborted games
        if df.empty:
            continue
        n += len(df)
        lengths.extend(df["length_in_moves"].tolist())
        timed += int(df["timed_out"].sum())
        for w in df["winner"].tolist():
            if str(int(w)) in seat_wins:
                seat_wins[str(int(w))] += 1
        mtimes.append(Path(f).stat().st_mtime)
    if n == 0:
        return {"available": False, "games_done": 0, "target": target}
    # rate from the span of shard mtimes (first game finalised -> now). Using
    # `now` (not max mtime) so the rate reflects the live wall-clock, including
    # games currently in flight.
    span_h = max((now - min(mtimes)) / 3600.0, 1e-6)
    rate = n / span_h
    remaining = max(0, target - n)
    return {
        "available": True,
        "games_done": n,
        "target": target,
        "fraction": n / target if target else 0.0,
        "mean_length": sum(lengths) / len(lengths),
        "max_length": max(lengths),
        "timed_out_fraction": timed / n,
        "seat_wins": seat_wins,
        "games_per_hour": round(rate, 1),
        "eta_hours": round(remaining / rate, 2) if rate > 0 else None,
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
