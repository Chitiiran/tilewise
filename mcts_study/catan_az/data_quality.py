"""Self-play data-quality summary + degeneracy gate (observability minimum,
shake-out journal 2026-06-19 §3): nobody computed a win/draw/timeout/seat/
length summary for a self-play run before this — data problems (e.g. an
all-timeout run) were invisible until training wasted hours on them.

Reads the same `games*.parquet` shards as catan_az.buffer.count_games and
catan_az.analytics.selfplay_health/selfplay_live (recorder.py's _GameRow
schema: seed, winner [-1 = no decisive winner], length_in_moves, timed_out).
This module intentionally reuses that glob + column convention rather than
inventing a new reader.

Terminology (matches the rest of catan_az, e.g. arena.py/analytics.py):
  - "no winner" / draw: winner == -1. In self-play this happens on any game
    that didn't reach a decisive winner (VP tie, aborted, OR timed out — a
    timed-out self-play game IS recorded winner=-1; self-play has no VP-leader
    tiebreak the way the arena does). timed_out is a separate, overlapping
    boolean field: a timeout always implies winner == -1 here, but not every
    winner == -1 game is a timeout (skip_game()/recorder.py fallback finalize
    can also write winner=-1 without timed_out).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Degeneracy thresholds (plan: >20% timeouts or >40% draws or 0 winners).
# Named constants, not magic numbers — degeneracy_verdict() lets a cfg object
# override any of them via matching attribute names (same convention as
# arena.py's cfg.arena_max_draw_rate / cfg.arena_min_decisive).
DATA_QUALITY_MAX_TIMEOUT_RATE = 0.20
DATA_QUALITY_MAX_DRAW_RATE = 0.40

def summarize_selfplay_dir(run_dir: Path) -> dict:
    """Win/draw/timeout/seat/length summary over a self-play run dir's
    games*.parquet shards (per-seed or compacted — same glob as
    buffer.count_games / analytics.selfplay_health).

    Returns:
      games: total game rows across all shards (NOT deduped by seed — a
        crash-recovered dir can have both per-seed and compacted shards;
        summarize_selfplay_dir is a diagnostic snapshot, not the training
        game count, so it intentionally does NOT dedup like buffer.count_games).
      winners_by_seat: {"0".."3": count} of decisive wins per seat.
      timeouts: count of timed_out == True rows.
      no_winner: count of winner == -1 rows (draws/timeouts/aborted).
      length_p50 / length_p90 / length_max: game-length (moves) percentiles.

    Tolerant of legacy/minimal shards missing length_in_moves or timed_out
    (e.g. test fixtures with only seed+winner) — those default to 0/False
    per row rather than dropping the whole shard, matching buffer.count_games'
    legacy-fallback convention.
    """
    run_dir = Path(run_dir)
    frames = []
    for shard in run_dir.glob("games*.parquet"):
        try:
            df = pd.read_parquet(shard)
        except Exception:
            continue
        if "winner" not in df.columns:
            continue   # not a games shard we can summarize
        if "length_in_moves" not in df.columns:
            df = df.assign(length_in_moves=0)
        if "timed_out" not in df.columns:
            df = df.assign(timed_out=False)
        frames.append(df[["seed", "winner", "length_in_moves", "timed_out"]])
    if not frames:
        return {
            "games": 0,
            "winners_by_seat": {"0": 0, "1": 0, "2": 0, "3": 0},
            "timeouts": 0,
            "no_winner": 0,
            "length_p50": 0,
            "length_p90": 0,
            "length_max": 0,
        }
    df = pd.concat(frames, ignore_index=True)

    winners_by_seat = {str(s): 0 for s in range(4)}
    decisive = df[df["winner"] != -1]
    for w in decisive["winner"].tolist():
        key = str(int(w))
        if key in winners_by_seat:
            winners_by_seat[key] += 1

    lengths = df["length_in_moves"].tolist()
    lengths_sorted = sorted(lengths)

    def _pct(p: float) -> float:
        if not lengths_sorted:
            return 0
        idx = min(len(lengths_sorted) - 1, max(0, round(p * (len(lengths_sorted) - 1))))
        return lengths_sorted[idx]

    return {
        "games": int(len(df)),
        "winners_by_seat": winners_by_seat,
        "timeouts": int(df["timed_out"].sum()),
        "no_winner": int((df["winner"] == -1).sum()),
        "length_p50": _pct(0.50),
        "length_p90": _pct(0.90),
        "length_max": int(max(lengths_sorted)) if lengths_sorted else 0,
    }


def degeneracy_verdict(summary: dict, cfg) -> str:
    """'ok' | 'degenerate'.

    Degenerate if: >20% timeouts, OR >40% draws (no_winner), OR 0 decisive
    winners at all (includes the empty-dir / 0-games case — never silently
    'ok' on no data). Thresholds come from cfg when present (getattr
    fallback, matching arena.py's should_promote), else the module's named
    default constants.
    """
    max_timeout_rate = getattr(cfg, "data_quality_max_timeout_rate",
                               DATA_QUALITY_MAX_TIMEOUT_RATE)
    max_draw_rate = getattr(cfg, "data_quality_max_draw_rate",
                            DATA_QUALITY_MAX_DRAW_RATE)

    games = summary.get("games", 0)
    total_winners = sum(summary.get("winners_by_seat", {}).values())
    if games == 0 or total_winners == 0:
        return "degenerate"

    timeout_rate = summary.get("timeouts", 0) / games
    draw_rate = summary.get("no_winner", 0) / games
    if timeout_rate > max_timeout_rate:
        return "degenerate"
    if draw_rate > max_draw_rate:
        return "degenerate"
    return "ok"
