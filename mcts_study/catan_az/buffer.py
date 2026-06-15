"""Sliding-window replay buffer (spec §2 BUFFER).

Canonical AZ trains on a window of RECENT self-play games, not all history —
old games reflect an older (weaker) policy and drag the target distribution
backward. The window is selected at run-dir granularity (train_main consumes
whole run dirs); newest-first dirs are taken until the window is filled.

No torch imports — pure pandas bookkeeping, cheap to unit test.
"""
from __future__ import annotations

import json as _json
import math as _math
from pathlib import Path

import pandas as pd


def count_games(run_dir: Path) -> int:
    """Unique FINISHED games across a run dir's games*.parquet shards.

    Dedup by seed and drop phantom winner=-1 rows (recorder.py:192-201 fallback
    finalize for aborted games). After a crash both per-seed shards and a
    compacted/_remainder shard can coexist, so a naive sum double-counts the same
    seed and over-reports the quota (deep-inspect LOW, cheap fix)."""
    seeds: set[int] = set()
    for shard in Path(run_dir).glob("games*.parquet"):
        try:
            cols = pd.read_parquet(shard, columns=["seed", "winner"])
            cols = cols[cols["winner"] != -1]
        except (ValueError, KeyError):
            cols = pd.read_parquet(shard, columns=["seed"])   # legacy: no winner
        seeds.update(int(s) for s in cols["seed"].tolist())
    return len(seeds)


def _read_meta(run_dir: Path) -> dict:
    """Run-dir tags {rules_id, champion}, or {} if untagged (legacy dirs)."""
    p = Path(run_dir) / "meta.json"
    return _json.loads(p.read_text()) if p.exists() else {}


def gen_window(dirs, *, gen_iter_min: int, rules_id: str) -> list:
    """Run dirs whose gen_iter >= gen_iter_min AND rules_id matches — the
    recency-based training window (spec 2026-06-14 §4). Missing gen_iter
    (legacy {champion,rules_id} dirs) = 0, so they're excluded by any boundary
    >= 1. This replaces name-based fresh selection and fixes the stale-data bug:
    a sticky champion no longer makes old games count as fresh forever."""
    out = []
    for d in dirs:
        m = _read_meta(d)
        if m.get("rules_id") != rules_id:
            continue
        if int(m.get("gen_iter", 0)) >= gen_iter_min:
            out.append(d)
    return out


def own_iter_games(dirs, *, gen_iter: int) -> int:
    """Games generated specifically in `gen_iter` — the per-iteration quota
    counter. Counting ONLY this iteration's own games is what kills the bug
    where other iterations' games silently satisfied the quota (2026-06-14)."""
    total = 0
    for d in dirs:
        if int(_read_meta(d).get("gen_iter", -1)) == gen_iter:
            total += count_games(d)
    return total


def fresh_deficit(iter_dirs_newest_first, *, champion: str, rules_id: str,
                  window_games: int, fresh_ratio: float) -> int:
    """Games still needed so current-champion games are >= fresh_ratio of the
    window. Counts only dirs tagged with `champion` AND `rules_id` (spec §5)."""
    fresh = 0
    for d in iter_dirs_newest_first:
        m = _read_meta(d)
        if m.get("champion") == champion and m.get("rules_id") == rules_id:
            fresh += count_games(d)
    target = _math.ceil(fresh_ratio * window_games)
    return max(0, target - fresh)


def select_window(iter_dirs_newest_first: list[Path], window_games: int,
                  rules_id: str | None = None) -> list[Path]:
    """Take newest-first run dirs until their game counts sum >= window_games.

    Caller supplies dirs ordered newest-first (the loop knows iteration
    order). Empty dirs are skipped. When `rules_id` is given, only dirs tagged
    with that rules_id are eligible (so a future engine-fidelity bump flushes
    stale-rules games instead of poisoning the window). Raises if no dir
    contains any eligible game — training on nothing should fail loudly.
    """
    selected: list[Path] = []
    total = 0
    for d in iter_dirs_newest_first:
        if rules_id is not None and _read_meta(d).get("rules_id") != rules_id:
            continue
        n = count_games(d)
        if n == 0:
            continue
        selected.append(d)
        total += n
        if total >= window_games:
            break
    if not selected:
        raise ValueError(
            f"buffer: no games in any of {len(iter_dirs_newest_first)} run dirs")
    return selected
