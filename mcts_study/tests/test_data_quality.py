"""Self-play data-quality summary + degeneracy gate (Task 7, observability
minimums — shake-out journal 2026-06-19 §3: a run's win/draw/timeout/seat/
length profile was invisible until training wasted hours on it).

Fixture pattern reused from test_az_buffer.py / test_az_loop.py /
test_az_analytics_live.py: a self-play run dir is a plain directory holding
`games*.parquet` shards written via `pandas.DataFrame(...).to_parquet(...)`,
matching the real recorder's schema (recorder.py `_GameRow`): seed, winner
(-1 = no decisive winner), final_vp, length_in_moves, timed_out.
"""
from __future__ import annotations

import pandas as pd
import pytest


def _mk_run_dir(root, name, games: list[dict]):
    """games: list of dicts with keys seed, winner, length_in_moves, timed_out.
    Mirrors recorder.py's _GameRow fields (final_vp omitted — unused here)."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    if games:
        df = pd.DataFrame(games)
        df.to_parquet(d / "games.abc123.parquet")
    return d


def _game(seed, winner, length, timed_out=False):
    return {"seed": seed, "winner": winner, "length_in_moves": length,
            "timed_out": timed_out}


# ---- summarize_selfplay_dir ----

def test_summarize_selfplay_dir_basic_fields(tmp_path):
    from catan_az.data_quality import summarize_selfplay_dir
    # 10 games, lengths 100..1000 (step 100), so p50/p90 are unambiguous under
    # nearest-rank percentile (same convention as analytics.py's per-game
    # quartiles: round(p * (n-1)) into the sorted array).
    games = [_game(i, winner=(i % 4 if i < 9 else -1), length=(i + 1) * 100,
                   timed_out=(i == 9))
             for i in range(10)]
    d = _mk_run_dir(tmp_path, "run1", games)
    s = summarize_selfplay_dir(d)
    assert s["games"] == 10
    # seeds 0..8 decisive winners i%4: seat0={0,4,8}=3, seat1={1,5}=2,
    # seat2={2,6}=2, seat3={3,7}=2; seed 9 -> winner=-1 (excluded)
    assert s["winners_by_seat"] == {"0": 3, "1": 2, "2": 2, "3": 2}
    assert s["timeouts"] == 1
    assert s["no_winner"] == 1          # winner == -1 games
    lengths_sorted = sorted((i + 1) * 100 for i in range(10))
    assert s["length_p50"] == lengths_sorted[round(0.50 * 9)]
    assert s["length_p90"] == lengths_sorted[round(0.90 * 9)]
    assert s["length_max"] == 1000


def test_summarize_selfplay_dir_spans_multiple_shards(tmp_path):
    """A run dir has multiple games.<label>.parquet shards (per-seed or
    compacted); the summary must read all of them, matching buffer.count_games."""
    from catan_az.data_quality import summarize_selfplay_dir
    d = tmp_path / "run2"
    d.mkdir()
    pd.DataFrame([_game(0, 0, 100), _game(1, 1, 150)]).to_parquet(d / "games.a.parquet")
    pd.DataFrame([_game(2, 2, 200)]).to_parquet(d / "games.b.parquet")
    s = summarize_selfplay_dir(d)
    assert s["games"] == 3
    assert s["winners_by_seat"]["2"] == 1


def test_summarize_selfplay_dir_empty(tmp_path):
    from catan_az.data_quality import summarize_selfplay_dir
    d = tmp_path / "empty"
    d.mkdir()
    s = summarize_selfplay_dir(d)
    assert s["games"] == 0
    assert s["timeouts"] == 0
    assert s["no_winner"] == 0
    assert s["length_p50"] == 0
    assert s["length_p90"] == 0
    assert s["length_max"] == 0


# ---- degeneracy_verdict ----

class _Cfg:
    """Stand-in for AzConfig — degeneracy_verdict must work off getattr
    fallbacks (mirrors arena.py's should_promote/cfg convention) so a plain
    object without the AZ-specific fields still degrades to the named
    default constants."""
    pass


def test_degeneracy_verdict_all_timeout_is_degenerate(tmp_path):
    from catan_az.data_quality import degeneracy_verdict, summarize_selfplay_dir
    games = [_game(i, winner=-1, length=999_999, timed_out=True) for i in range(10)]
    d = _mk_run_dir(tmp_path, "all_timeout", games)
    s = summarize_selfplay_dir(d)
    assert degeneracy_verdict(s, _Cfg()) == "degenerate"


def test_degeneracy_verdict_healthy_dir_is_ok(tmp_path):
    from catan_az.data_quality import degeneracy_verdict, summarize_selfplay_dir
    games = [_game(i, winner=i % 4, length=300 + i) for i in range(20)]
    d = _mk_run_dir(tmp_path, "healthy", games)
    s = summarize_selfplay_dir(d)
    assert degeneracy_verdict(s, _Cfg()) == "ok"


def test_degeneracy_verdict_high_timeout_rate_is_degenerate(tmp_path):
    """> 20% timeouts -> degenerate, even with plenty of decisive winners."""
    from catan_az.data_quality import degeneracy_verdict, summarize_selfplay_dir
    games = [_game(i, winner=i % 4, length=300) for i in range(75)]
    games += [_game(1000 + i, winner=-1, length=999_999, timed_out=True)
              for i in range(25)]   # 25/100 = 25% timeouts
    d = _mk_run_dir(tmp_path, "hi_timeout", games)
    s = summarize_selfplay_dir(d)
    assert degeneracy_verdict(s, _Cfg()) == "degenerate"


def test_degeneracy_verdict_high_draw_rate_is_degenerate(tmp_path):
    """> 40% draws (winner == -1, NOT timed_out) -> degenerate."""
    from catan_az.data_quality import degeneracy_verdict, summarize_selfplay_dir
    games = [_game(i, winner=i % 4, length=300) for i in range(55)]
    games += [_game(1000 + i, winner=-1, length=300, timed_out=False)
              for i in range(45)]   # 45/100 = 45% draws (not timeouts)
    d = _mk_run_dir(tmp_path, "hi_draw", games)
    s = summarize_selfplay_dir(d)
    assert degeneracy_verdict(s, _Cfg()) == "degenerate"


def test_degeneracy_verdict_zero_winners_is_degenerate(tmp_path):
    from catan_az.data_quality import degeneracy_verdict, summarize_selfplay_dir
    games = [_game(i, winner=-1, length=300, timed_out=False) for i in range(5)]
    d = _mk_run_dir(tmp_path, "zero_winners", games)
    s = summarize_selfplay_dir(d)
    assert degeneracy_verdict(s, _Cfg()) == "degenerate"


def test_degeneracy_verdict_empty_dir_is_degenerate(tmp_path):
    """No games at all -> 0 winners -> degenerate (never silently 'ok')."""
    from catan_az.data_quality import degeneracy_verdict, summarize_selfplay_dir
    d = tmp_path / "empty"
    d.mkdir()
    s = summarize_selfplay_dir(d)
    assert degeneracy_verdict(s, _Cfg()) == "degenerate"


def test_degeneracy_thresholds_respect_cfg_override(tmp_path):
    """Thresholds are named constants but overridable via cfg attributes
    (matching arena.py's cfg.arena_max_draw_rate convention) — a caller with
    a looser tolerance can still call this 'ok'."""
    from catan_az.data_quality import degeneracy_verdict, summarize_selfplay_dir
    games = [_game(i, winner=i % 4, length=300) for i in range(70)]
    games += [_game(1000 + i, winner=-1, length=999_999, timed_out=True)
              for i in range(30)]   # 30% timeouts: degenerate under default 20%
    d = _mk_run_dir(tmp_path, "cfg_override", games)
    s = summarize_selfplay_dir(d)
    assert degeneracy_verdict(s, _Cfg()) == "degenerate"

    class _LooseCfg:
        data_quality_max_timeout_rate = 0.50
    assert degeneracy_verdict(s, _LooseCfg()) == "ok"
