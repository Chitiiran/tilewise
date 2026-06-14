"""gen_iter-based window + own-iter quota (spec 2026-06-14 §4). Replaces the
name-based fresh selection that caused the stale-data bug."""
from __future__ import annotations

import json

import pandas as pd


def _mk(root, name, n, gen_iter, rules_id="v3-full", gen="az_iter_1"):
    d = root / name
    d.mkdir(parents=True)
    pd.DataFrame({"seed": range(n), "winner": [0] * n}).to_parquet(d / "games.x.parquet")
    (d / "meta.json").write_text(json.dumps(
        {"rules_id": rules_id, "generator_name": gen, "gen_iter": gen_iter}))
    return d


def test_gen_window_includes_only_at_or_after_boundary(tmp_path):
    from catan_az.buffer import gen_window
    a = _mk(tmp_path, "a", 100, gen_iter=2)
    b = _mk(tmp_path, "b", 100, gen_iter=3)
    _mk(tmp_path, "c", 100, gen_iter=1)   # before boundary -> excluded
    sel = gen_window([a, b, tmp_path / "c"], gen_iter_min=2, rules_id="v3-full")
    assert set(sel) == {a, b}


def test_gen_window_filters_rules_id(tmp_path):
    from catan_az.buffer import gen_window
    a = _mk(tmp_path, "a", 100, gen_iter=2, rules_id="v3-full")
    b = _mk(tmp_path, "b", 100, gen_iter=2, rules_id="v4-trades")
    sel = gen_window([a, b], gen_iter_min=1, rules_id="v3-full")
    assert sel == [a]


def test_gen_window_legacy_dirs_excluded(tmp_path):
    """Dirs without gen_iter (old {champion,rules_id}) = gen_iter 0 -> excluded
    by any boundary >= 1 (old stale data can't pollute new windows)."""
    from catan_az.buffer import gen_window
    d = tmp_path / "legacy"
    d.mkdir()
    pd.DataFrame({"seed": [1], "winner": [0]}).to_parquet(d / "games.x.parquet")
    (d / "meta.json").write_text(json.dumps({"rules_id": "v3-full", "champion": "x"}))
    sel = gen_window([d], gen_iter_min=1, rules_id="v3-full")
    assert sel == []


def test_own_iter_games_counts_only_this_gen_iter(tmp_path):
    from catan_az.buffer import own_iter_games
    a = _mk(tmp_path, "a", 60, gen_iter=4)
    b = _mk(tmp_path, "b", 40, gen_iter=4)
    _mk(tmp_path, "c", 99, gen_iter=3)   # different iter -> not counted
    assert own_iter_games([a, b, tmp_path / "c"], gen_iter=4) == 100
