"""rules_id-tagged dirs + fresh-deficit window (spec 2026-06-13 §5)."""
from __future__ import annotations

import json

import pandas as pd


def _mk_dir(root, name, n, rules_id="v3-full", champ="cell6"):
    d = root / name
    d.mkdir(parents=True)
    pd.DataFrame({"seed": range(n), "winner": [0] * n}).to_parquet(
        d / "games.x.parquet")
    (d / "meta.json").write_text(json.dumps({"rules_id": rules_id,
                                             "champion": champ}))
    return d


def test_fresh_deficit_counts_only_current_champion_and_rules(tmp_path):
    from catan_az.buffer import fresh_deficit
    _mk_dir(tmp_path, "old", 300, champ="cell6")
    _mk_dir(tmp_path, "new", 100, champ="az_iter_1")
    d = fresh_deficit([tmp_path / "new", tmp_path / "old"],
                      champion="az_iter_1", rules_id="v3-full",
                      window_games=1000, fresh_ratio=0.70)
    assert d == 600   # ceil(0.70*1000)=700, have 100 fresh


def test_fresh_deficit_zero_when_met(tmp_path):
    from catan_az.buffer import fresh_deficit
    _mk_dir(tmp_path, "new", 800, champ="az_iter_1")
    d = fresh_deficit([tmp_path / "new"], champion="az_iter_1",
                      rules_id="v3-full", window_games=1000, fresh_ratio=0.70)
    assert d == 0   # 800 >= 700


def test_select_window_filters_rules_id(tmp_path):
    from catan_az.buffer import select_window
    a = _mk_dir(tmp_path, "a", 100, rules_id="v3-full")
    b = _mk_dir(tmp_path, "b", 100, rules_id="v4-trades")
    sel = select_window([a, b], window_games=1000, rules_id="v3-full")
    assert sel == [a]


def test_select_window_no_rules_filter_backcompat(tmp_path):
    from catan_az.buffer import select_window
    a = _mk_dir(tmp_path, "a", 100)
    b = _mk_dir(tmp_path, "b", 100, rules_id="v4-trades")
    sel = select_window([a, b], window_games=1000)   # no rules_id -> old behavior
    assert sel == [a, b]
