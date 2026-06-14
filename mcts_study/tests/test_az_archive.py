"""HDD archive of out-of-window games (spec 2026-06-13 §8)."""
from __future__ import annotations

import json

import pandas as pd


def _mk(root, name, n):
    d = root / name
    d.mkdir(parents=True)
    pd.DataFrame({"seed": range(n), "winner": [0] * n}).to_parquet(
        d / "games.x.parquet")
    (d / "meta.json").write_text(json.dumps({"rules_id": "v3-full",
                                            "champion": "c"}))
    return d


def test_archive_moves_only_out_of_window(tmp_path):
    from catan_az.archive import archive_out_of_window
    hdd = tmp_path / "hdd"
    hdd.mkdir()
    in_win = _mk(tmp_path, "in", 10)
    out_win = _mk(tmp_path, "out", 10)
    archive_out_of_window(window_dirs=[in_win], all_dirs=[in_win, out_win],
                          archive_root=hdd, rules_id="v3-full")
    assert not list(out_win.glob("*.parquet"))
    assert (out_win / "ARCHIVED.txt").exists()
    assert list(in_win.glob("*.parquet"))
    assert list(hdd.rglob("games.x.parquet"))


def test_archive_idempotent(tmp_path):
    from catan_az.archive import archive_out_of_window
    hdd = tmp_path / "hdd"
    hdd.mkdir()
    in_win = _mk(tmp_path, "in", 10)
    out_win = _mk(tmp_path, "out", 10)
    archive_out_of_window(window_dirs=[in_win], all_dirs=[in_win, out_win],
                          archive_root=hdd, rules_id="v3-full")
    archive_out_of_window(window_dirs=[in_win], all_dirs=[in_win, out_win],
                          archive_root=hdd, rules_id="v3-full")
    assert (out_win / "ARCHIVED.txt").exists()
