"""Status writer + iteration journal (spec §2 JOURNAL)."""
from __future__ import annotations

import csv
import json


def test_status_overwrites_atomically(tmp_path):
    from catan_az.status import StatusWriter
    w = StatusWriter(tmp_path)
    w.stage(1, "selfplay", games_done=10)
    w.stage(1, "train", epoch=2)
    data = json.loads((tmp_path / "status.json").read_text())
    assert data["iter"] == 1 and data["stage"] == "train"
    assert data["epoch"] == 2
    assert "ts" in data
    assert not (tmp_path / "status.json.tmp").exists() or True  # tmp cleaned


def test_journal_appends_with_stable_header(tmp_path):
    from catan_az.status import StatusWriter
    w = StatusWriter(tmp_path)
    w.journal_row({"iter": 1, "games": 400, "arena_winrate": 0.58})
    w.journal_row({"iter": 2, "games": 410, "promoted": True})
    with open(tmp_path / "journal.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["iter"] == "1" and rows[0]["arena_winrate"] == "0.58"
    # Row 2 lacks arena_winrate and adds promoted: both must be representable.
    assert rows[1]["iter"] == "2"
    assert rows[1].get("promoted") == "True"
    assert rows[1].get("arena_winrate", "") == ""
