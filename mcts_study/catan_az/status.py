"""Status heartbeat + iteration journal (spec §2 JOURNAL).

status.json: latest stage of the current iteration, atomically replaced —
the dashboard's single source of truth and the "is it stuck?" probe.
journal.csv: one row per iteration, append-only; the Elo-vs-games curve we
extrapolate time-to-strength from lives here. Header grows as new fields
appear (file is rewritten — it's tiny — so old rows stay readable).
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path


class StatusWriter:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.status_path = self.root / "status.json"
        self.journal_path = self.root / "journal.csv"

    def stage(self, iter_n: int, stage: str, **fields) -> None:
        data = {"iter": iter_n, "stage": stage, "ts": time.time(), **fields}
        tmp = self.status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.status_path)

    def journal_row(self, row: dict) -> None:
        existing: list[dict] = []
        header: list[str] = []
        if self.journal_path.exists():
            with open(self.journal_path, newline="") as f:
                reader = csv.DictReader(f)
                header = list(reader.fieldnames or [])
                existing = list(reader)
        # Union header, preserving existing order; new keys append at the end.
        for k in row:
            if k not in header:
                header.append(k)
        with open(self.journal_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header, restval="")
            writer.writeheader()
            for r in existing:
                writer.writerow(r)
            writer.writerow({k: row.get(k, "") for k in header})
