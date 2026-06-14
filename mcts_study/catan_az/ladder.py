"""Elo ladder + champion registry (spec §2 PUBLISH).

State lives in <root>/ladder.json. Old champions stay on the ladder —
they feed the web difficulty tiers (humans always have opponents at every
strength). Writes are atomic (tmp + rename) so a crash mid-write never
corrupts the registry.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

K_FACTOR = 24.0


def _expected(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


class Ladder:
    def __init__(self, root: Path, *, champion_checkpoint: str | None = None,
                 champion_name: str = "seed") -> None:
        self.path = Path(root) / "ladder.json"
        if self.path.exists():
            self._data = json.loads(self.path.read_text())
        else:
            if champion_checkpoint is None:
                raise ValueError("fresh ladder needs champion_checkpoint")
            self._data = {
                "entries": {champion_name: {
                    "name": champion_name, "checkpoint": champion_checkpoint,
                    "elo": 1000.0, "games": 0, "created_iter": 0}},
                "champion": champion_name,
                "history": [],
                "last_promotion_iter": 0,
            }
            self._save()

    # -- queries ----------------------------------------------------------
    def champion(self) -> dict:
        return self._data["entries"][self._data["champion"]]

    def entry(self, name: str) -> dict | None:
        return self._data["entries"].get(name)

    def history(self) -> list[dict]:
        return self._data["history"]

    def last_promotion_iter(self) -> int:
        """Iteration of the last promotion — the training-window reset boundary
        (spec 2026-06-14). 0 on a fresh ladder."""
        return self._data.get("last_promotion_iter", 0)

    # -- mutations --------------------------------------------------------
    def register_candidate(self, name: str, checkpoint: str, *, created_iter: int) -> None:
        # New candidates start at the champion's rating: they're a continuation
        # of it, not an unknown — keeps early Elo deltas meaningful.
        self._data["entries"][name] = {
            "name": name, "checkpoint": checkpoint,
            "elo": self.champion()["elo"], "games": 0,
            "created_iter": created_iter}
        self._save()

    def record_arena(self, name_a: str, name_b: str, *, wins_a: int,
                     wins_b: int, draws: int = 0) -> None:
        """Zero-sum Elo update from an arena series (draws count half)."""
        a, b = self._data["entries"][name_a], self._data["entries"][name_b]
        n = wins_a + wins_b + draws
        if n == 0:
            return
        score_a = (wins_a + 0.5 * draws) / n
        delta = K_FACTOR * (score_a - _expected(a["elo"], b["elo"]))
        a["elo"] += delta
        b["elo"] -= delta
        a["games"] += n
        b["games"] += n
        self._save()

    def promote(self, name: str, *, promoted_at_iter: int | None = None) -> None:
        """Crown `name`. Idempotent on the reset boundary: if `name` is already
        champion (run_iteration's PUBLISH promoted it; run_cycle re-calls to set
        the boundary), only update last_promotion_iter — don't append a
        duplicate history entry (spec 2026-06-14)."""
        if name not in self._data["entries"]:
            raise KeyError(name)
        if self._data["champion"] != name:
            self._data["history"].append({"promoted": name,
                                          "previous": self._data["champion"]})
            self._data["champion"] = name
        if promoted_at_iter is not None:
            self._data["last_promotion_iter"] = promoted_at_iter
        self._save()

    # -- io ----------------------------------------------------------------
    def _save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        os.replace(tmp, self.path)
