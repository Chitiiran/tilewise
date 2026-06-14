"""PROGRESS.md — human-readable, append-only, one row per iteration.

The machine-readable journal.csv has the metrics and SELFPLAY.done has the exact
training dirs, but nothing summarized "what did iter-N actually train on, and did
it generate new data?" at a glance — so on 2026-06-14 the question "are we making
new data?" needed file archaeology. This is the at-a-glance answer, written each
iteration by the loop. The key column is **new_games** (0 = retrained on stale
data) and **trained_on** (which iterations the window's data came from)."""
from __future__ import annotations

from pathlib import Path

_HEADER = ("| iter | champion | new_games | window_dirs | trained_on | "
           "verdict | winrate | draws |\n"
           "|---|---|---|---|---|---|---|---|\n")


def append_progress(loop_root, *, iter_n: int, champion: str, new_games: int,
                    window_games: int, window_dirs: int, all_from_iters,
                    verdict: str, winrate: float, draw_rate: float) -> None:
    p = Path(loop_root) / "PROGRESS.md"
    if not p.exists():
        p.write_text("# AZ daily training — progress log\n\n"
                     "One row per iteration. `new_games=0` means the iteration "
                     "retrained on EXISTING data (no fresh self-play) — watch "
                     "for this.\n\n" + _HEADER)
    iters = sorted(set(int(i) for i in all_from_iters))
    trained = ("STALE iter " + ",".join(str(i) for i in iters)
               if new_games == 0 else "iter " + ",".join(str(i) for i in iters))
    row = (f"| {iter_n} | {champion} | {new_games} | {window_dirs} | "
           f"{trained} ({iters}) | {verdict} | {winrate:.0%} | "
           f"{draw_rate:.0%} |\n")
    with open(p, "a") as f:
        f.write(row)
