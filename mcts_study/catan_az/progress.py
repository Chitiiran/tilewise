"""PROGRESS.md — human-readable, append-only, one row per iteration.

The machine-readable journal.csv has the metrics and SELFPLAY.done has the exact
training dirs, but nothing summarized "what did iter-N actually train on, and did
it generate new data?" at a glance — so on 2026-06-14 the question "are we making
new data?" needed file archaeology. This is the at-a-glance answer, written each
iteration by the loop. The key column is **new_games** (0 = retrained on stale
data) and **trained_on** (which iterations the window's data came from)."""
from __future__ import annotations

from pathlib import Path

_HEADER = ("| iter | champion | generator | new_games | window_dirs | "
           "window_iters | verdict | winrate | draws |\n"
           "|---|---|---|---|---|---|---|---|---|\n")


def append_progress(loop_root, *, iter_n: int, champion: str, new_games: int,
                    window_dirs: int, verdict: str, winrate: float,
                    draw_rate: float, generator: str = None,
                    window_iters=None, all_from_iters=None,
                    window_games: int = None) -> None:
    """One PROGRESS.md row. `generator` = the net that made this iter's games
    (redesign 2026-06-14); `window_iters` = which gen_iters the training window
    spanned. (Legacy `all_from_iters`/`window_games`/no-generator calls still
    work.) `new_games=0` is flagged STALE — the bug we made visible."""
    p = Path(loop_root) / "PROGRESS.md"
    if not p.exists():
        p.write_text("# AZ daily training — progress log\n\n"
                     "One row per iteration. `new_games=0` means the iteration "
                     "retrained on EXISTING data (no fresh self-play) — watch "
                     "for this.\n\n" + _HEADER)
    src = window_iters if window_iters is not None else (all_from_iters or [])
    iters = sorted(set(int(i) for i in src))
    iters_str = ",".join(str(i) for i in iters)
    tag = "STALE " if new_games == 0 else ""
    gen = generator if generator is not None else champion
    row = (f"| {iter_n} | {champion} | {gen} | {new_games} | {window_dirs} | "
           f"{tag}{iters_str} | {verdict} | {winrate:.0%} | {draw_rate:.0%} |\n")
    with open(p, "a") as f:
        f.write(row)
