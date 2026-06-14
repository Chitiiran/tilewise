"""PROGRESS.md — a human-readable, append-only record of what each iteration
did and TRAINED ON (2026-06-14: we couldn't answer 'what data did iter-N use?'
at a glance because nothing summarized it; the facts were in SELFPLAY.done
manifests but not surfaced). One line per iteration."""
from __future__ import annotations


def test_append_progress_writes_header_then_row(tmp_path):
    from catan_az.progress import append_progress
    append_progress(tmp_path, iter_n=3, champion="az_iter_1",
                    new_games=810, window_games=10000, window_dirs=13,
                    all_from_iters=[3], verdict="hold", winrate=0.437,
                    draw_rate=0.275)
    text = (tmp_path / "PROGRESS.md").read_text()
    assert "iter" in text and "new_games" in text   # header
    assert "3" in text and "az_iter_1" in text and "hold" in text


def test_append_progress_is_append_only(tmp_path):
    from catan_az.progress import append_progress
    append_progress(tmp_path, iter_n=3, champion="c", new_games=810,
                    window_games=1, window_dirs=1, all_from_iters=[3],
                    verdict="hold", winrate=0.4, draw_rate=0.2)
    append_progress(tmp_path, iter_n=4, champion="c", new_games=0,
                    window_games=1, window_dirs=21, all_from_iters=[3],
                    verdict="hold", winrate=0.5, draw_rate=0.38)
    # data rows start "| <number> |"; exclude the header ("| iter |") + separator
    import re
    lines = [l for l in (tmp_path / "PROGRESS.md").read_text().splitlines()
             if re.match(r"\|\s*\d+\s*\|", l)]
    assert len(lines) == 2   # two data rows, header preserved


def test_progress_flags_zero_new_games(tmp_path):
    """The row must make 'this iteration generated NO new data' obvious — the
    exact failure mode we missed (iters 4/5 retrained on iter-3's pool)."""
    from catan_az.progress import append_progress
    append_progress(tmp_path, iter_n=4, champion="az_iter_1", new_games=0,
                    window_games=1, window_dirs=21, all_from_iters=[3],
                    verdict="hold", winrate=0.5, draw_rate=0.38)
    text = (tmp_path / "PROGRESS.md").read_text()
    # new_games=0 + trained-on-data-from earlier iters is visible in the row
    assert "0" in text
    assert "STALE" in text or "iter 3" in text or "[3]" in text
