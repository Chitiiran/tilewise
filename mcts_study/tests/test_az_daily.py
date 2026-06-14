"""Daily driver: manifest + run_day loop + resume numbering. Stage fns
injected/monkeypatched — no GPU. (The run_cycle/generate/terminal behavior is
covered by test_az_redesign_cycle.py + test_az_generate_iter.py after the
2026-06-14 candidate-self-play redesign.)"""
from __future__ import annotations


def test_manifest_round_trip(tmp_path):
    from catan_az.daily import DailyManifest
    m = DailyManifest(iter=3, stage="selfplay", champion="az_iter_1",
                      fresh_target=700, fresh_done=120, rules_id="v3-full")
    m.save(tmp_path)
    back = DailyManifest.load(tmp_path)
    assert back == m


def test_run_day_runs_until_stop(tmp_path):
    from catan_az.daily import run_day
    from catan_az.config import AzConfig
    calls = []

    def fake_cycle(cfg, loop_root, iter_n, capped_procs):
        calls.append(iter_n)
        if iter_n == 2:
            (loop_root / "STOP").write_text("")
        return "promote"

    run_day(AzConfig(), loop_root=tmp_path, capped_procs=5,
            cycle_fn=fake_cycle, max_iters=10)
    assert calls == [1, 2]


def test_run_day_respects_max_iters(tmp_path):
    from catan_az.daily import run_day
    from catan_az.config import AzConfig
    calls = []
    run_day(AzConfig(), loop_root=tmp_path, capped_procs=5,
            cycle_fn=lambda cfg, root, n, procs: calls.append(n) or "hold",
            max_iters=3)
    assert len(calls) == 3


def test_resumes_incomplete_iteration(tmp_path):
    """An iteration with a dir but no PUBLISH.done must be RESUMED (its
    self-play + train are salvaged via done-markers), not skipped to the next
    number (2026-06-14: iter-3 training crash wasted 6h when resume jumped to
    iter-4)."""
    from catan_az.daily import _next_iter_number
    (tmp_path / "iter_1").mkdir()
    (tmp_path / "iter_1" / "PUBLISH.done").write_text("{}")
    (tmp_path / "iter_3").mkdir()
    (tmp_path / "iter_3" / "SELFPLAY.done").write_text("{}")  # no PUBLISH.done
    assert _next_iter_number(tmp_path) == 3   # resume iter_3, not 4


def test_next_iter_after_all_complete(tmp_path):
    from catan_az.daily import _next_iter_number
    (tmp_path / "iter_1").mkdir()
    (tmp_path / "iter_1" / "PUBLISH.done").write_text("{}")
    (tmp_path / "iter_2").mkdir()
    (tmp_path / "iter_2" / "PUBLISH.done").write_text("{}")
    assert _next_iter_number(tmp_path) == 3   # all done -> next is 3


def test_next_iter_empty(tmp_path):
    from catan_az.daily import _next_iter_number
    assert _next_iter_number(tmp_path) == 1
