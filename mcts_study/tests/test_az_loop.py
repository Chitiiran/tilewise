"""Orchestrator (spec §2): stage order, resume, promote/hold, STOP sentinel.

Stage functions are injected fakes — these tests prove the loop's
bookkeeping without GPUs or games.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from catan_az.arena import ArenaResult
from catan_az.config import AzConfig


def _mk_ckpt(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-weights")
    return path


@pytest.fixture()
def loop_env(tmp_path):
    """loop_root with a seeded ladder + fake champion checkpoint + call log."""
    from catan_az.ladder import Ladder
    root = tmp_path / "loop"
    root.mkdir()
    champ = _mk_ckpt(tmp_path / "ckpts" / "cell6.pt")
    Ladder(root, champion_checkpoint=str(champ), champion_name="cell6")
    calls: list[str] = []

    def fake_selfplay(cfg, iter_dir, champion_ckpt):
        calls.append("selfplay")
        d = iter_dir / "selfplay_run"
        d.mkdir(parents=True, exist_ok=True)
        # Minimal games shard so the buffer's window selection has content.
        import pandas as pd
        pd.DataFrame({"seed": range(5), "winner": [0, 1, 2, 3, 0]}
                     ).to_parquet(d / "games.fake.parquet")
        return [d]

    def fake_train(cfg, run_dirs, iter_dir, init_ckpt):
        calls.append("train")
        return _mk_ckpt(iter_dir / "candidate.pt")

    def winning_arena(cfg, cand, champ, iter_dir):
        calls.append("arena")
        return ArenaResult(wins_cand=80, wins_champ=40, draws=0, timeouts=0)

    def losing_arena(cfg, cand, champ, iter_dir):
        calls.append("arena")
        return ArenaResult(wins_cand=40, wins_champ=80, draws=0, timeouts=0)

    return root, calls, fake_selfplay, fake_train, winning_arena, losing_arena


def test_iteration_runs_stages_in_order_and_journals(loop_env):
    from catan_az.loop import run_iteration
    root, calls, sp, tr, win_arena, _ = loop_env
    cfg = AzConfig()
    verdict = run_iteration(cfg, root, 1, selfplay_fn=sp, train_fn=tr,
                            arena_fn=win_arena)
    assert calls == ["selfplay", "train", "arena"]
    assert verdict == "promote"
    journal = (root / "journal.csv").read_text()
    assert "promote" in journal
    status = json.loads((root / "status.json").read_text())
    assert status["stage"] == "done"


def test_promote_updates_ladder_and_copies_checkpoint(loop_env):
    from catan_az.ladder import Ladder
    from catan_az.loop import run_iteration
    root, _, sp, tr, win_arena, _ = loop_env
    run_iteration(AzConfig(), root, 1, selfplay_fn=sp, train_fn=tr,
                  arena_fn=win_arena)
    ladder = Ladder(root)
    assert ladder.champion()["name"] == "az_iter_1"
    promoted = Path(ladder.champion()["checkpoint"])
    assert promoted.exists()
    assert promoted.name == "az_iter_1.pt"


def test_hold_keeps_champion(loop_env):
    from catan_az.ladder import Ladder
    from catan_az.loop import run_iteration
    root, _, sp, tr, _, lose_arena = loop_env
    verdict = run_iteration(AzConfig(), root, 1, selfplay_fn=sp, train_fn=tr,
                            arena_fn=lose_arena)
    assert verdict == "hold"
    assert Ladder(root).champion()["name"] == "cell6"


def test_resume_skips_done_stages(loop_env):
    from catan_az.loop import run_iteration
    root, calls, sp, tr, win_arena, _ = loop_env
    cfg = AzConfig()

    def exploding_arena(cfg, cand, champ, iter_dir):
        calls.append("arena")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_iteration(cfg, root, 1, selfplay_fn=sp, train_fn=tr,
                      arena_fn=exploding_arena)
    assert calls == ["selfplay", "train", "arena"]
    # Rerun: selfplay+train have .done markers -> only arena re-executes.
    run_iteration(cfg, root, 1, selfplay_fn=sp, train_fn=tr,
                  arena_fn=win_arena)
    assert calls == ["selfplay", "train", "arena", "arena"]


def test_rerun_completed_iteration_is_noop(loop_env):
    """A fully-completed iteration re-run must NOT double-count Elo or append
    a duplicate journal row (2026-06-13: a restart double-published iter 1)."""
    from catan_az.ladder import Ladder
    from catan_az.loop import run_iteration
    root, _, sp, tr, win_arena, _ = loop_env
    cfg = AzConfig()

    run_iteration(cfg, root, 1, selfplay_fn=sp, train_fn=tr, arena_fn=win_arena)
    champ_after_1 = Ladder(root).champion()
    elo_1 = champ_after_1["elo"]
    games_1 = champ_after_1["games"]
    journal_1 = (root / "journal.csv").read_text()

    # Re-run the identical completed iteration: every stage skipped via
    # done-markers, publish guarded -> no state change at all.
    run_iteration(cfg, root, 1, selfplay_fn=sp, train_fn=tr, arena_fn=win_arena)
    champ_after_2 = Ladder(root).champion()
    assert champ_after_2["name"] == champ_after_1["name"]
    assert champ_after_2["elo"] == elo_1            # no double Elo
    assert champ_after_2["games"] == games_1        # no double game count
    assert (root / "journal.csv").read_text() == journal_1   # no dup row
    assert len(Ladder(root).history()) == 1         # one promote, not two


def test_stop_sentinel_halts_between_stages(loop_env):
    from catan_az.loop import run_iteration, StopRequested
    root, calls, sp, tr, win_arena, _ = loop_env

    def stopping_train(cfg, run_dirs, iter_dir, init_ckpt):
        calls.append("train")
        (root / "STOP").write_text("")
        return _mk_ckpt(iter_dir / "candidate.pt")

    with pytest.raises(StopRequested):
        run_iteration(AzConfig(), root, 1, selfplay_fn=sp,
                      train_fn=stopping_train, arena_fn=win_arena)
    assert calls == ["selfplay", "train"]   # arena never ran


def test_invalid_arena_neither_promotes_nor_marks_done(loop_env):
    from catan_az.ladder import Ladder
    from catan_az.loop import run_iteration
    root, _, sp, tr, _, _ = loop_env

    def censored_arena(cfg, cand, champ, iter_dir):
        return ArenaResult(wins_cand=80, wins_champ=30, draws=0, timeouts=10)

    verdict = run_iteration(AzConfig(), root, 1, selfplay_fn=sp, train_fn=tr,
                            arena_fn=censored_arena)
    assert verdict == "invalid"
    assert Ladder(root).champion()["name"] == "cell6"
    # invalid arenas must NOT leave a done marker — rerun must redo the arena
    assert not (root / "iter_1" / "ARENA.done").exists()
