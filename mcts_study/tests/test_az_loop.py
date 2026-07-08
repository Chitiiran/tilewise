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


def test_existing_window_is_honored_not_recapped(loop_env, tmp_path):
    """When run_iteration is given existing_selfplay_dirs (the accumulating
    window gen_window built in run_cycle), it must TRAIN ON EXACTLY THOSE dirs —
    not re-derive its own list and re-truncate to cfg.window_games.

    2026-06-17 bug: loop.py rebuilt all_dirs + select_window(cfg.window_games),
    silently capping the 2786-game accumulating window back to 1200. The design
    (docs 2026-06-14/15) is an accumulating window that grows until promotion."""
    import pandas as pd
    from catan_az.loop import run_iteration
    root, _, _, _, win_arena, _ = loop_env

    # Build N accumulating-window dirs, each with games, N chosen so total games
    # exceeds cfg.window_games (1200) -> a re-cap WOULD drop some.
    win_dirs = []
    for i in range(20):
        d = tmp_path / f"acc_{i}"
        d.mkdir()
        pd.DataFrame({"seed": range(i * 100, i * 100 + 100),
                      "winner": [0, 1, 2, 3] * 25}).to_parquet(d / "games.x.parquet")
        win_dirs.append(d)
    # 20 dirs x 100 games = 2000 games > window_games(1200): a re-cap truncates.

    captured = {}

    def capturing_train(cfg, run_dirs, iter_dir, init_ckpt):
        captured["window"] = [Path(p) for p in run_dirs]
        return _mk_ckpt(iter_dir / "candidate.pt")

    run_iteration(AzConfig(), root, 1, selfplay_fn=None,
                  train_fn=capturing_train, arena_fn=win_arena,
                  existing_selfplay_dirs=win_dirs)

    trained = {p.resolve() for p in captured["window"]}
    expected = {p.resolve() for p in win_dirs}
    assert trained == expected, (
        f"training window must equal the passed accumulating window "
        f"({len(expected)} dirs), got {len(trained)} — it was re-capped")


def test_resume_window_follows_accumulating_dirs_not_selfplay_marker(loop_env, tmp_path):
    """On RESUME, SELFPLAY.done lists only THIS iteration's self-play dirs (so
    self-play isn't regenerated). The training WINDOW must still follow the
    accumulating existing_selfplay_dirs run_cycle passes — NOT the marker's
    smaller list. 2026-06-17 resume-window bug: run_dirs (from the marker)
    shadowed the accumulating window, training on ~945 games not the full set."""
    import json
    import pandas as pd
    from catan_az.loop import run_iteration
    root, _, _, _, win_arena, _ = loop_env

    iter_dir = root / "iter_1"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Pre-seed SELFPLAY.done with only 2 "own-iter" dirs (simulating a resume).
    own = []
    for i in range(2):
        d = tmp_path / f"own_{i}"; d.mkdir()
        pd.DataFrame({"seed": range(i*10, i*10+10), "winner": [0,1,2,3]*2+[0,1]
                      }).to_parquet(d / "games.x.parquet")
        own.append(d)
    (iter_dir / "SELFPLAY.done").write_text(
        json.dumps({"run_dirs": [str(p) for p in own], "source": "existing"}))

    # The accumulating window run_cycle passes is LARGER (10 dirs).
    acc = []
    for i in range(10):
        d = tmp_path / f"acc_{i}"; d.mkdir()
        pd.DataFrame({"seed": range(1000+i*10, 1000+i*10+10), "winner": [0,1,2,3]*2+[0,1]
                      }).to_parquet(d / "games.x.parquet")
        acc.append(d)

    captured = {}
    def capturing_train(cfg, run_dirs, iter_dir_, init_ckpt):
        captured["window"] = [Path(p) for p in run_dirs]
        return _mk_ckpt(iter_dir_ / "candidate.pt")

    run_iteration(AzConfig(), root, 1, selfplay_fn=None,
                  train_fn=capturing_train, arena_fn=win_arena,
                  existing_selfplay_dirs=acc)

    trained = {p.resolve() for p in captured["window"]}
    assert trained == {p.resolve() for p in acc}, (
        "resume must window over the accumulating dirs, not the SELFPLAY.done list")


def test_selfplay_writes_data_quality_json(loop_env):
    """Task 7: after fresh self-play, the loop must write data_quality.json
    next to SELFPLAY.done summarizing the run (shake-out journal §3 — data
    problems were invisible until training wasted hours on them)."""
    from catan_az.loop import run_iteration
    root, _, sp, tr, win_arena, _ = loop_env
    run_iteration(AzConfig(), root, 1, selfplay_fn=sp, train_fn=tr,
                  arena_fn=win_arena)
    dq_path = root / "iter_1" / "data_quality.json"
    assert dq_path.exists()
    dq = json.loads(dq_path.read_text())
    assert dq["games"] == 5
    assert dq["verdict"] == "ok"   # fake_selfplay's 5 decisive games, no timeouts


def test_selfplay_refuses_to_mark_healthy_when_degenerate(loop_env, tmp_path):
    """A self-play stage that produces an all-timeout run must NOT proceed to
    train — the loop refuses (raises), mirroring the M4 own-iter-floor
    RuntimeError pattern in daily.py's generate_iter_games."""
    import pandas as pd
    import pytest as _pytest
    from catan_az.loop import run_iteration
    root, calls, _, tr, win_arena, _ = loop_env

    def degenerate_selfplay(cfg, iter_dir, champion_ckpt):
        calls.append("selfplay")
        d = iter_dir / "selfplay_run"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "seed": range(10),
            "winner": [-1] * 10,
            "length_in_moves": [999_999] * 10,
            "timed_out": [True] * 10,
        }).to_parquet(d / "games.fake.parquet")
        return [d]

    with _pytest.raises(RuntimeError, match="degenerate"):
        run_iteration(AzConfig(), root, 1, selfplay_fn=degenerate_selfplay,
                      train_fn=tr, arena_fn=win_arena)
    # train/arena must never have run
    assert calls == ["selfplay"]
    # data_quality.json is still written (diagnostic even on refusal) but
    # SELFPLAY.done must NOT exist -> a rerun retries self-play, not skips it
    dq_path = root / "iter_1" / "data_quality.json"
    assert dq_path.exists()
    assert json.loads(dq_path.read_text())["verdict"] == "degenerate"
    assert not (root / "iter_1" / "SELFPLAY.done").exists()


def test_existing_selfplay_dirs_skip_degeneracy_gate(loop_env, tmp_path):
    """The existing_selfplay_dirs salvage path (explicit human-authorized
    reuse of pre-existing data, e.g. iteration-1 salvage) is NOT re-gated —
    the gate is for the loop's OWN fresh self-play stage."""
    import pandas as pd
    from catan_az.loop import run_iteration
    root, _, _, tr, win_arena, _ = loop_env
    d = tmp_path / "salvage"
    d.mkdir()
    pd.DataFrame({
        "seed": range(5), "winner": [-1] * 5,
        "length_in_moves": [999_999] * 5, "timed_out": [True] * 5,
    }).to_parquet(d / "games.fake.parquet")
    # Must NOT raise even though this data is degenerate by the gate's rules.
    verdict = run_iteration(AzConfig(), root, 1, selfplay_fn=None, train_fn=tr,
                            arena_fn=win_arena, existing_selfplay_dirs=[d])
    assert verdict == "promote"


def test_invalid_arena_neither_promotes_nor_marks_done(loop_env):
    from catan_az.ladder import Ladder
    from catan_az.loop import run_iteration
    root, _, sp, tr, _, _ = loop_env

    def censored_arena(cfg, cand, champ, iter_dir):
        # Too many VP ties (no-signal) -> invalid under the new draw-rate guard.
        return ArenaResult(wins_cand=40, wins_champ=20, draws=60, timeouts=120)

    verdict = run_iteration(AzConfig(), root, 1, selfplay_fn=sp, train_fn=tr,
                            arena_fn=censored_arena)
    assert verdict == "invalid"
    assert Ladder(root).champion()["name"] == "cell6"
    # invalid arenas must NOT leave a done marker — rerun must redo the arena
    assert not (root / "iter_1" / "ARENA.done").exists()
