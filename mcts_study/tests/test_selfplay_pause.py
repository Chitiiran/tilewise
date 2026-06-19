"""Pausability of self-play: chunked processing, PAUSE sentinel stops between
chunks, resume skips done seeds. Mocks the Rust engine so it's fast + GPU-free.
"""
import sys
import types
from pathlib import Path

import pytest

from catan_mcts.experiments import self_play_rust as spr


def _fake_run_selfplay(ts, seeds, *a, **k):
    return [{
        "seed": int(s), "terminal": True, "winner": 0,
        "final_vp": [10, 0, 0, 0], "length_in_moves": 5,
        "action_history": [1, 2, 3],
        "moves": [{"current_player": 0, "move_index": 0,
                   "legal_mask": [True] + [False] * 279,
                   "visit_counts": [1] + [0] * 279,
                   "action_taken": 0, "root_value": 0.5}],
    } for s in seeds]


@pytest.fixture(autouse=True)
def _stub_engine(monkeypatch):
    """Hermetic stub: bind the lazy engine handle to a fake + dummy .ts export.
    No sys.modules pollution (engine is lazily imported via _engine())."""
    fake = types.ModuleType("catan_mcts_rs")
    fake.run_selfplay = _fake_run_selfplay
    monkeypatch.setattr(spr, "catan_mcts_rs", fake, raising=False)
    monkeypatch.setattr(spr, "export",
                        lambda **k: Path(k["out_ts"]).write_text("ts") or Path(k["out_ts"]))
    monkeypatch.setattr(spr, "export_batched",
                        lambda **k: Path(k["out_ts"]).write_text("b") or Path(k["out_ts"]))
    yield


def _run(out, n_games, n_concurrent, pause_dir=None, resume_dir=None):
    return spr.run_self_play_rust(
        out_root=out, checkpoint=out / "ck.pt", num_games=n_games, n_sims=4,
        hidden_dim=32, num_layers=2, vp_target=10, bonuses=True,
        seed_base=1000, self_play=True, max_steps=100,
        game_deadline_seconds=None, resume_dir=resume_dir,
        n_concurrent=n_concurrent, pause_dir=pause_dir)


def test_chunked_completes_all(tmp_path):
    out = _run(tmp_path, n_games=10, n_concurrent=4)
    done = spr.SelfPlayRecorder(out, config={}).done_seeds()
    assert done == {1000 + i for i in range(10)}


def test_pause_stops_between_chunks_and_resume_finishes(tmp_path):
    (tmp_path / "ck.pt").write_text("x")
    out = spr.make_run_dir(tmp_path, "self_play_rust")
    # PAUSE present from the start -> stops before any chunk runs.
    (out / "PAUSE").write_text("")
    _run(tmp_path, n_games=10, n_concurrent=4, pause_dir=out, resume_dir=out)
    done0 = spr.SelfPlayRecorder(out, config={}).done_seeds()
    assert len(done0) == 0, "PAUSE should stop before any games"

    # Remove PAUSE, resume -> all 10 finish, no duplicates.
    (out / "PAUSE").unlink()
    _run(tmp_path, n_games=10, n_concurrent=4, pause_dir=out, resume_dir=out)
    done1 = spr.SelfPlayRecorder(out, config={}).done_seeds()
    assert done1 == {1000 + i for i in range(10)}


def test_paused_sentinel_detection(tmp_path):
    assert not spr._paused(tmp_path)
    (tmp_path / "PAUSE").write_text("")
    assert spr._paused(tmp_path)
