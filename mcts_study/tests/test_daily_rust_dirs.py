"""Regression tests for the code-review findings (2026-06-19):
C1 — _all_selfplay_dirs must find self_play_rust dirs (not just async).
C2 — the rust self-play launch uses a deterministic resume dir + no seed offset.
"""
from pathlib import Path

from catan_az import daily
from catan_az.config import AzConfig


def test_all_selfplay_dirs_finds_rust_dirs(tmp_path):
    """C1: prior rust self-play dirs must be visible to the window/resume logic.
    The old *self_play_async* glob hid them -> window collapsed to one iter."""
    # two iters, each with a rust self-play dir.
    for it in (1, 2):
        d = tmp_path / f"iter_{it}" / "selfplay" / "self_play_rust"
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
    # an async dir too (mixed history must also work).
    da = tmp_path / "iter_3" / "selfplay" / "2026-self_play_async-pX"
    da.mkdir(parents=True)

    found = daily._all_selfplay_dirs(tmp_path)
    names = {p.name for p in found}
    assert "self_play_rust" in names, "C1: rust dirs must be found"
    assert any("self_play_async" in n for n in names), "async dirs still found"
    assert len(found) == 3


def test_rust_launch_uses_deterministic_resume_dir(monkeypatch):
    """C2: rust path passes --resume-dir (fixed dir) and NO seed offset, so a
    resume reuses the dir + done.txt skip-list (single dedup mechanism)."""
    captured = {}

    class _FakePopen:
        def __init__(self, cmd, **kw):
            captured["cmd"] = [str(c) for c in cmd]

        def wait(self, timeout=None):
            return 0

        @property
        def returncode(self):
            return 0

    monkeypatch.setattr(daily.subprocess, "Popen", _FakePopen)
    cfg = AzConfig(engine="rust", worker_nice=10, sims=8, n_concurrent=256,
                   max_batch=32, hidden_dim=32, num_layers=2, vp_target=10)
    try:
        daily._launch_selfplay_procs(
            cfg, out_dir=Path("/tmp/c2_sp"), checkpoint=Path("x.pt"),
            n_games=50, n_procs=7, generator_name="g", gen_iter=2,
            rules_id="v3-full", seed_offset=999)
    except Exception:
        pass
    cmd = captured["cmd"]
    assert "--resume-dir" in cmd, "rust must pass --resume-dir for dedup-by-done"
    # deterministic dir name (contains self_play_rust so globs find it)
    rd = cmd[cmd.index("--resume-dir") + 1]
    assert rd.endswith("self_play_rust"), rd
    # seed-base must NOT include the 999 offset for rust (offset is async-only)
    sb = cmd[cmd.index("--seed-base") + 1]
    expected = daily._worker_seed_base(gen_iter=2, i=0)  # i=0, no offset
    assert int(sb) == expected, f"rust seed-base must omit offset: {sb} != {expected}"
