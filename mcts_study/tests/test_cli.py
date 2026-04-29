import subprocess
import sys
from pathlib import Path


def test_cli_help_lists_experiments():
    r = subprocess.run(
        [sys.executable, "-m", "catan_mcts", "run", "--help"],
        capture_output=True, text=True, check=True,
    )
    out = r.stdout + r.stderr
    for name in ("e1", "e2", "e3", "e4", "all"):
        assert name in out


def test_cli_forwards_out_root_to_experiment(tmp_path: Path):
    """Regression: top-level `--out-root` was being consumed by cli.main but not
    forwarded to the experiment's cli_main, which then used its own default of
    'runs'. Result: every multi-worker sweep landed at worktree-root `runs/`
    instead of the user's specified path. This test runs a tiny e1 sweep with
    --out-root pointed into tmp_path and asserts the run dir lands there."""
    out_root = tmp_path / "specified_out_root"
    out_root.mkdir()
    r = subprocess.run(
        [sys.executable, "-m", "catan_mcts", "run", "e1",
         "--out-root", str(out_root),
         "--num-games", "1",
         "--sims-grid", "2",
         "--seed-base", "8888",
         "--max-seconds", "300"],
        capture_output=True, text=True, check=True,
    )
    # An e1 run dir should exist under our specified out_root, not at the cwd's `runs/`.
    candidates = list(out_root.glob("*-e1_*"))
    assert candidates, (
        f"expected e1 run dir under {out_root}; "
        f"stdout={r.stdout[-500:]} stderr={r.stderr[-500:]}"
    )
