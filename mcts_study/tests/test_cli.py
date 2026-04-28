import subprocess
import sys


def test_cli_help_lists_experiments():
    r = subprocess.run(
        [sys.executable, "-m", "catan_mcts", "run", "--help"],
        capture_output=True, text=True, check=True,
    )
    out = r.stdout + r.stderr
    for name in ("e1", "e2", "e3", "e4", "all"):
        assert name in out
