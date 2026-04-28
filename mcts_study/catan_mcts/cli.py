"""CLI dispatcher: `python -m catan_mcts run <experiment> [args...]`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .experiments import (
    e1_winrate_vs_random,
    e2_ucb_c_sweep,
    e3_rollout_policy,
    e4_tournament,
)


_EXPERIMENTS = {
    "e1": e1_winrate_vs_random,
    "e2": e2_ucb_c_sweep,
    "e3": e3_rollout_policy,
    "e4": e4_tournament,
}


def _run_all(out_root: Path) -> None:
    for name, mod in _EXPERIMENTS.items():
        print(f"=== {name} ===", flush=True)
        mod.main(out_root=out_root)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="catan_mcts")
    sub = p.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run", help="run an experiment (e1, e2, e3, e4, or all)")
    runp.add_argument("name", choices=list(_EXPERIMENTS) + ["all"])
    runp.add_argument("--out-root", type=Path, default=Path("runs"))
    args, rest = p.parse_known_args(argv)

    if args.cmd != "run":
        p.error(f"unknown command {args.cmd}")
        return 2

    if args.name == "all":
        _run_all(args.out_root)
    else:
        mod = _EXPERIMENTS[args.name]
        # Forward unknown args to the experiment's CLI parser by pushing them onto sys.argv.
        sys.argv = [f"catan_mcts.{args.name}", *rest]
        mod.cli_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
