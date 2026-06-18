"""Phase 5 gate: Rust MCTS search reproduces async_mcts.py BIT-EXACT.

For each golden case (fixed seed, net, state, n_sims, eps): the Rust
debug_search must return identical visit_counts AND chosen action AND
root_value (spec §5 'Rust MCTS search' row). Covers eps=0 (arena/greedy) and
eps>0 (self-play Dirichlet) cases.

Run scripts/dump_mcts_golden.py first to (re)generate tests/data/mcts_golden.json.
"""
import json
from pathlib import Path

import numpy as np
import pytest

catan_mcts_rs = pytest.importorskip("catan_mcts_rs")

GOLDEN = Path(__file__).resolve().parent / "data" / "mcts_golden.json"


def _cases():
    if not GOLDEN.exists():
        pytest.skip("run scripts/dump_mcts_golden.py to generate the golden")
    return json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("case", _cases(), ids=lambda c: f"seed{c['seed']}_sims{c['n_sims']}_eps{c['eps']}")
def test_visit_counts_action_rootvalue_parity(case):
    entries = [(bool(ic), int(i)) for ic, i in case["entries"]]
    vc, action, root_value = catan_mcts_rs.debug_search(
        case["net_ts"], case["seed"], entries, case["n_sims"],
        case["rng_seed"], case["eps"], case["alpha"])
    assert list(vc) == case["visit_counts"], (
        f"visit_counts diverge (seed {case['seed']}, sims {case['n_sims']}, "
        f"eps {case['eps']})")
    assert action == case["best_action"], "chosen action diverges"
    # root_value bit-exact at full f64 (same accumulation order both sides).
    assert np.float64(root_value).tobytes() == np.float64(case["root_value"]).tobytes(), (
        f"root_value {root_value!r} != {case['root_value']!r}")
