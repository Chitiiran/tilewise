"""Golden guard for the playback -> shared-module extraction.

Captures the per-state list and layout dict produced by the CURRENT
playback internals, so the Phase-1 refactor can be proven to preserve
them exactly. Run BEFORE extraction to write the golden; after extraction
the same assertions must still pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def minimal_run_dir(tmp_path_factory):
    from catan_mcts.experiments.e1_winrate_vs_random import main
    out_root = tmp_path_factory.mktemp("golden_runs")
    return main(
        out_root=out_root,
        num_games=1, sims_per_move_grid=[2],
        seed_base=4242, max_seconds=300.0,
    )


def test_replay_states_shape_is_stable(minimal_run_dir):
    """Every state dict carries the full field set the viewer renders."""
    from catan_mcts import playback
    seed = 4242 + 2 * 1_000
    history, winner, final_vp = playback._read_action_history(minimal_run_dir, seed)
    states = playback._replay_to_states(seed, history)
    assert len(states) >= 1
    required = {"n", "cp", "phase", "s", "c", "r", "rh", "vp", "hands",
                "bank", "dev_held", "ports", "lr_len", "knights", "built",
                "lr_holder", "la_holder", "vp_played"}
    for st in states:
        assert required.issubset(st.keys()), f"missing fields: {required - st.keys()}"
    json.dumps(states)
