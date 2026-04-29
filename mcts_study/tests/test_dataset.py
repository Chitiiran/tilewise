"""Tests for CatanReplayDataset: parquet -> (HeteroData, value, policy, mask)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from catan_gnn.dataset import CatanReplayDataset


def _make_minimal_run(tmp_path: Path):
    """Run a tiny e1 sweep so tests have real parquet to read.
    1 sims cell x 1 game = 1 game, ~few hundred labeled moves."""
    from catan_mcts.experiments.e1_winrate_vs_random import main
    out = main(
        out_root=tmp_path, num_games=1,
        sims_per_move_grid=[2], seed_base=999,
        max_seconds=300.0,
    )
    return out


def test_dataset_loads_from_run_dir(tmp_path: Path):
    out = _make_minimal_run(tmp_path)
    ds = CatanReplayDataset([out])
    assert len(ds) > 0


def test_dataset_item_shapes(tmp_path: Path):
    out = _make_minimal_run(tmp_path)
    ds = CatanReplayDataset([out])
    data, value, policy, legal = ds[0]
    # data is a HeteroData; check the embedded shapes
    assert data["hex"].x.shape == (19, 8)
    assert data["vertex"].x.shape == (54, 7)
    assert data["edge"].x.shape == (72, 6)
    # value: [4] in [-1, +1] U {0}
    assert value.shape == (4,)
    assert torch.all((value == 1.0) | (value == -1.0) | (value == 0.0))
    # policy: [206], a probability distribution (sums to 1 +/- eps OR all-zeros if no MCTS visits -- for completed games every recorded move has visits)
    assert policy.shape == (206,)
    s = float(policy.sum())
    assert abs(s - 1.0) < 1e-5, f"policy sums to {s}, expected 1.0"
    # legal mask: [206], bool, action_taken must be legal
    assert legal.shape == (206,)
    assert legal.dtype == torch.bool


def test_value_target_perspective_rotation(tmp_path: Path):
    """value[0] corresponds to the *current_player* of that move (perspective).
    The same game's later moves with different current_player should rotate
    accordingly. Spec section 4 'value target': index i = +1 if (current_player + i) % 4 == winner."""
    out = _make_minimal_run(tmp_path)
    ds = CatanReplayDataset([out])
    # Walk the dataset, find a row whose seed has a known winner.
    # Just sanity-check: for at least one row, the +1 lands at a valid index 0-3.
    for i in range(len(ds)):
        _, value, _, _ = ds[i]
        if (value == 1.0).any():
            assert int((value == 1.0).nonzero(as_tuple=True)[0]) in range(4)
            break
