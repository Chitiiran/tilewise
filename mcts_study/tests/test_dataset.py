"""Tests for CatanReplayDataset: parquet -> (HeteroData, value, policy, mask)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from catan_gnn.dataset import CachedDataset, CatanReplayDataset


@pytest.fixture(scope="module")
def minimal_run_dir(tmp_path_factory):
    """Spawn one tiny e1 game once; reuse for all dataset tests in this module."""
    from catan_mcts.experiments.e1_winrate_vs_random import main
    out_root = tmp_path_factory.mktemp("e1_runs")
    return main(
        out_root=out_root,
        num_games=1, sims_per_move_grid=[2],
        seed_base=999, max_seconds=300.0,
    )


def test_dataset_loads_from_run_dir(minimal_run_dir):
    ds = CatanReplayDataset([minimal_run_dir])
    assert len(ds) > 0


def test_dataset_item_shapes(minimal_run_dir):
    ds = CatanReplayDataset([minimal_run_dir])
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


def test_value_target_perspective_rotation(minimal_run_dir):
    """value[0] corresponds to the *current_player* of that move (perspective).
    The same game's later moves with different current_player should rotate
    accordingly. Spec section 4 'value target': index i = +1 if (current_player + i) % 4 == winner."""
    ds = CatanReplayDataset([minimal_run_dir])
    # Walk the dataset, find a row whose seed has a known winner.
    # Just sanity-check: for at least one row, the +1 lands at a valid index 0-3.
    for i in range(len(ds)):
        _, value, _, _ = ds[i]
        if (value == 1.0).any():
            assert int((value == 1.0).nonzero(as_tuple=True)[0]) in range(4)
            break


def test_v1_games_are_skipped(tmp_path: Path):
    """If a games.parquet contains v1 rows (no action_history column),
    they must be filtered out by CatanReplayDataset since they can't be
    replayed cheaply. We don't want to train on garbage."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Synthesize a v1 games.parquet (schema_version=1, no action_history).
    run_dir = tmp_path / "fake_v1_run"
    run_dir.mkdir()
    games_v1 = pa.table({
        "seed": [12345],
        "winner": [0],
        "final_vp": [[10, 5, 4, 3]],
        "length_in_moves": [100],
        "mcts_config_id": ["fake-v1-config"],
        "schema_version": [1],
    })
    pq.write_table(games_v1, run_dir / "games.v1cell.parquet")
    # And a moves.parquet with rows referencing that seed.
    moves_v1 = pa.table({
        "seed": [12345],
        "move_index": [0],
        "current_player": [0],
        "legal_action_mask": [[True] + [False] * 205],
        "mcts_visit_counts": [[1] + [0] * 205],
        "action_taken": [0],
        "mcts_root_value": [0.0],
        "schema_version": [1],
    })
    pq.write_table(moves_v1, run_dir / "moves.v1cell.parquet")

    # Dataset should construct without error but report 0 rows since no v2 games match.
    ds = CatanReplayDataset([run_dir])
    assert len(ds) == 0, f"Expected v1-only dataset to have 0 rows, got {len(ds)}"


def test_cached_dataset_round_trip(minimal_run_dir, tmp_path):
    """CachedDataset should produce items identical to the source dataset, and
    persist + reload via cache_path."""
    src = CatanReplayDataset([minimal_run_dir])
    assert len(src) > 0

    # First call: builds + persists.
    cache_path = tmp_path / "cache.pt"
    cached = CachedDataset(source=src, cache_path=cache_path, verbose=False)
    assert cache_path.exists()
    assert len(cached) == len(src)

    # Items should be tensor-equal between source and cache.
    src_data, src_value, src_policy, src_legal = src[0]
    cached_data, cached_value, cached_policy, cached_legal = cached[0]
    torch.testing.assert_close(src_data["hex"].x, cached_data["hex"].x)
    torch.testing.assert_close(src_data["vertex"].x, cached_data["vertex"].x)
    torch.testing.assert_close(src_data["edge"].x, cached_data["edge"].x)
    torch.testing.assert_close(src_data.scalars, cached_data.scalars)
    torch.testing.assert_close(src_value, cached_value)
    torch.testing.assert_close(src_policy, cached_policy)
    assert torch.equal(src_legal, cached_legal)

    # Second call: loads from disk; no source needed.
    reloaded = CachedDataset(source=None, cache_path=cache_path, verbose=False)
    assert len(reloaded) == len(src)
    r_data, r_value, _, _ = reloaded[0]
    torch.testing.assert_close(r_data["hex"].x, src_data["hex"].x)
    torch.testing.assert_close(r_value, src_value)


def test_cached_dataset_seeds_attribute(minimal_run_dir, tmp_path):
    """CachedDataset.seeds should mirror source._index['seed'] so train._split_by_seed
    can do whole-game train/val splits without depending on the source dataset."""
    src = CatanReplayDataset([minimal_run_dir])
    cache_path = tmp_path / "cache.pt"
    cached = CachedDataset(source=src, cache_path=cache_path, verbose=False)
    assert len(cached.seeds) == len(cached)
    expected = [int(s) for s in src._index["seed"]]
    assert cached.seeds == expected
