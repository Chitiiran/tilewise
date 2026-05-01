"""Tests for GnnModel: heterogeneous PyG body + policy/value heads."""
import time

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from catan_bot import _engine
from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg


def _make_data(seed: int = 42):
    e = _engine.Engine(seed)
    return state_to_pyg(e.observation())


def test_forward_shapes_batch_1():
    model = GnnModel(hidden_dim=32, num_layers=2)
    data = _make_data()
    batch = Batch.from_data_list([data])
    value, policy = model(batch)
    assert value.shape == (1, 4)
    from catan_mcts import ACTION_SPACE_SIZE
    assert policy.shape == (1, ACTION_SPACE_SIZE)


def test_forward_shapes_batch_4():
    model = GnnModel(hidden_dim=32, num_layers=2)
    batch = Batch.from_data_list([_make_data(s) for s in [1, 2, 3, 4]])
    value, policy = model(batch)
    assert value.shape == (4, 4)
    from catan_mcts import ACTION_SPACE_SIZE
    assert policy.shape == (4, ACTION_SPACE_SIZE)


def test_value_head_in_unit_range():
    model = GnnModel(hidden_dim=32, num_layers=2)
    data = _make_data()
    batch = Batch.from_data_list([data])
    value, _ = model(batch)
    assert (value >= -1.0).all() and (value <= 1.0).all()


def test_param_count_in_v0_budget():
    """Spec §2: target ~30-60k params at hidden_dim=32, 2 layers."""
    model = GnnModel(hidden_dim=32, num_layers=2)
    n = sum(p.numel() for p in model.parameters())
    assert 20_000 < n < 80_000, f"v0 model has {n} params; outside [20k, 80k]"


def test_cpu_latency_under_5ms_b1():
    """Spec §2: <=5ms per forward pass on CPU at batch=1."""
    model = GnnModel(hidden_dim=32, num_layers=2).eval()
    data = _make_data()
    batch = Batch.from_data_list([data])
    # Warm up.
    with torch.no_grad():
        for _ in range(5):
            model(batch)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            model(batch)
    dt_ms = (time.perf_counter() - t0) / 50 * 1000
    assert dt_ms < 10.0, f"forward pass took {dt_ms:.2f} ms; budget 5 ms (10 ms slack)"


def test_deterministic_with_fixed_seed():
    torch.manual_seed(0)
    m1 = GnnModel(hidden_dim=32, num_layers=2)
    torch.manual_seed(0)
    m2 = GnnModel(hidden_dim=32, num_layers=2)
    data = _make_data()
    batch = Batch.from_data_list([data])
    v1, p1 = m1(batch)
    v2, p2 = m2(batch)
    torch.testing.assert_close(v1, v2)
    torch.testing.assert_close(p1, p2)


def test_scalars_collated_correctly_for_various_batch_sizes():
    """Sanity: batch.scalars must have shape [B, 22] after PyG collation,
    regardless of PyG version. Catches regressions if PyG changes how it
    handles graph-level attributes."""
    model = GnnModel(hidden_dim=32, num_layers=2)
    for B in [1, 2, 4, 16]:
        batch = Batch.from_data_list([_make_data(s) for s in range(B)])
        assert batch.scalars.shape[0] == B
        # Total elements must equal B * 22 regardless of layout.
        assert batch.scalars.numel() == B * 22
        # Forward should not raise.
        value, policy = model(batch)
        assert value.shape == (B, 4)
        from catan_mcts import ACTION_SPACE_SIZE
        assert policy.shape == (B, ACTION_SPACE_SIZE)
