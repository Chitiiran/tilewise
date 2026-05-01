"""Tests for state_to_pyg: engine observation -> PyG HeteroData."""
import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from catan_bot import _engine
from catan_gnn.state_to_pyg import state_to_pyg


def _fresh_observation():
    e = _engine.Engine(42)
    return e.observation()


def test_returns_heterodata():
    obs = _fresh_observation()
    data = state_to_pyg(obs)
    assert isinstance(data, HeteroData)


def test_node_feature_shapes():
    obs = _fresh_observation()
    data = state_to_pyg(obs)
    assert data["hex"].x.shape == (19, 8)
    assert data["vertex"].x.shape == (54, 7)
    assert data["edge"].x.shape == (72, 6)
    # Scalars stored as graph-level attribute with leading batch dim [1, N_SCALARS]
    # so PyG Batch collation yields [B, N_SCALARS] across PyG versions.
    from catan_gnn.gnn_model import N_SCALARS
    assert data.scalars.shape == (1, N_SCALARS)


def test_edge_indices_present():
    obs = _fresh_observation()
    data = state_to_pyg(obs)
    # PyG conventions for heterogeneous edges:
    #   data[("hex", "to", "vertex")].edge_index
    #   data[("vertex", "to", "hex")].edge_index
    #   data[("vertex", "to", "edge")].edge_index
    #   data[("edge", "to", "vertex")].edge_index
    h2v = data["hex", "to", "vertex"].edge_index
    v2h = data["vertex", "to", "hex"].edge_index
    v2e = data["vertex", "to", "edge"].edge_index
    e2v = data["edge", "to", "vertex"].edge_index
    assert h2v.shape == (2, 114)
    assert v2h.shape == (2, 114)
    assert v2e.shape == (2, 144)
    assert e2v.shape == (2, 144)


def test_dtypes():
    obs = _fresh_observation()
    data = state_to_pyg(obs)
    assert data["hex"].x.dtype == torch.float32
    assert data["vertex"].x.dtype == torch.float32
    assert data["edge"].x.dtype == torch.float32
    assert data.scalars.dtype == torch.float32
    assert data["hex", "to", "vertex"].edge_index.dtype == torch.long


def test_does_not_mutate_obs_dict():
    obs = _fresh_observation()
    obs_copy = {k: v.copy() if hasattr(v, "copy") else v for k, v in obs.items()}
    state_to_pyg(obs)
    for k in obs_copy:
        if hasattr(obs[k], "shape"):
            np.testing.assert_array_equal(obs[k], obs_copy[k])


def test_legal_mask_passed_through():
    """legal_mask is not part of the GNN graph -- but state_to_pyg may carry
    it through as an aux field for downstream policy masking. Spec section 3
    says legal_mask is applied to logits externally, but the dataset/evaluator
    needs it. Store it as `data.legal_mask` (graph-level)."""
    from catan_mcts import ACTION_SPACE_SIZE
    obs = _fresh_observation()
    data = state_to_pyg(obs)
    assert data.legal_mask.shape == (ACTION_SPACE_SIZE,)
    assert data.legal_mask.dtype == torch.bool
