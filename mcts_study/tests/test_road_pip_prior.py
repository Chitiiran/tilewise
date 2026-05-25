"""Tests for Cand 11 (road-pip prior) — math + gate behavior."""
from __future__ import annotations

import pytest
import torch

from catan_gnn.road_pip_prior import (
    VERTEX_NEIGHBORS,
    EDGE_TO_VERTICES_TENSOR,
    far_endpoint,
    settlement_legal_mask,
    compute_road_scores,
    build_road_pip_target,
    road_pip_prior_loss,
)


def test_vertex_neighbors_via_edges_matches_adjacency():
    """VERTEX_NEIGHBORS[v] must equal the set of vertices reachable from v
    via one edge in EDGE_TO_VERTICES.

    Cited adjacency.py:46-119 — vertex 0 appears in edges 0 ([0,3]) and 1 ([0,4]),
    so neighbors(0) = {3, 4}.
    """
    assert set(VERTEX_NEIGHBORS[0].tolist()) == {3, 4}
    # Vertex 12 appears in edges 11 ([7,12]), 12 ([8,12]), 19 ([12,17]).
    assert set(VERTEX_NEIGHBORS[12].tolist()) == {7, 8, 17}
    # Every neighbor must be a valid vertex id.
    for v in range(54):
        for u in VERTEX_NEIGHBORS[v].tolist():
            assert 0 <= u < 54
            # Symmetric: u should also have v as a neighbor.
            assert v in VERTEX_NEIGHBORS[u].tolist(), f"asymmetric: {v} -> {u}"
