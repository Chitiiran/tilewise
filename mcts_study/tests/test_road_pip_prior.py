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


def _build_edge_features(viewer_road_edges: list[int]) -> torch.Tensor:
    """Helper: build a [72, 6] edge_features tensor where the listed edges
    are owned by the viewer (col 2 == 1), all others empty (col 0 == 1)."""
    ef = torch.zeros(72, 6)
    for e in range(72):
        if e in viewer_road_edges:
            ef[e, 1] = 1.0  # has road
            ef[e, 2] = 1.0  # viewer owns
        else:
            ef[e, 0] = 1.0  # empty
    return ef


def _build_vertex_features(occupied_vertices: list[int]) -> torch.Tensor:
    """Helper: [54, 13] vertex_features where listed vertices have a
    settlement and all others are empty. We set col 1 (settle) for
    occupied and col 0 (empty) for the rest."""
    vf = torch.zeros(54, 13)
    for v in range(54):
        if v in occupied_vertices:
            vf[v, 1] = 1.0  # settle
            # owner one-hot omitted — we don't read it
        else:
            vf[v, 0] = 1.0  # empty
    return vf


def test_far_endpoint_picks_new_vertex():
    """With a single viewer road at edge 0 ([0,3]):
      - Edge 6 [3,7]: v0=3 in frontier (yes — endpoint of edge 0),
                       v1=7 not in frontier → far = 7.
      - Edge 1 [0,4]: v0=0 in frontier, v1=4 not → far = 4.
    """
    ef = _build_edge_features(viewer_road_edges=[0])
    far_e6 = far_endpoint(edge_id=6, edge_features=ef)
    far_e1 = far_endpoint(edge_id=1, edge_features=ef)
    assert far_e6 == 7, f"expected 7, got {far_e6}"
    assert far_e1 == 4, f"expected 4, got {far_e1}"


def test_far_endpoint_returns_minus_one_when_both_in_frontier():
    """Viewer owns edges 0 ([0,3]) and 4 ([2,5]). Frontier = {0, 2, 3, 5}.
    Candidate edge 2 ([1,4]): v0=1 not in frontier, v1=4 not in frontier
    → both NOT in frontier → far = -1.
    """
    ef = _build_edge_features(viewer_road_edges=[0, 4])
    far_e2 = far_endpoint(edge_id=2, edge_features=ef)
    assert far_e2 == -1, (
        f"both endpoints (1, 4) not in viewer frontier {{0,2,3,5}}, "
        f"expected -1, got {far_e2}"
    )


def test_settlement_legal_mask_distance_rule():
    """A vertex is settlement-legal iff itself empty AND all its
    edge-neighbors empty.

    With vertex 0 occupied: vertex 0 not legal. Vertices 3 and 4 (neighbors of 0)
    not legal. Vertex 7 (neighbor of 3, but 3 is empty itself) IS legal.
    """
    vf = _build_vertex_features(occupied_vertices=[0])
    mask = settlement_legal_mask(vf)
    assert mask.dtype == torch.bool
    assert mask.shape == (54,)
    assert not mask[0].item(), "v0 occupied → not legal"
    assert not mask[3].item(), "v3 neighbor of occupied v0 → not legal"
    assert not mask[4].item(), "v4 neighbor of occupied v0 → not legal"
    assert mask[7].item(), "v7 neighbor of empty v3 (v3 itself empty) → legal"
