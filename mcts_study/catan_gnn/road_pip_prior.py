"""Cand 11 (road-pip prior) of the loss-augmentation roadmap.

Per chat 2026-05-25:
  - Pure pip score (no port multiplier, no resource diversity, no resource check).
  - Per legal road action, score = pip(v_new) if v_new is settlement-legal else 0.
    v_new = the far endpoint of the road relative to the viewer's existing road network.
  - Gate A: fires only when NO settlement action is legal in the sample
            AND at least one legal road has nonzero score.
  - Layer 1: KL over an independent softmax of legal-road logits (no global mass change).
  - Default lambda_road = 0.05.

Math is documented in docs/superpowers/plans/2026-05-25-road-pip-prior.md
("Mathematical Specification" section).

Cited:
  - actions.rs:121 — road action_id = 108 + edge_id (range 108..179).
  - observation.rs:38-44 — vertex_features[v, 0] is the "empty" flag.
  - observation.rs:44 — edge_features[e, 2] is viewer's road (perspective-rotated;
                       perspective_idx 0 == viewer).
  - adjacency.EDGE_TO_VERTICES — 72 edges × 2 endpoint vertices.
  - settlement_vertex_prior.hex_features_to_pip / compute_vertex_score — reused.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .adjacency import EDGE_TO_VERTICES, NUM_EDGES, NUM_VERTICES
from .settlement_vertex_prior import compute_vertex_score, hex_features_to_pip


ROAD_ACTION_OFFSET = 108  # action_id = 108 + edge_id


def _build_vertex_neighbors() -> list[torch.Tensor]:
    """For each vertex v, list of vertex ids u such that some edge has
    endpoints {v, u}. Each entry is a 1-D long tensor, variable length
    (2 or 3 neighbors depending on board position).
    """
    nbrs: list[set[int]] = [set() for _ in range(NUM_VERTICES)]
    for e, vs in enumerate(EDGE_TO_VERTICES):
        a, b = int(vs[0]), int(vs[1])
        nbrs[a].add(b)
        nbrs[b].add(a)
    return [torch.tensor(sorted(s), dtype=torch.long) for s in nbrs]


VERTEX_NEIGHBORS: list[torch.Tensor] = _build_vertex_neighbors()


def _build_edge_to_vertices_tensor() -> torch.Tensor:
    """Shape [72, 2], dtype long. Mirrors adjacency.EDGE_TO_VERTICES as a tensor
    for vectorized lookup."""
    t = torch.zeros(NUM_EDGES, 2, dtype=torch.long)
    for e, vs in enumerate(EDGE_TO_VERTICES):
        t[e, 0] = int(vs[0])
        t[e, 1] = int(vs[1])
    return t


EDGE_TO_VERTICES_TENSOR: torch.Tensor = _build_edge_to_vertices_tensor()


def _viewer_frontier_vertices_from_edges(edge_features: torch.Tensor) -> set[int]:
    """Return the set of vertex ids that are endpoints of any edge owned
    by the viewer.

    Args:
        edge_features: shape [72, 6]. Col 2 = viewer's road flag.

    Returns:
        Python set of vertex ids (0..53).
    """
    viewer_owns = (edge_features[:, 2] >= 0.5)  # [72] bool
    front: set[int] = set()
    for e in range(72):
        if viewer_owns[e].item():
            v0, v1 = EDGE_TO_VERTICES[e]
            front.add(int(v0))
            front.add(int(v1))
    return front


def far_endpoint(*, edge_id: int, edge_features: torch.Tensor) -> int:
    """For road action targeting edge_id, return the vertex id that is NOT
    already in the viewer's road-network frontier. Returns -1 if both
    endpoints are in the frontier or neither is (no clear "new" vertex).

    Args:
        edge_id: int in 0..71.
        edge_features: shape [72, 6], single-sample tensor (no batch dim).

    Returns:
        int in 0..53, or -1 if no unambiguous far endpoint exists.
    """
    front = _viewer_frontier_vertices_from_edges(edge_features)
    v0, v1 = EDGE_TO_VERTICES[edge_id]
    v0_in = int(v0) in front
    v1_in = int(v1) in front
    if v0_in and not v1_in:
        return int(v1)
    if v1_in and not v0_in:
        return int(v0)
    return -1


def settlement_legal_mask(vertex_features: torch.Tensor) -> torch.Tensor:
    """Per-vertex boolean: True iff settlement-legal under the distance rule.

    A vertex is settlement-legal iff (a) the vertex itself is empty AND
    (b) every neighbor vertex (via VERTEX_NEIGHBORS) is empty.

    Args:
        vertex_features: shape [54, 13]. Col 0 = empty flag.

    Returns:
        Bool tensor of shape [54].
    """
    empty = (vertex_features[:, 0] >= 0.5)  # [54] bool
    out = torch.zeros(NUM_VERTICES, dtype=torch.bool, device=vertex_features.device)
    for v in range(NUM_VERTICES):
        if not empty[v].item():
            continue
        nbrs = VERTEX_NEIGHBORS[v].to(vertex_features.device)
        if bool(empty[nbrs].all().item()):
            out[v] = True
    return out


def compute_road_scores(
    *,
    edge_features: torch.Tensor,
    vertex_features: torch.Tensor,
    hex_features: torch.Tensor,
    legal_road_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the per-road pip score for a single sample.

    For each legal road action:
        score(r) = pip(v_new) * 1[v_new >= 0] * 1[settlement_legal(v_new)]

    Args:
        edge_features:   shape [72, 6]
        vertex_features: shape [54, 13]
        hex_features:    shape [19, 8]
        legal_road_mask: shape [72], bool. True if action 108+r is legal.

    Returns:
        Float tensor of shape [72]. Zero on illegal roads.
    """
    device = edge_features.device
    out = torch.zeros(NUM_EDGES, dtype=torch.float32, device=device)

    # Pip per vertex, from hex_features.
    hex_pip = hex_features_to_pip(hex_features)         # [19]
    vertex_pip = compute_vertex_score(hex_pip)          # [54]

    # Settlement-legal mask per vertex.
    settle_legal = settlement_legal_mask(vertex_features)  # [54] bool

    # Iterate over legal roads. The loop is over <= 72 elements per sample;
    # vectorizing across the batch happens in road_pip_prior_loss.
    for e in range(NUM_EDGES):
        if not bool(legal_road_mask[e].item()):
            continue
        v_new = far_endpoint(edge_id=e, edge_features=edge_features)
        if v_new < 0:
            continue
        if not bool(settle_legal[v_new].item()):
            continue
        out[e] = vertex_pip[v_new]
    return out


def build_road_pip_target(
    road_scores: torch.Tensor,
    legal_road_mask: torch.Tensor,
) -> torch.Tensor:
    """Linear normalization across legal roads.

    target[e] = road_scores[e] / sum(road_scores[legal_road_mask])
              = 0  on illegal roads or if all legal scores are zero.

    Args:
        road_scores: shape [72], float.
        legal_road_mask: shape [72], bool.

    Returns:
        Float tensor shape [72]. Sums to 1.0 if any score > 0; else all zero.
    """
    legal_f = legal_road_mask.to(road_scores.dtype)
    weighted = road_scores * legal_f
    total = weighted.sum()
    if float(total.item()) <= 0.0:
        return torch.zeros_like(road_scores)
    return weighted / total


def road_pip_prior_loss(**kwargs) -> torch.Tensor:
    raise NotImplementedError("Implemented in Task 4.")
