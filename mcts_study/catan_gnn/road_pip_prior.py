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


# Forward declarations for Tasks 2-4. Tests import all symbols up front; the
# stubs let Task 1's test run in isolation. They are overwritten in later tasks.
def far_endpoint(*, edge_id: int, edge_features: torch.Tensor) -> int:  # noqa: D401
    raise NotImplementedError("Implemented in Task 2.")


def settlement_legal_mask(vertex_features: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Implemented in Task 2.")


def compute_road_scores(**kwargs) -> torch.Tensor:
    raise NotImplementedError("Implemented in Task 3.")


def build_road_pip_target(
    road_scores: torch.Tensor,
    legal_road_mask: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError("Implemented in Task 3.")


def road_pip_prior_loss(**kwargs) -> torch.Tensor:
    raise NotImplementedError("Implemented in Task 4.")
