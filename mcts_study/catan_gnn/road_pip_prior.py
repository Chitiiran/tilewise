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

Performance note (2026-05-25 RCA):
  Original Python-loop implementations are kept as `*_loop` variants for
  equivalence testing. Production path uses fully batched tensor ops to
  avoid .item()/CUDA-sync overhead — see
  docs/superpowers/journals/2026-05-25-cand11-perf-rca.md.

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


# -----------------------------------------------------------------------------
# Module-level static topology tables (built once at import).
# -----------------------------------------------------------------------------

def _build_vertex_neighbors() -> list[torch.Tensor]:
    """For each vertex v, list of vertex ids u such that some edge has
    endpoints {v, u}. Each entry is a 1-D long tensor, variable length
    (2 or 3 neighbors depending on board position)."""
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


def _build_nbrs_padded() -> torch.Tensor:
    """Dense [54, max_nbrs] long tensor for vectorized distance-rule check.

    Vertices with fewer than `max_nbrs` real neighbors get the sentinel
    index NUM_VERTICES (= 54), which by convention is treated as "always
    empty" via a padded sentinel column in the empty-flag tensor.
    """
    max_nbrs = max(len(n) for n in VERTEX_NEIGHBORS)
    # Sentinel = NUM_VERTICES; the empty mask is padded with one extra
    # always-True column at that index.
    pad = torch.full((NUM_VERTICES, max_nbrs), NUM_VERTICES, dtype=torch.long)
    for v, nbrs in enumerate(VERTEX_NEIGHBORS):
        for i, u in enumerate(nbrs.tolist()):
            pad[v, i] = u
    return pad


NBRS_PADDED: torch.Tensor = _build_nbrs_padded()  # [54, 3]
NBRS_MAX: int = NBRS_PADDED.shape[1]


# -----------------------------------------------------------------------------
# LOOP IMPLEMENTATIONS (reference; used for equivalence testing).
# These match the original Python-loop versions verbatim. Production code
# should call the batched variants below.
# -----------------------------------------------------------------------------

def _viewer_frontier_vertices_from_edges_loop(edge_features: torch.Tensor) -> set[int]:
    """Original loop impl. Returns Python set of vertex ids in the viewer's
    road-network frontier. Use _frontier_batched for production."""
    viewer_owns = (edge_features[:, 2] >= 0.5)
    front: set[int] = set()
    for e in range(72):
        if viewer_owns[e].item():
            v0, v1 = EDGE_TO_VERTICES[e]
            front.add(int(v0))
            front.add(int(v1))
    return front


def far_endpoint_loop(*, edge_id: int, edge_features: torch.Tensor) -> int:
    """Original loop impl for a single edge query."""
    front = _viewer_frontier_vertices_from_edges_loop(edge_features)
    v0, v1 = EDGE_TO_VERTICES[edge_id]
    v0_in = int(v0) in front
    v1_in = int(v1) in front
    if v0_in and not v1_in:
        return int(v1)
    if v1_in and not v0_in:
        return int(v0)
    return -1


def settlement_legal_mask_loop(vertex_features: torch.Tensor) -> torch.Tensor:
    """Original loop impl. Use settlement_legal_mask_batched for production."""
    empty = (vertex_features[:, 0] >= 0.5)
    out = torch.zeros(NUM_VERTICES, dtype=torch.bool, device=vertex_features.device)
    for v in range(NUM_VERTICES):
        if not empty[v].item():
            continue
        nbrs = VERTEX_NEIGHBORS[v].to(vertex_features.device)
        if bool(empty[nbrs].all().item()):
            out[v] = True
    return out


def compute_road_scores_loop(
    *,
    edge_features: torch.Tensor,
    vertex_features: torch.Tensor,
    hex_features: torch.Tensor,
    legal_road_mask: torch.Tensor,
) -> torch.Tensor:
    """Original per-sample loop impl. Use compute_road_scores_batched for production."""
    device = edge_features.device
    out = torch.zeros(NUM_EDGES, dtype=torch.float32, device=device)
    hex_pip = hex_features_to_pip(hex_features)
    vertex_pip = compute_vertex_score(hex_pip)
    settle_legal = settlement_legal_mask_loop(vertex_features)
    for e in range(NUM_EDGES):
        if not bool(legal_road_mask[e].item()):
            continue
        v_new = far_endpoint_loop(edge_id=e, edge_features=edge_features)
        if v_new < 0:
            continue
        if not bool(settle_legal[v_new].item()):
            continue
        out[e] = vertex_pip[v_new]
    return out


# -----------------------------------------------------------------------------
# BATCHED IMPLEMENTATIONS (production path).
# All ops are pure tensor — no Python loops, no .item() in hot paths.
# -----------------------------------------------------------------------------

def settlement_legal_mask_batched(vertex_features: torch.Tensor) -> torch.Tensor:
    """Batched distance-rule check.

    Args:
        vertex_features: shape [..., 54, 13]. Last two dims fixed; any
            leading batch dims supported.

    Returns:
        Bool tensor of shape [..., 54].
    """
    # empty: shape [..., 54]
    empty = (vertex_features[..., 0] >= 0.5)
    # Pad an always-True sentinel column at index NUM_VERTICES so
    # variable-length neighbor lists can be packed into a dense gather.
    pad_shape = list(empty.shape[:-1]) + [1]
    sentinel = torch.ones(pad_shape, dtype=torch.bool, device=empty.device)
    empty_padded = torch.cat([empty, sentinel], dim=-1)  # [..., 55]

    nbrs = NBRS_PADDED.to(empty.device)                   # [54, 3]
    # Gather: nbr_empty[..., v, k] = empty_padded[..., nbrs[v, k]]
    # Broadcast nbrs over batch dims by indexing the last dim of empty_padded.
    nbr_empty = empty_padded[..., nbrs]                   # [..., 54, 3]
    all_nbrs_empty = nbr_empty.all(dim=-1)                # [..., 54]
    return empty & all_nbrs_empty


def far_endpoint_batched(edge_features: torch.Tensor) -> torch.Tensor:
    """Batched far-endpoint computation across all (sample, edge) pairs.

    Args:
        edge_features: shape [B, 72, 6].

    Returns:
        Long tensor of shape [B, 72]. Each entry is the vertex id that is
        NOT in the viewer's current frontier; -1 if both endpoints are
        in the frontier or neither is.
    """
    device = edge_features.device
    B = edge_features.shape[0]

    viewer_owns = (edge_features[..., 2] >= 0.5)          # [B, 72] bool
    ep = EDGE_TO_VERTICES_TENSOR.to(device)               # [72, 2]
    # Per-sample frontier mask: [B, 54] bool.
    # Build by scatter: for each (b, e) where viewer_owns[b, e], mark
    # ep[e, 0] and ep[e, 1] as True in frontier[b, :].
    frontier = torch.zeros(B, NUM_VERTICES, dtype=torch.bool, device=device)

    # Expand endpoints to [B, 72, 2] aligned with viewer_owns.
    ep_expanded = ep.unsqueeze(0).expand(B, -1, -1)       # [B, 72, 2]
    # Compute boolean "this endpoint is owned by viewer for this sample".
    owned_mask = viewer_owns.unsqueeze(-1).expand_as(ep_expanded)  # [B, 72, 2]

    # Flatten to [B*72*2] for scatter.
    flat_ep = ep_expanded.reshape(-1)                     # [B*144]
    flat_owned = owned_mask.reshape(-1)                   # [B*144]
    flat_b = torch.arange(B, device=device).repeat_interleave(72 * 2)  # [B*144]

    # Keep only owned endpoints.
    sel_b = flat_b[flat_owned]
    sel_v = flat_ep[flat_owned]
    if sel_v.numel() > 0:
        frontier[sel_b, sel_v] = True

    # Now for each candidate (sample, edge), check frontier membership of both endpoints.
    v0 = ep_expanded[..., 0]                              # [B, 72]
    v1 = ep_expanded[..., 1]                              # [B, 72]
    v0_in = frontier.gather(1, v0)                        # [B, 72]
    v1_in = frontier.gather(1, v1)                        # [B, 72]

    far = torch.full_like(v0, -1)
    far = torch.where(v0_in & ~v1_in, v1, far)
    far = torch.where(v1_in & ~v0_in, v0, far)
    return far                                            # [B, 72]


def compute_road_scores_batched(
    *,
    edge_features: torch.Tensor,
    vertex_features: torch.Tensor,
    hex_features: torch.Tensor,
    legal_road_mask: torch.Tensor,
) -> torch.Tensor:
    """Batched per-(sample, road) pip-score computation.

    Args:
        edge_features:   shape [B, 72, 6]
        vertex_features: shape [B, 54, 13]
        hex_features:    shape [B, 19, 8]
        legal_road_mask: shape [B, 72] bool

    Returns:
        Float tensor of shape [B, 72]. Zero where the road is illegal,
        has no clear far endpoint, or the far endpoint is not settlement-legal.
    """
    hex_pip = hex_features_to_pip(hex_features)                 # [B, 19]
    vertex_pip = compute_vertex_score(hex_pip)                  # [B, 54]

    settle_legal = settlement_legal_mask_batched(vertex_features)  # [B, 54] bool
    far = far_endpoint_batched(edge_features)                   # [B, 72] long, -1 if invalid

    valid_far = (far >= 0)                                      # [B, 72]
    far_safe = far.clamp(min=0)                                 # [B, 72] (safe index)
    gathered_pip = vertex_pip.gather(1, far_safe)               # [B, 72]
    gathered_legal = settle_legal.gather(1, far_safe)           # [B, 72]

    valid = valid_far & gathered_legal & legal_road_mask        # [B, 72]
    return torch.where(valid, gathered_pip, torch.zeros_like(gathered_pip))


# -----------------------------------------------------------------------------
# Public API — single-sample wrappers retained for backward compat with
# existing tests. Production batched path is used by road_pip_prior_loss.
# -----------------------------------------------------------------------------

def far_endpoint(*, edge_id: int, edge_features: torch.Tensor) -> int:
    """Single-edge convenience wrapper around far_endpoint_batched.

    Preserves the original int-returning API used by unit tests and
    compute_road_scores_loop.
    """
    # Add batch dim, run batched op, extract scalar.
    far = far_endpoint_batched(edge_features.unsqueeze(0))      # [1, 72]
    return int(far[0, edge_id].item())


def settlement_legal_mask(vertex_features: torch.Tensor) -> torch.Tensor:
    """Single-sample convenience wrapper. Returns [54] bool tensor."""
    # Add batch dim, run batched op, squeeze.
    return settlement_legal_mask_batched(vertex_features.unsqueeze(0))[0]


def compute_road_scores(
    *,
    edge_features: torch.Tensor,
    vertex_features: torch.Tensor,
    hex_features: torch.Tensor,
    legal_road_mask: torch.Tensor,
) -> torch.Tensor:
    """Single-sample convenience wrapper. Returns [72] float tensor."""
    return compute_road_scores_batched(
        edge_features=edge_features.unsqueeze(0),
        vertex_features=vertex_features.unsqueeze(0),
        hex_features=hex_features.unsqueeze(0),
        legal_road_mask=legal_road_mask.unsqueeze(0),
    )[0]


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


# -----------------------------------------------------------------------------
# Production loss — fully batched.
# -----------------------------------------------------------------------------

def road_pip_prior_loss(
    *,
    p_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    edge_features: torch.Tensor,
    vertex_features: torch.Tensor,
    hex_features: torch.Tensor,
) -> torch.Tensor:
    """Cand 11 auxiliary loss — batched implementation.

    Math: for each sample where gate A fires
        q[r]      = softmax(p_logits[L_R])[r]
        prior[r]  = score(r) / sum(score(legal_R))
        sample_L  = -sum(prior[r] * log q[r])  for r in L_R

    Returns mean over firing samples. If no sample fires the gate, returns
    a 0 tensor (no grad path).

    Args:
        p_logits:        shape [B, 280]
        legal_mask:      shape [B, 280], bool
        edge_features:   shape [B, 72, 6]
        vertex_features: shape [B, 54, 13]
        hex_features:    shape [B, 19, 8]

    Returns:
        Scalar tensor on the same device.
    """
    if legal_mask.dtype != torch.bool:
        legal_mask = legal_mask.bool()

    device = p_logits.device

    legal_settle_any = legal_mask[:, 0:54].any(dim=-1)
    legal_road_mask = legal_mask[:, ROAD_ACTION_OFFSET:ROAD_ACTION_OFFSET + NUM_EDGES]
    has_legal_road = legal_road_mask.any(dim=-1)

    candidates = (~legal_settle_any) & has_legal_road
    # Note: we deliberately do NOT short-circuit on `candidates.any().item()`
    # because that forces a CUDA sync. The batched scoring below is cheap;
    # the cost of running it on a fully-zero candidate batch is one extra
    # pass over zeros.

    # Fully batched scoring — one call, no Python loop over the batch.
    scores = compute_road_scores_batched(
        edge_features=edge_features,
        vertex_features=vertex_features,
        hex_features=hex_features,
        legal_road_mask=legal_road_mask,
    )                                                       # [B, 72]
    # Zero scores for non-candidate samples (they may still have legal roads
    # via has_legal_road but a settlement is legal so gate A doesn't fire).
    scores = scores * candidates.unsqueeze(-1).to(scores.dtype)

    has_score = (scores.sum(dim=-1) > 0)                    # [B]
    firing = candidates & has_score                         # [B]

    # Per-sample target via linear normalization.
    score_sums = scores.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    target = scores / score_sums                            # [B, 72]
    # Zero out non-firing samples (target row sums become 0 → contributes 0 to loss).
    firing_mask = firing.unsqueeze(-1).to(target.dtype)
    target = target * firing_mask

    # Layer 1: independent softmax over the road slice.
    road_logits = p_logits[:, ROAD_ACTION_OFFSET:ROAD_ACTION_OFFSET + NUM_EDGES]
    masked = road_logits.masked_fill(~legal_road_mask, float("-inf"))
    log_q = F.log_softmax(masked, dim=-1)
    log_q = log_q.masked_fill(~legal_road_mask, 0.0)

    # Per-sample CE: -sum(target * log_q)
    sample_loss = -(target * log_q).sum(dim=-1)             # [B]
    sample_loss = sample_loss * firing.to(sample_loss.dtype)

    # Mean over firing samples. Use .sum() and clamp; this is one fewer
    # .item()/sync than the previous impl's early-return paths.
    n_firing = firing.to(sample_loss.dtype).sum().clamp(min=1)
    return sample_loss.sum() / n_firing
