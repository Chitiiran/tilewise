"""Cand 1 (settlement-vertex pip-weighted prior) of the loss-augmentation
roadmap.

Per chat 2026-05-12:
  - Pure pip score (no resource VP weighting). vertex_score[v] = sum of
    pip[h] for h adjacent to v. Robber ignored (Option A).
  - Fires whenever any BuildSettlement action (action_id 0..53) is legal.
    No phase gating.
  - Default lambda_settle = 0.20 (Cell 2 first run).

Math:
  For each training sample where settlement is legal:
    legal_settle = legal_mask[..., 0:54]
    settle_logits = p_logits[..., 0:54]
    settle_target = vertex_score normalized over legal vertices
    (uniform fallback when all legal scores are zero)

    log_pred = log_softmax(settle_logits over legal vertices)
    sample_loss = -(settle_target * log_pred).sum()

Cited:
  - actions.rs:121 — BuildSettlement action_id = vertex_id (0..53)
  - observation.rs:75-86 — hex_features layout
        [h, 0..4] resource one-hot (wood/brick/sheep/wheat/ore)
        [h, 5]    (dice_num - 7) / 5  normalized
        [h, 6]    robber flag
        [h, 7]    desert flag
  - adjacency.HEX_TO_VERTICES — list of 6 vertex IDs per hex (19 x 6)
  - Standard Catan pip table
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .adjacency import HEX_TO_VERTICES


# Standard Catan pip values. Index = dice number 0..12.
# dice=0 (desert / no number) and dice=7 (never on a tile) -> 0 pip.
PIP_BY_DICE: dict[int, int] = {
    0: 0,
    2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
    7: 0,
    8: 5, 9: 4, 10: 3, 11: 2, 12: 1,
}

# Precomputed tensor for vectorized lookup.
_PIP_TABLE = torch.tensor(
    [PIP_BY_DICE.get(i, 0) for i in range(13)],
    dtype=torch.float32,
)


def _build_hex_to_vertex_matrix() -> torch.Tensor:
    """Sparse-ish 19x54 0/1 matrix M where M[h, v] = 1 iff hex h is
    adjacent to vertex v. Used to scatter pip from hex space to vertex
    space via a simple matrix multiply.

    vertex_score = hex_pip @ M
    """
    M = torch.zeros(19, 54, dtype=torch.float32)
    for h, vs in enumerate(HEX_TO_VERTICES):
        for v in vs:
            M[h, int(v)] = 1.0
    return M


_HEX_TO_VERTEX_MATRIX = _build_hex_to_vertex_matrix()


def hex_features_to_pip(hex_features: torch.Tensor) -> torch.Tensor:
    """Extract per-hex pip values from hex_features tensor.

    Args:
        hex_features: shape [..., 19, 8]. Per cited observation.rs:
            [h, 5] = (dice_num - 7) / 5, [h, 7] = desert flag.

    Returns:
        Per-hex pip, shape [..., 19], float32. Desert hexes -> 0.
    """
    # Recover dice number: dice_num = round(hex_features[..., 5] * 5 + 7)
    # Then clamp to [0, 12] for safe indexing.
    dice_num = (hex_features[..., 5] * 5.0 + 7.0).round().long().clamp(0, 12)
    pip = _PIP_TABLE.to(hex_features.device)[dice_num]
    # Zero out desert hexes (per cited observation.rs:80, feature[7] flags desert).
    is_desert = hex_features[..., 7] >= 0.5
    pip = pip.masked_fill(is_desert, 0.0)
    return pip


def compute_vertex_score(hex_pip: torch.Tensor) -> torch.Tensor:
    """Sum pip over hexes adjacent to each vertex.

    Args:
        hex_pip: shape [..., 19]
    Returns:
        vertex_score: shape [..., 54]
    """
    M = _HEX_TO_VERTEX_MATRIX.to(hex_pip.device)
    return hex_pip @ M  # [..., 54]


def build_settlement_target(
    vertex_score: torch.Tensor,
    legal_settle: torch.Tensor,
) -> torch.Tensor:
    """Build a per-sample target distribution over the 54 settlement
    vertices. Mass is proportional to vertex_score on legal vertices,
    zero on illegal. Falls back to uniform over legal if all legal
    vertices have zero score.

    Args:
        vertex_score: shape [..., 54], non-negative float.
        legal_settle: shape [..., 54], bool.
    Returns:
        target: shape [..., 54], sums to 1.0 per sample.
    """
    legal_f = legal_settle.to(torch.float32)
    weighted = vertex_score * legal_f
    score_sum = weighted.sum(dim=-1, keepdim=True)
    has_signal = score_sum > 0
    # Normalize where signal exists; uniform-over-legal otherwise.
    legal_count = legal_f.sum(dim=-1, keepdim=True).clamp(min=1)
    uniform = legal_f / legal_count
    target = torch.where(has_signal, weighted / score_sum.clamp(min=1e-12), uniform)
    return target


def settlement_prior_loss(
    p_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    hex_features: torch.Tensor,
) -> torch.Tensor:
    """Cand 1 auxiliary loss: CE between policy's softmax-over-legal-
    settlements and the pip-weighted target distribution.

    Args:
        p_logits: shape [B, ACTION_SPACE_SIZE]
        legal_mask: shape [B, ACTION_SPACE_SIZE], bool
        hex_features: shape [B, 19, 8]

    Returns:
        Scalar loss. Averaged over samples where at least one settlement
        is legal. Returns 0.0 if no sample has a legal settlement.
    """
    if legal_mask.dtype != torch.bool:
        legal_mask = legal_mask.bool()
    legal_settle = legal_mask[..., 0:54]
    has_legal = legal_settle.any(dim=-1)  # [B]

    if not has_legal.any():
        # No sample needs the prior; return zero (and avoid NaN paths)
        return torch.zeros((), dtype=p_logits.dtype, device=p_logits.device)

    # Compute vertex scores from hex_features
    hex_pip = hex_features_to_pip(hex_features)
    vertex_score = compute_vertex_score(hex_pip)  # [B, 54]

    # Build target distribution
    target = build_settlement_target(vertex_score, legal_settle)  # [B, 54]

    # Restrict logits to settlement subset; log-softmax over legal
    settle_logits = p_logits[..., 0:54]
    masked = settle_logits.masked_fill(~legal_settle, float("-inf"))
    log_pred = F.log_softmax(masked, dim=-1)
    # IEEE 754 trap: 0 * -inf = NaN; zero out log_pred at illegal positions.
    log_pred = log_pred.masked_fill(~legal_settle, 0.0)

    # Per-sample CE: -sum(target * log_pred)
    sample_loss = -(target * log_pred).sum(dim=-1)  # [B]

    # Mask samples with no legal settlement (their loss is 0 by definition
    # of build_settlement_target's uniform fallback, but we want them
    # excluded from the mean).
    sample_loss = sample_loss * has_legal.to(sample_loss.dtype)
    n_firing = has_legal.to(sample_loss.dtype).sum().clamp(min=1)
    return sample_loss.sum() / n_firing
