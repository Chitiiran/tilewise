"""Cand 10 (1-step VP comparison target swap) of the loss-augmentation
roadmap.

The original plan called for replaying the engine at training time to
compute vp(a_model) and vp(a_teacher) one step forward. Insight from
chat 2026-05-11: in v3 (bonuses=False), an action's 1-step VP delta
is FULLY determined by its action_class via CLASS_VP_VALUE.

Cited rules.rs:
  - rules.rs:97, 121, 181 — Settlement actions grant +1 VP unconditionally
  - rules.rs:210 — BuildCity grants +1 VP unconditionally
  - rules.rs:266 — PlayVpCard grants +1 VP unconditionally
  - rules.rs:760-870 — longest-road / largest-army VP transfers all
    gated by `if state.bonuses_enabled` (False in v3)
  - state.rs:230 — dev card deck [14, 2, 2, 2, 5]; BuyDevCard expected
    VP = 5/25 = 0.20

So the comparison vp(a_model) > vp(a_teacher) reduces to
CLASS_VP_VALUE[a_model] > CLASS_VP_VALUE[a_teacher]. No engine call
needed. ~3 lines of tensor math per batch.

The rule:
  for each sample:
    a_model = argmax(masked_logits)
    a_teacher = argmax(policy_t)
    if CLASS_VP_VALUE[a_model] > CLASS_VP_VALUE[a_teacher]:
        target = one_hot(a_model)         # reinforce model's better choice
    else:
        target = policy_t (unchanged)     # keep teacher
"""
from __future__ import annotations

import torch

from .action_classes import _ACTION_VP_VALUE_TENSOR


def vp_compare_swap_target(
    logits: torch.Tensor,
    policy_t: torch.Tensor,
    legal_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Swap supervised target to one-hot(a_model) on samples where the
    model picks a higher-VP action than the teacher.

    Args:
        logits: raw policy head output, shape [B, ACTION_SPACE].
        policy_t: teacher visit-count distribution, shape [B, ACTION_SPACE].
            Pre-normalized (sums to 1 per sample).
        legal_mask: bool mask of legal actions, shape [B, ACTION_SPACE].

    Returns:
        new_target: tensor of shape [B, ACTION_SPACE] with the swap
            applied per-sample. For samples that DON'T swap, the row
            is byte-identical to policy_t.
        swap_count: int, number of samples where the swap fired
            (useful for logging).
    """
    # a_model: argmax over LEGAL logits.
    masked = logits.masked_fill(~legal_mask, float("-inf"))
    a_model = masked.argmax(dim=1)  # [B]

    # a_teacher: argmax of policy_t. (Already zero on illegal positions
    # because the dataset zeros visit counts there.)
    a_teacher = policy_t.argmax(dim=1)  # [B]

    vp_table = _ACTION_VP_VALUE_TENSOR.to(logits.device)
    vp_model = vp_table[a_model]        # [B]
    vp_teacher = vp_table[a_teacher]    # [B]

    # Strict >: a tie (both VP-yielding, or both non-VP) does NOT swap.
    should_swap = vp_model > vp_teacher  # [B] bool

    # Build one-hot(a_model) only for samples that need it.
    # torch.where avoids creating a fresh one-hot for rows that don't swap.
    swap_indices = torch.where(should_swap)[0]
    new_target = policy_t.clone()
    if swap_indices.numel() > 0:
        # zero out those rows, set one-hot at a_model
        new_target[swap_indices] = 0.0
        new_target[swap_indices, a_model[swap_indices]] = 1.0

    swap_count = int(should_swap.sum().item())
    return new_target, swap_count
