"""Tests for the visits^p distillation target (--policy-sharpen).

Why: the 2026-06-01 deploy-valley investigation showed PureGnn's plateau is
caused by honestly-soft visit-count targets + argmax discarding value info.
The recommended (untested) lever is sharpening the teacher's visit
distribution so the student's argmax inherits the searcher's decision.
Sharpening operates on the normalized target: (v/s)^p ∝ v^p, so this equals
sharpening raw visits. Applied in the train loop (like Cand 7/10 transforms)
so cached datasets stay valid.
"""
from __future__ import annotations

import pytest
import torch


def _sharpen(t, p):
    from catan_gnn.train import sharpen_policy_target
    return sharpen_policy_target(t, p)


def test_p1_is_identity():
    t = torch.tensor([[0.6, 0.3, 0.1]])
    out = _sharpen(t, 1.0)
    assert torch.allclose(out, t)


def test_p2_sharpens_toward_argmax():
    t = torch.tensor([[0.6, 0.3, 0.1]])
    out = _sharpen(t, 2.0)
    expected = torch.tensor([[0.36, 0.09, 0.01]]) / 0.46
    assert torch.allclose(out, expected, atol=1e-6)
    # Strictly sharper: top prob grows, tail shrinks.
    assert out[0, 0] > t[0, 0] and out[0, 2] < t[0, 2]


def test_exact_ties_preserved():
    t = torch.tensor([[0.5, 0.5, 0.0]])
    out = _sharpen(t, 2.0)
    assert torch.allclose(out, t)


def test_rows_stay_normalized():
    g = torch.Generator().manual_seed(0)
    t = torch.rand(8, 280, generator=g)
    t = t / t.sum(dim=1, keepdim=True)
    out = _sharpen(t, 2.0)
    assert torch.allclose(out.sum(dim=1), torch.ones(8), atol=1e-5)


def test_zero_row_stays_zero_no_nan():
    t = torch.zeros(2, 5)
    t[1, 2] = 1.0
    out = _sharpen(t, 2.0)
    assert not torch.isnan(out).any()
    assert torch.all(out[0] == 0)
    assert out[1, 2] == pytest.approx(1.0)


def test_no_grad_leak():
    t = torch.tensor([[0.7, 0.3]], requires_grad=True)
    out = _sharpen(t, 2.0)
    assert not out.requires_grad


def test_train_main_accepts_policy_sharpen_kwarg():
    """train_main() must expose policy_sharpen; checked via signature so the
    test stays fast (no actual training)."""
    import inspect
    from catan_gnn.train import train_main
    assert "policy_sharpen" in inspect.signature(train_main).parameters
