"""Dataset must survive a single un-replayable position instead of crashing
the whole training run (2026-06-14: a rare, non-reproducible replay failure
on one self-play game killed iter-3 training after 6h of self-play).

Defense-in-depth (spec 2026-06-13 §2): an algorithmic/data glitch on one
position is surfaced + skipped, never fatal. The dataset substitutes the next
valid position so the batch stays full and training continues."""
from __future__ import annotations

import torch


class _FakeReplayDataset:
    """Minimal stand-in exercising the skip-and-substitute contract without a
    full corpus: __getitem__ raises for a 'poisoned' index, and the resilient
    wrapper must return a valid neighbor instead of propagating."""

    def __init__(self, n, poisoned):
        self.n = n
        self.poisoned = set(poisoned)

    def __len__(self):
        return self.n

    def _replay_getitem(self, i):
        if i in self.poisoned:
            raise RuntimeError(f"Could not replay position {i}")
        # valid sample: (data, value, policy, legal)
        return (f"data{i}", torch.zeros(4), torch.zeros(280),
                torch.ones(280, dtype=torch.bool))


def test_skip_and_substitute_returns_valid_neighbor():
    from catan_gnn.dataset import resilient_getitem
    ds = _FakeReplayDataset(n=10, poisoned={3})
    # index 3 is poisoned -> must return a VALID neighbor, never raise
    data, value, policy, legal = resilient_getitem(ds, 3)
    assert data != "data3"          # substituted, not the poisoned one
    assert legal.dtype == torch.bool


def test_valid_index_passes_through():
    from catan_gnn.dataset import resilient_getitem
    ds = _FakeReplayDataset(n=10, poisoned={3})
    data, *_ = resilient_getitem(ds, 5)
    assert data == "data5"


def test_all_poisoned_raises_loudly():
    """If EVERY candidate fails, that's a real corruption -> fail loud (not an
    infinite loop)."""
    import pytest

    from catan_gnn.dataset import resilient_getitem
    ds = _FakeReplayDataset(n=4, poisoned={0, 1, 2, 3})
    with pytest.raises(RuntimeError, match="no replayable"):
        resilient_getitem(ds, 0, max_tries=4)
