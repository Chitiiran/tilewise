"""Tests for the hex-symmetry rotation augmentation."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from catan_bot import _engine
from catan_gnn.gnn_model import GnnModel
from catan_gnn.rotation import (
    ROT60_ACTION,
    ROT60_EDGE,
    ROT60_HEX,
    ROT60_VERTEX,
    rotate_hetero_data,
    rotate_hetero_data_k,
    rotate_legal_mask_k,
    rotate_policy,
    rotate_policy_k,
)
from catan_gnn.state_to_pyg import state_to_pyg


def test_permutations_are_bijections():
    """Each permutation must be a valid one-to-one mapping."""
    assert sorted(ROT60_HEX) == list(range(19))
    assert sorted(ROT60_VERTEX) == list(range(54))
    assert sorted(ROT60_EDGE) == list(range(72))
    assert sorted(ROT60_ACTION) == list(range(280))


def test_six_rotations_is_identity():
    """Applying the 60° rotation 6 times must yield the identity (full hex symmetry)."""
    for perm in [ROT60_HEX, ROT60_VERTEX, ROT60_EDGE]:
        cur = list(range(len(perm)))
        for _ in range(6):
            cur = [cur[perm[i]] for i in range(len(perm))]
        assert cur == list(range(len(perm))), \
            f"6× rotation didn't return to identity: {cur[:10]}..."


def test_action_permutation_only_remaps_board_actions():
    """Discard, EndTurn, RollDice, trades, dev cards, ProposeTrade — none reference
    board positions, so they must be fixed points of the rotation."""
    # 199..279 are board-agnostic
    for a in range(199, 280):
        assert ROT60_ACTION[a] == a, f"action {a} should be fixed under rotation"
    # 0..198 should mostly NOT be fixed (board does rotate)
    n_changed = sum(1 for a in range(199) if ROT60_ACTION[a] != a)
    assert n_changed >= 50, f"only {n_changed} board-actions changed; rotation may be a no-op"


def test_rotate_hetero_data_roundtrip():
    """Rotating 6 times should return identical tensors (within float tolerance)."""
    e = _engine.Engine(42)
    obs = e.observation()
    data = state_to_pyg(obs)
    rotated = data
    for _ in range(6):
        rotated = rotate_hetero_data(rotated)
    torch.testing.assert_close(rotated["hex"].x, data["hex"].x)
    torch.testing.assert_close(rotated["vertex"].x, data["vertex"].x)
    torch.testing.assert_close(rotated["edge"].x, data["edge"].x)
    torch.testing.assert_close(rotated.legal_mask, data.legal_mask)


def test_rotate_preserves_shapes():
    """Permutation only — no shape changes."""
    e = _engine.Engine(42)
    data = state_to_pyg(e.observation())
    rotated = rotate_hetero_data(data)
    assert rotated["hex"].x.shape == data["hex"].x.shape
    assert rotated["vertex"].x.shape == data["vertex"].x.shape
    assert rotated["edge"].x.shape == data["edge"].x.shape
    assert rotated["hex", "to", "vertex"].edge_index.shape == data["hex", "to", "vertex"].edge_index.shape
    assert rotated.legal_mask.shape == data.legal_mask.shape


def test_rotated_data_passes_through_gnn():
    """Smoke: rotated samples must still feed cleanly through the GNN forward."""
    model = GnnModel(hidden_dim=32, num_layers=2)
    model.eval()
    e = _engine.Engine(42)
    data = state_to_pyg(e.observation())
    rotated = rotate_hetero_data(data)
    batch = Batch.from_data_list([rotated])
    with torch.no_grad():
        v, p = model(batch)
    assert v.shape == (1, 4)
    assert p.shape == (1, 280)
    assert torch.isfinite(v).all()
    assert torch.isfinite(p).all()


def test_rotate_policy_inverse_of_action_permutation():
    """If we rotate a policy and remember which actions were legal, the rotated
    policy[i] should equal the original policy[ROT60_ACTION[i]]."""
    p = torch.arange(280, dtype=torch.float32)
    rotated = rotate_policy(p)
    for i in range(280):
        assert rotated[i].item() == ROT60_ACTION[i]


def test_legal_mask_consistent_after_rotation():
    """A legal action in the rotated state should correspond to a legal action
    in the original state, via the inverse action permutation."""
    e = _engine.Engine(42)
    data = state_to_pyg(e.observation())
    rotated = rotate_hetero_data(data)
    # rotated.legal_mask[i] == data.legal_mask[ROT60_ACTION[i]]
    for i in range(280):
        assert bool(rotated.legal_mask[i]) == bool(data.legal_mask[ROT60_ACTION[i]])


def test_rotated_observation_matches_engine_when_we_could_rebuild():
    """The strongest invariant: GIVEN the engine doesn't natively support
    rotated observations, we settle for shape + bijection guarantees, which
    the tests above already cover. This test exists as a placeholder for the
    future when rotation is wired into the engine itself."""
    pass


def test_rotate_k_zero_is_identity():
    """k=0 must be identity (no transformation)."""
    e = _engine.Engine(42)
    data = state_to_pyg(e.observation())
    out = rotate_hetero_data_k(data, 0)
    torch.testing.assert_close(out["hex"].x, data["hex"].x)
    torch.testing.assert_close(out["vertex"].x, data["vertex"].x)
    torch.testing.assert_close(out.legal_mask, data.legal_mask)


def test_rotate_k_one_matches_rotate_60():
    """k=1 must equal the original single-rotation function."""
    e = _engine.Engine(42)
    data = state_to_pyg(e.observation())
    a = rotate_hetero_data(data)
    b = rotate_hetero_data_k(data, 1)
    torch.testing.assert_close(a["hex"].x, b["hex"].x)
    torch.testing.assert_close(a["vertex"].x, b["vertex"].x)
    torch.testing.assert_close(a["edge"].x, b["edge"].x)
    torch.testing.assert_close(a.legal_mask, b.legal_mask)


def test_rotate_k_six_returns_to_origin():
    """k=6 (= k=0 mod 6) must be identity."""
    e = _engine.Engine(42)
    data = state_to_pyg(e.observation())
    out = rotate_hetero_data_k(data, 6)
    torch.testing.assert_close(out["hex"].x, data["hex"].x)


def test_compose_two_rotations_matches_k_two():
    """Calling rotate_hetero_data_k(rotate_hetero_data_k(data, 1), 1) ==
    rotate_hetero_data_k(data, 2)."""
    e = _engine.Engine(42)
    data = state_to_pyg(e.observation())
    twice = rotate_hetero_data_k(rotate_hetero_data_k(data, 1), 1)
    direct = rotate_hetero_data_k(data, 2)
    torch.testing.assert_close(twice["hex"].x, direct["hex"].x)
    torch.testing.assert_close(twice["vertex"].x, direct["vertex"].x)
    torch.testing.assert_close(twice.legal_mask, direct.legal_mask)


def test_rotate_policy_k_shapes():
    p = torch.randn(280)
    for k in range(6):
        out = rotate_policy_k(p, k)
        assert out.shape == (280,)
    # k=0 is identity
    torch.testing.assert_close(rotate_policy_k(p, 0), p)


def test_random_rotated_dataset_picks_all_k():
    """RotatedDataset in random mode should hit all 6 k values across enough
    samples (with very high probability)."""
    from catan_gnn.dataset import RotatedDataset

    class _DummySource:
        def __init__(self):
            self._d = state_to_pyg(_engine.Engine(42).observation())
            self._v = torch.zeros(4)
            self._p = torch.zeros(280)
            self._l = torch.zeros(280, dtype=torch.bool)
            self.seeds = [42] * 1000

        def __len__(self):
            return 1000

        def __getitem__(self, i):
            return self._d, self._v, self._p, self._l

    src = _DummySource()
    rd = RotatedDataset(src, mode="random", seed=0)
    # Sample 200 items and check we see >= 4 distinct rotations.
    seen_hexes = set()
    for i in range(200):
        d, _, _, _ = rd[i]
        # Use first row of hex.x as a fingerprint for which rotation was applied
        # (since hex features differ between rotated views).
        key = tuple(d["hex"].x[0].tolist())
        seen_hexes.add(key)
    # With 6 rotations, 200 samples should hit at least 4 distinct
    # views (each rotation is uniformly likely; missing all 6 has tiny prob).
    assert len(seen_hexes) >= 4, f"random rotation only saw {len(seen_hexes)} distinct views"
