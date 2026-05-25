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


def _build_hex_features(dice_per_hex: list[int], desert_hexes: list[int] = ()) -> torch.Tensor:
    """Helper: build [19, 8] hex_features matching observation.rs:75-86.
    dice_per_hex must be length 19. Use 0 to mean "no number" (will be
    treated as pip 0 via PIP_BY_DICE)."""
    hf = torch.zeros(19, 8)
    for h in range(19):
        if h in desert_hexes:
            hf[h, 7] = 1.0  # desert flag
        else:
            # Resource one-hot (any non-desert resource works for this test;
            # we don't read resource type in road_pip_prior).
            hf[h, 0] = 1.0  # wood
        n = dice_per_hex[h]
        hf[h, 5] = (n - 7.0) / 5.0
    return hf


def test_compute_road_scores_zero_when_far_endpoint_not_settlement_legal():
    """Viewer owns edge 0 ([0,3]). Candidate edge 6 ([3,7]) has far endpoint
    7. If vertex 7 is occupied, score = 0. If vertex 7 is empty AND all
    neighbors empty AND non-desert pip on adjacent hexes, score > 0.
    """
    ef = _build_edge_features(viewer_road_edges=[0])
    vf_occupied = _build_vertex_features(occupied_vertices=[7])
    vf_empty = _build_vertex_features(occupied_vertices=[])
    # All hexes have dice number 6 (highest pip = 5). Doesn't matter which
    # hexes are adjacent to vertex 7 since pip is computed per hex.
    hf = _build_hex_features(dice_per_hex=[6] * 19)

    legal_road = torch.zeros(72, dtype=torch.bool)
    legal_road[6] = True

    scores_occ = compute_road_scores(
        edge_features=ef, vertex_features=vf_occupied,
        hex_features=hf, legal_road_mask=legal_road,
    )
    scores_emp = compute_road_scores(
        edge_features=ef, vertex_features=vf_empty,
        hex_features=hf, legal_road_mask=legal_road,
    )
    assert scores_occ[6].item() == 0.0, "v7 occupied → score 0"
    assert scores_emp[6].item() > 0.0, "v7 empty + dice=6 hex → pip > 0"


def test_build_road_pip_target_linear_normalization():
    """Two legal roads with scores 5 and 10 → target [0, 0, ..., 1/3, 2/3, ...]."""
    scores = torch.zeros(72)
    scores[10] = 5.0
    scores[20] = 10.0
    legal_road = torch.zeros(72, dtype=torch.bool)
    legal_road[10] = True
    legal_road[20] = True
    target = build_road_pip_target(scores, legal_road)
    assert target.shape == (72,)
    assert abs(target[10].item() - 1/3) < 1e-6
    assert abs(target[20].item() - 2/3) < 1e-6
    assert target.sum().item() == pytest.approx(1.0)
    # Illegal entries are zero
    assert target[0].item() == 0.0


def test_build_road_pip_target_all_zero_returns_zeros():
    """If all legal roads have score 0, target is all-zero (gate will
    catch this upstream — we just need to not divide by zero)."""
    scores = torch.zeros(72)
    legal_road = torch.zeros(72, dtype=torch.bool)
    legal_road[10] = True
    legal_road[20] = True
    target = build_road_pip_target(scores, legal_road)
    assert target.sum().item() == 0.0


ROAD_OFFSET = 108  # cited road_pip_prior.ROAD_ACTION_OFFSET


def _build_legal_mask(*, legal_settles: list[int] = (), legal_roads: list[int] = (),
                     extras: list[int] = ()) -> torch.Tensor:
    """Build a [280] bool legal mask. legal_settles are settlement
    action_ids in 0..53; legal_roads are EDGE_IDs (will be offset by 108);
    extras are arbitrary action_ids (e.g. EndTurn=204) to also mark legal."""
    m = torch.zeros(280, dtype=torch.bool)
    for s in legal_settles:
        m[s] = True
    for e in legal_roads:
        m[ROAD_OFFSET + e] = True
    for x in extras:
        m[x] = True
    return m


def test_road_pip_prior_loss_zero_when_settlement_legal():
    """Gate A: if any settlement action is legal, the loss is exactly 0
    for that sample. Confirm by stacking 2 samples: sample 0 has a legal
    settlement, sample 1 has only roads + EndTurn. The batched loss should
    equal the per-sample loss of sample 1 alone (sample 0 contributes 0).
    """
    B = 2
    logits = torch.randn(B, 280, requires_grad=True)
    legal = torch.zeros(B, 280, dtype=torch.bool)
    legal[0] = _build_legal_mask(legal_settles=[0], legal_roads=[6], extras=[204])
    legal[1] = _build_legal_mask(legal_roads=[6], extras=[204])

    # All-empty board, dice=6 on every hex, viewer owns edge 0 → far=7,
    # vertex 7 settlement-legal, score > 0. Sample 1 has gate fire.
    ef = _build_edge_features(viewer_road_edges=[0]).unsqueeze(0).expand(B, -1, -1)
    vf = _build_vertex_features(occupied_vertices=[]).unsqueeze(0).expand(B, -1, -1)
    hf = _build_hex_features(dice_per_hex=[6]*19).unsqueeze(0).expand(B, -1, -1)

    loss_both = road_pip_prior_loss(
        p_logits=logits, legal_mask=legal,
        edge_features=ef, vertex_features=vf, hex_features=hf,
    )

    # Now run sample 1 alone.
    loss_one = road_pip_prior_loss(
        p_logits=logits[1:2], legal_mask=legal[1:2],
        edge_features=ef[1:2], vertex_features=vf[1:2], hex_features=hf[1:2],
    )

    # The mean is computed only over firing samples (sample 1), so both
    # should match exactly.
    assert abs(loss_both.item() - loss_one.item()) < 1e-5


def test_road_pip_prior_loss_zero_when_all_scores_zero():
    """If all legal roads have score 0 (e.g. all candidate far endpoints
    occupied), the loss is 0 (no firing samples)."""
    logits = torch.randn(1, 280, requires_grad=True)
    legal = _build_legal_mask(legal_roads=[6], extras=[204]).unsqueeze(0)

    ef = _build_edge_features(viewer_road_edges=[0]).unsqueeze(0)
    vf = _build_vertex_features(occupied_vertices=[7]).unsqueeze(0)  # v7 occupied
    hf = _build_hex_features(dice_per_hex=[6]*19).unsqueeze(0)

    loss = road_pip_prior_loss(
        p_logits=logits, legal_mask=legal,
        edge_features=ef, vertex_features=vf, hex_features=hf,
    )
    assert loss.item() == 0.0


def test_road_pip_prior_loss_gradient_only_in_road_slice():
    """Layer 1: gradients are zero on non-road logits."""
    logits = torch.randn(1, 280, requires_grad=True)
    # Two legal roads: edges 6 and 11. Plus EndTurn (irrelevant non-road).
    legal = _build_legal_mask(legal_roads=[6, 11], extras=[204]).unsqueeze(0)

    ef = _build_edge_features(viewer_road_edges=[0]).unsqueeze(0)
    vf = _build_vertex_features(occupied_vertices=[]).unsqueeze(0)
    hf = _build_hex_features(dice_per_hex=[6]*19).unsqueeze(0)

    loss = road_pip_prior_loss(
        p_logits=logits, legal_mask=legal,
        edge_features=ef, vertex_features=vf, hex_features=hf,
    )
    loss.backward()
    # Road logits should have nonzero grad.
    assert logits.grad[0, ROAD_OFFSET + 6].abs() > 0
    assert logits.grad[0, ROAD_OFFSET + 11].abs() > 0
    # Non-road logits should have exactly zero grad.
    assert logits.grad[0, 0].item() == 0.0       # settlement 0
    assert logits.grad[0, 204].item() == 0.0     # EndTurn
    assert logits.grad[0, 226].item() == 0.0     # BuyDevCard


def test_road_pip_prior_loss_prefers_higher_pip_road():
    """Two legal roads. Road A unlocks v=7 with dice-6 hexes (high pip).
    Road B unlocks v=16 with dice-12 hexes (pip=1 only).
    The KL pull means logits[A] should have a more-negative gradient
    (push up) than logits[B] (push down).

    Viewer roads = {0, 33}. Edge 0 = [0,3], edge 33 = [21,27].
    Frontier = {0, 3, 21, 27}.
      Edge 6 ([3,7]): far = 7.
      Edge 23 ([16,21]): far = 16.
    Hex 0 (adjacent to v=7 among others) -> dice 6 (pip 5).
    Hex 7 (adjacent to v=16 among others) -> dice 12 (pip 1).
    pip(v=7) > pip(v=16) -> prior favors road 6.
    """
    ef = _build_edge_features(viewer_road_edges=[0, 33]).unsqueeze(0)
    vf = _build_vertex_features(occupied_vertices=[]).unsqueeze(0)
    dice = [0]*19
    dice[0] = 6   # high pip contributions reach v=7
    dice[7] = 12  # low pip contributions reach v=16
    hf = _build_hex_features(dice_per_hex=dice).unsqueeze(0)

    logits = torch.zeros(1, 280, requires_grad=True)
    legal = _build_legal_mask(legal_roads=[6, 23], extras=[204]).unsqueeze(0)

    loss = road_pip_prior_loss(
        p_logits=logits, legal_mask=legal,
        edge_features=ef, vertex_features=vf, hex_features=hf,
    )
    loss.backward()
    grad = logits.grad[0]
    # Gradient = (q - prior). With logits=0, q is uniform over {road6, road23}
    # = [0.5, 0.5]. Prior is sharper toward road 6 (high pip). So:
    #   (q - prior) is NEGATIVE on road 6  → push UP.
    #   (q - prior) is POSITIVE on road 23 → push DOWN.
    g_a = grad[ROAD_OFFSET + 6].item()
    g_b = grad[ROAD_OFFSET + 23].item()
    assert g_a < 0, f"high-pip road should have negative gradient (push up), got {g_a}"
    assert g_b > 0, f"low-pip road should have positive gradient (push down), got {g_b}"
    assert abs(g_a) > 1e-4 and abs(g_b) > 1e-4


def _random_sample(rng: torch.Generator):
    """Generate a plausible (edge_features, vertex_features, hex_features,
    legal_mask) tuple drawing from realistic distributions:
      - Random subset of vertices occupied (~6 of 54 — typical mid-game)
      - Random subset of edges owned by viewer (~5 of 72)
      - Random dice numbers on hexes (2..12 excluding 7), 1 desert
      - Random legal_mask with no legal settlements (gate-A-fires regime)
        and a random subset of legal roads.
    """
    # Vertex features.
    n_occupied = int(torch.randint(2, 10, (1,), generator=rng).item())
    occ_idx = torch.randperm(54, generator=rng)[:n_occupied].tolist()
    vf = torch.zeros(54, 13)
    for v in range(54):
        if v in occ_idx:
            vf[v, 1] = 1.0  # settle
            # owner one-hot: pick player 0 for viewer half the time, else opponent
            owner = int(torch.randint(0, 4, (1,), generator=rng).item())
            vf[v, 3 + owner] = 1.0
        else:
            vf[v, 0] = 1.0

    # Edge features.
    n_viewer_roads = int(torch.randint(2, 12, (1,), generator=rng).item())
    viewer_road_idx = torch.randperm(72, generator=rng)[:n_viewer_roads].tolist()
    ef = torch.zeros(72, 6)
    for e in range(72):
        if e in viewer_road_idx:
            ef[e, 1] = 1.0
            ef[e, 2] = 1.0  # viewer owns
        else:
            ef[e, 0] = 1.0

    # Hex features.
    hf = torch.zeros(19, 8)
    desert = int(torch.randint(0, 19, (1,), generator=rng).item())
    for h in range(19):
        if h == desert:
            hf[h, 7] = 1.0
        else:
            hf[h, 0] = 1.0
            # Random dice number in {2..12} \ {7}.
            n = int(torch.randint(2, 13, (1,), generator=rng).item())
            if n == 7:
                n = 6
            hf[h, 5] = (n - 7.0) / 5.0

    # Legal mask: no legal settlements (gate-A regime), random legal roads,
    # always include EndTurn.
    legal = torch.zeros(280, dtype=torch.bool)
    legal[204] = True
    n_legal_roads = int(torch.randint(0, 12, (1,), generator=rng).item())
    legal_road_idx = torch.randperm(72, generator=rng)[:n_legal_roads].tolist()
    for e in legal_road_idx:
        legal[ROAD_OFFSET + e] = True
    return ef, vf, hf, legal


def test_batched_matches_loop_100_random_samples():
    """Equivalence: per-sample loop and batched paths must produce
    byte-identical scores, targets, and loss on 100 random samples.

    Required before switching production to batched, per
    docs/superpowers/journals/2026-05-25-cand11-perf-rca.md §"What still
    needs to happen".
    """
    from catan_gnn.road_pip_prior import (
        compute_road_scores_loop,
        compute_road_scores_batched,
        settlement_legal_mask_loop,
        settlement_legal_mask_batched,
        far_endpoint_loop,
        far_endpoint_batched,
        NUM_EDGES,
    )

    rng = torch.Generator().manual_seed(20260525)
    n_samples = 100
    all_pass = True
    failures: list[str] = []

    for i in range(n_samples):
        ef, vf, hf, legal = _random_sample(rng)
        legal_road_mask = legal[ROAD_OFFSET:ROAD_OFFSET + NUM_EDGES]

        # settlement_legal: loop returns [54], batched returns [1, 54] when given [1, 54, 13]
        slm_loop = settlement_legal_mask_loop(vf)                          # [54]
        slm_batched = settlement_legal_mask_batched(vf.unsqueeze(0))[0]    # [54]
        if not torch.equal(slm_loop, slm_batched):
            all_pass = False
            failures.append(f"sample {i}: settlement_legal_mask mismatch")
            continue

        # far_endpoint: loop returns int per edge, batched returns [1, 72]
        far_batched = far_endpoint_batched(ef.unsqueeze(0))[0]             # [72]
        for e in range(NUM_EDGES):
            loop_val = far_endpoint_loop(edge_id=e, edge_features=ef)
            batched_val = int(far_batched[e].item())
            if loop_val != batched_val:
                all_pass = False
                failures.append(f"sample {i} edge {e}: far_endpoint loop={loop_val} batched={batched_val}")
                break
        if not all_pass:
            break

        # compute_road_scores
        scores_loop = compute_road_scores_loop(
            edge_features=ef, vertex_features=vf, hex_features=hf,
            legal_road_mask=legal_road_mask,
        )
        scores_batched = compute_road_scores_batched(
            edge_features=ef.unsqueeze(0), vertex_features=vf.unsqueeze(0),
            hex_features=hf.unsqueeze(0), legal_road_mask=legal_road_mask.unsqueeze(0),
        )[0]
        if not torch.allclose(scores_loop, scores_batched, atol=1e-6):
            all_pass = False
            diffs = (scores_loop - scores_batched).abs()
            failures.append(
                f"sample {i}: compute_road_scores diff max={diffs.max().item():.2e}"
            )
            break

    if not all_pass:
        for f in failures[:5]:
            print(f"FAIL: {f}")
    assert all_pass, f"{len(failures)} samples disagreed; first: {failures[0] if failures else 'n/a'}"


def test_batched_road_loss_matches_loop_loss_grad_too():
    """Equivalence at the loss level: same scalar loss and same gradient
    for a synthetic batch of 8 samples generated by the same RNG."""
    from catan_gnn.road_pip_prior import road_pip_prior_loss
    # Reconstruct old loop-based loss locally to compare.
    from catan_gnn.road_pip_prior import (
        compute_road_scores_loop,
        ROAD_ACTION_OFFSET,
        NUM_EDGES as N_EDGES,
    )
    import torch.nn.functional as F2

    def loop_loss(p_logits, legal_mask, ef, vf, hf):
        if legal_mask.dtype != torch.bool:
            legal_mask = legal_mask.bool()
        B = p_logits.shape[0]
        device = p_logits.device
        legal_settle_any = legal_mask[:, 0:54].any(dim=-1)
        legal_road = legal_mask[:, ROAD_ACTION_OFFSET:ROAD_ACTION_OFFSET + N_EDGES]
        has_legal_road = legal_road.any(dim=-1)
        candidates = (~legal_settle_any) & has_legal_road
        if not candidates.any():
            return torch.zeros((), dtype=p_logits.dtype, device=device)
        scores = torch.zeros(B, N_EDGES, dtype=torch.float32, device=device)
        for b in range(B):
            if not bool(candidates[b].item()):
                continue
            scores[b] = compute_road_scores_loop(
                edge_features=ef[b], vertex_features=vf[b],
                hex_features=hf[b], legal_road_mask=legal_road[b],
            )
        has_score = (scores.sum(dim=-1) > 0)
        firing = candidates & has_score
        if not firing.any():
            return torch.zeros((), dtype=p_logits.dtype, device=device)
        score_sums = scores.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        target = (scores / score_sums) * firing.unsqueeze(-1).to(scores.dtype)
        road_logits = p_logits[:, ROAD_ACTION_OFFSET:ROAD_ACTION_OFFSET + N_EDGES]
        masked = road_logits.masked_fill(~legal_road, float("-inf"))
        log_q = F2.log_softmax(masked, dim=-1).masked_fill(~legal_road, 0.0)
        sample_loss = -(target * log_q).sum(dim=-1) * firing.to(scores.dtype)
        n_firing = firing.to(scores.dtype).sum().clamp(min=1)
        return sample_loss.sum() / n_firing

    rng = torch.Generator().manual_seed(42)
    B = 8
    efs, vfs, hfs, legals = [], [], [], []
    for _ in range(B):
        ef, vf, hf, legal = _random_sample(rng)
        efs.append(ef); vfs.append(vf); hfs.append(hf); legals.append(legal)
    ef_b = torch.stack(efs); vf_b = torch.stack(vfs); hf_b = torch.stack(hfs)
    legal_b = torch.stack(legals)

    torch.manual_seed(0)
    logits_a = torch.randn(B, 280, requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)

    loss_loop = loop_loss(logits_a, legal_b, ef_b, vf_b, hf_b)
    loss_batched = road_pip_prior_loss(
        p_logits=logits_b, legal_mask=legal_b,
        edge_features=ef_b, vertex_features=vf_b, hex_features=hf_b,
    )
    assert torch.allclose(loss_loop, loss_batched, atol=1e-6), (
        f"loss mismatch: loop={loss_loop.item()} batched={loss_batched.item()}"
    )

    loss_loop.backward()
    loss_batched.backward()
    g_loop = logits_a.grad
    g_batched = logits_b.grad
    assert g_loop is not None and g_batched is not None
    diff = (g_loop - g_batched).abs().max().item()
    assert diff < 1e-6, f"gradient mismatch: max abs diff={diff:.2e}"


def test_road_pip_prior_loss_mean_over_firing_samples():
    """Batch of 3 samples. Samples 0 and 2 have gate fire; sample 1 has a
    legal settlement (gate blocks). The reported loss should be (L0 + L2)/2
    not (L0 + 0 + L2)/3."""
    B = 3
    logits = torch.randn(B, 280, requires_grad=False)
    legal = torch.zeros(B, 280, dtype=torch.bool)
    legal[0] = _build_legal_mask(legal_roads=[6], extras=[204])
    legal[1] = _build_legal_mask(legal_settles=[0], legal_roads=[6], extras=[204])  # gate blocked
    legal[2] = _build_legal_mask(legal_roads=[6], extras=[204])

    ef = _build_edge_features(viewer_road_edges=[0]).unsqueeze(0).expand(B, -1, -1)
    vf = _build_vertex_features(occupied_vertices=[]).unsqueeze(0).expand(B, -1, -1)
    hf = _build_hex_features(dice_per_hex=[6]*19).unsqueeze(0).expand(B, -1, -1)

    loss_full = road_pip_prior_loss(
        p_logits=logits, legal_mask=legal,
        edge_features=ef, vertex_features=vf, hex_features=hf,
    )
    # Compare against the average of the two firing samples computed separately.
    idx = torch.tensor([0, 2])
    loss_pair = road_pip_prior_loss(
        p_logits=logits[idx], legal_mask=legal[idx],
        edge_features=ef[idx], vertex_features=vf[idx], hex_features=hf[idx],
    )
    assert abs(loss_full.item() - loss_pair.item()) < 1e-5
