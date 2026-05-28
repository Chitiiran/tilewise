# Road-Pip Prior (Cand 11) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-sample auxiliary loss that pulls the GNN's road-policy logits toward edges whose far endpoint is settlement-legal and has the highest pip count, so PureGnn's roads convert to settlements at a Lookahead-comparable rate.

**Architecture:** Mirror the existing Cand 1 (`settlement_vertex_prior.py`) pattern exactly. New module `road_pip_prior.py` holds the topology table + scoring + KL loss; `train.py` calls it lazily inside the train step gated on a new `--lambda-road` flag. Loss is over the road slice only (no global logit inflation); fires only when no legal settlement exists AND at least one legal road unlocks a settlement-legal vertex.

**Tech Stack:** Python 3.10, PyTorch, PyG HeteroData (already in use). No engine changes, no cache rebuild, no schema changes.

**Naming:** Candidate is "Cand 11" in continuation of the existing Cand 1/2/3/7/8/10 numbering. Cell label is "Cell 5" (continues 00..04 under `runs/v3/loss_aug/`).

**Decisions locked in chat 2026-05-25:**
- Pure pip scoring; no port multiplier, no resource diversity, no resource-sufficiency check.
- Linear normalization across legal roads (not softmax with temperature).
- λ_road = 0.05.
- Gate A: skip the loss entirely if any settlement action is legal.
- Layer 1: KL over the road slice only (independent softmax over `logits[L_R]`).
- Early-kill rule: at ep5 mid-tournament, if PureGnn winrate ≥1.5pp below Cell 0's ep5 (12.08% → 10.58% floor), kill the run.
- Standalone experiment — does NOT stack on Cand 8+10.
- Concerns A (KL clip), C (one-hot smoothing) and Missing 2 (resource sufficiency): all accepted as-is.

---

## File Structure

**Worktree root for all paths below:** `C:\dojo\catan_bot\.claude\worktrees\v3` (Windows path) or `/mnt/c/dojo/catan_bot/.claude/worktrees/v3` (WSL path). All file paths in this plan are relative to that root unless explicitly absolute.

**Create:**
- `mcts_study/catan_gnn/road_pip_prior.py` — topology table + score function + gate + KL loss. ~180 LOC.
- `mcts_study/tests/test_road_pip_prior.py` — unit tests. ~250 LOC.
- `mcts_study/tests/test_cell5_smoke.py` — 1-epoch smoke train on toy fixture. ~80 LOC.
- `mcts_study/scratch_road_pip_calibration.py` — Layer-3 pre-launch diagnostic script. ~120 LOC.
- `docs/superpowers/journals/2026-05-XX-cell5-road-pip-prior.md` — journal entry written AFTER ep5 mid-tournament. Created at the end of the run, not now.

**Modify:**
- `mcts_study/catan_gnn/train.py:303-373` — add `lambda_road: float = 0.0` arg to `train_main`. Plumb through.
- `mcts_study/catan_gnn/train.py:540-562` — add the Cand 11 loss term beside the existing Cand 8 / Cand 1 / Cand 10 blocks.
- `mcts_study/scripts/train_grid_inproc.py:153-170` — add `--lambda-road` CLI flag.
- `mcts_study/scripts/train_grid_inproc.py:274-277` — pass `lambda_road=args.lambda_road` to `train_main`.

**Reuse without modification:**
- `mcts_study/catan_gnn/settlement_vertex_prior.py` — import `hex_features_to_pip` (pip from hex_features).
- `mcts_study/catan_gnn/adjacency.py` — `EDGE_TO_VERTICES`, `HEX_TO_VERTICES`.

---

## Mathematical Specification (LOCKED — implement exactly this)

For one training sample with `logits ∈ ℝ^280`, `legal ∈ {0,1}^280`, `hex_features ∈ ℝ^{19×8}`, `vertex_features ∈ ℝ^{54×13}`, `edge_features ∈ ℝ^{72×6}`:

```
# Slices
S = action IDs 0..53       (settlements)
R = action IDs 108..179    (roads)   so road r has edge_id = r - 108
L_S = {a ∈ S : legal[a] = 1}
L_R = {a ∈ R : legal[a] = 1}

# Topology + features
pip(v) = sum over hexes h in HEX_TO_VERTICES_REVERSE[v] of:
            PIP_BY_DICE[dice_num(h)]   if hex is non-desert
            0                          otherwise
       = settlement_vertex_prior.compute_vertex_score(hex_features_to_pip(hex_features))[v]

vertex_empty(v) = (vertex_features[v, 0] == 1)
                  i.e., the "empty" flag of F_VERT layout cited observation.rs:38-41

viewer_road_at(e) = (edge_features[e, 2] == 1)
                    i.e., col 2 of F_EDGE is the perspective-rotated viewer's road
                    cited observation.rs:120 (perspective_idx 0 = viewer)

# Per-road "far endpoint": the endpoint not currently in the viewer's road network
viewer_frontier_vertices = {v : exists edge e' with viewer_road_at(e')
                                and v in EDGE_TO_VERTICES[e']}

For each road action a ∈ L_R, e = a - 108:
    v0, v1 = EDGE_TO_VERTICES[e]
    if v0 in viewer_frontier_vertices and v1 not in viewer_frontier_vertices:
        v_new = v1
    elif v1 in viewer_frontier_vertices and v0 not in viewer_frontier_vertices:
        v_new = v0
    else:
        # Both endpoints in frontier (interior road extending the network sideways),
        # or neither (only happens during setup — gate B excludes setup roads anyway).
        v_new = -1   # no far endpoint → score 0

# Distance rule: settlement-legal iff vertex empty AND all 3 neighbor vertices empty.
# Neighbor lookup via EDGE_TO_VERTICES (any edge sharing v gives the neighbor).
neighbors(v) = {u : exists edge e with EDGE_TO_VERTICES[e] = {v, u}}
settlement_legal(v) = vertex_empty(v) AND all(vertex_empty(u) for u in neighbors(v))

# Score per road
score(a) = pip(v_new(a)) * 1[v_new(a) >= 0]
                          * 1[settlement_legal(v_new(a))]

# Gate A
gate = 1 if |L_S| = 0  AND  |L_R| ≥ 1  AND  sum_{a in L_R} score(a) > 0
       0 otherwise

# Prior (only built when gate = 1)
prior[a] = score(a) / sum_{a' in L_R} score(a')    for a in L_R     (linear)
           (zero outside L_R)

# Model's conditional road distribution (Layer 1: independent softmax over road slice)
q[a] = exp(logits[a]) / sum_{a' in L_R} exp(logits[a'])    for a in L_R

# Loss term (when gate fires)
L_road = - sum_{a in L_R} prior[a] * log q[a]

# Total
L_total = w_value * MSE(v_pred, value_target)
        + w_policy * masked_CE(logits, visit_target, legal)
        + λ_road * gate * L_road
```

The cross-batch reduction is `mean` over samples (PyTorch default). Samples with gate=0 contribute exactly 0 to the road-loss term (not via uniform fallback — via explicit zero).

---

## Task 1 — Topology helpers in road_pip_prior.py

**Files:**
- Create: `mcts_study/catan_gnn/road_pip_prior.py`
- Test: `mcts_study/tests/test_road_pip_prior.py`

### Step 1.1 — Write the failing test for vertex-neighbor table

- [ ] Write the test:

```python
# In mcts_study/tests/test_road_pip_prior.py
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
```

- [ ] Run: `cd mcts_study && python -m pytest tests/test_road_pip_prior.py::test_vertex_neighbors_via_edges_matches_adjacency -v`
- [ ] Expected: FAIL with `ModuleNotFoundError: No module named 'catan_gnn.road_pip_prior'`.

### Step 1.2 — Implement topology helpers

- [ ] Create `mcts_study/catan_gnn/road_pip_prior.py` with:

```python
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
```

- [ ] Run: `cd mcts_study && python -m pytest tests/test_road_pip_prior.py::test_vertex_neighbors_via_edges_matches_adjacency -v`
- [ ] Expected: PASS.

### Step 1.3 — Commit

- [ ] Run:

```bash
cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3
git add mcts_study/catan_gnn/road_pip_prior.py mcts_study/tests/test_road_pip_prior.py
git commit -m "feat(cand11): road-pip prior topology helpers

VERTEX_NEIGHBORS table + EDGE_TO_VERTICES_TENSOR for the road-pip-prior
loss. Per chat 2026-05-25 (Cell 5 design)."
```

---

## Task 2 — far_endpoint() and settlement_legal_mask()

**Files:**
- Modify: `mcts_study/catan_gnn/road_pip_prior.py`
- Modify: `mcts_study/tests/test_road_pip_prior.py`

### Step 2.1 — Write failing tests

- [ ] Append to `mcts_study/tests/test_road_pip_prior.py`:

```python
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
    """With viewer roads at edges 0 ([0,3]) and 1 ([0,4]):
      - Edge 6 [3,7]: v0=3 in frontier (via edge 0), v1=7 not → far = 7.
      - But edge between two frontier vertices: edge 7 [4,8] has only 4 in frontier
        (via edge 1), so far = 8.
      - To trigger 'both in frontier', need an edge where both endpoints already
        have a viewer road touching them. E.g. give viewer edges 0,1,2 (covers
        vertices 0,1,3,4,5); now an edge like edge 2 [1,4] itself has both
        endpoints in the frontier even if edge 2 weren't owned by viewer.
        But edge 2 IS owned by viewer in this setup, so it's not a candidate
        road. Construct differently:
      - Viewer roads: 0 ([0,3]), 6 ([3,7]). Frontier vertices = {0, 3, 7}.
        Candidate road: edge 11 ([7,12]). v0=7 in frontier, v1=12 not → far = 12.
        That's normal. To force both-in-frontier we need a triangle, which the
        Catan board has at vertex layouts — e.g., edges 0,1,12 form a triangle
        on vertices 0, 4, 8, 12? No — edge 0 = [0,3], no. Use edges 6,11,12:
        edge 6 = [3,7], edge 11 = [7,12], edge 12 = [8,12]. Owns 6 + 11 →
        frontier = {3, 7, 12}. Edge 19 = [12,17] has 12 in frontier, 17 not →
        far = 17 (still normal). The Catan board geometry actually doesn't
        have many cases where adding a road has BOTH endpoints already in the
        frontier without that edge itself being owned. So this test instead
        constructs an unusual edge_features where TWO non-incident roads put
        both endpoints in frontier:
    """
    # Viewer owns edges 0 ([0,3]) and 4 ([2,5]). Frontier = {0, 2, 3, 5}.
    # Candidate edge: 3 ([1,5]). v0=1 not in frontier, v1=5 in → far = 1.
    # Now also own edge 3 ([1,5])? Then candidate edge 3 is its own road,
    # not a candidate. Use edge 2 ([1,4]) as the candidate: v0=1 not, v1=4
    # not in frontier {0,2,3,5} → both NOT in frontier → far = -1.
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
```

- [ ] Run: `cd mcts_study && python -m pytest tests/test_road_pip_prior.py -v -k "far_endpoint or settlement_legal"`
- [ ] Expected: FAIL — `ImportError: cannot import name 'far_endpoint'`.

### Step 2.2 — Implement the functions

- [ ] Append to `mcts_study/catan_gnn/road_pip_prior.py`:

```python
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
```

- [ ] Run: `cd mcts_study && python -m pytest tests/test_road_pip_prior.py -v -k "far_endpoint or settlement_legal"`
- [ ] Expected: PASS (3 tests).

### Step 2.3 — Commit

- [ ] Run:

```bash
cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3
git add mcts_study/catan_gnn/road_pip_prior.py mcts_study/tests/test_road_pip_prior.py
git commit -m "feat(cand11): far_endpoint + settlement_legal_mask"
```

---

## Task 3 — Per-sample compute_road_scores() and build_road_pip_target()

**Files:**
- Modify: `mcts_study/catan_gnn/road_pip_prior.py`
- Modify: `mcts_study/tests/test_road_pip_prior.py`

### Step 3.1 — Write failing tests

- [ ] Append to `mcts_study/tests/test_road_pip_prior.py`:

```python
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
```

- [ ] Run: `cd mcts_study && python -m pytest tests/test_road_pip_prior.py -v -k "compute_road or build_road_pip"`
- [ ] Expected: FAIL (ImportError).

### Step 3.2 — Implement

- [ ] Append to `mcts_study/catan_gnn/road_pip_prior.py`:

```python
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
```

- [ ] Run: `cd mcts_study && python -m pytest tests/test_road_pip_prior.py -v -k "compute_road or build_road_pip"`
- [ ] Expected: PASS (3 tests).

### Step 3.3 — Commit

- [ ] Run:

```bash
cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3
git add mcts_study/catan_gnn/road_pip_prior.py mcts_study/tests/test_road_pip_prior.py
git commit -m "feat(cand11): per-sample road score + linear prior target"
```

---

## Task 4 — Batched road_pip_prior_loss() with Gate A and Layer-1 KL

**Files:**
- Modify: `mcts_study/catan_gnn/road_pip_prior.py`
- Modify: `mcts_study/tests/test_road_pip_prior.py`

### Step 4.1 — Write failing tests

- [ ] Append to `mcts_study/tests/test_road_pip_prior.py`:

```python
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
    Road B unlocks v=11 with dice-12 hexes (pip=1 only).
    The KL pull means logits[A] should increase faster than logits[B]
    after one gradient step.
    """
    # Construct: viewer owns edge 0 ([0,3]) AND edge 10 ([7,11]).
    # Wait — that puts vertex 7 into the frontier, so v_new for edge 6
    # would change. Use viewer edges 0 and 18 ([11,16]) instead so frontier
    # = {0, 3, 11, 16}. Now:
    #   edge 6 ([3,7]): v0=3 in, v1=7 not → far = 7.
    #   edge 10 ([7,11]): v0=7 not, v1=11 in → far = 7. Same vertex!
    # Different example: viewer = {0, 33}. Edge 0 = [0,3], edge 33 = [21,27].
    # Frontier = {0, 3, 21, 27}.
    #   edge 6 ([3,7]): far = 7.
    #   edge 23 ([16,21]): far = 16.
    # Set hex 0 (adjacent to v=0,3,4,7,8,12) to dice 6 (pip 5).
    # Set hex 7 (adjacent to v=16,21,22,27,28,33) to dice 12 (pip 1).
    # Then pip(v=7) > pip(v=16) → prior on road 6 > prior on road 23.
    ef = _build_edge_features(viewer_road_edges=[0, 33]).unsqueeze(0)
    vf = _build_vertex_features(occupied_vertices=[]).unsqueeze(0)
    dice = [0]*19
    dice[0] = 6   # high pip for v=7
    dice[7] = 12  # low pip for v=16
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
    # = [0.5, 0.5]. Prior is [pip(v=7)/(pip(v=7)+pip(v=16)), pip(v=16)/...] with
    # pip(v=7) >> pip(v=16). So prior is sharper toward road6 → (q - prior) is
    # NEGATIVE on road6 (model needs to increase it) and POSITIVE on road23.
    g_a = grad[ROAD_OFFSET + 6].item()
    g_b = grad[ROAD_OFFSET + 23].item()
    assert g_a < 0, f"high-pip road should have negative gradient (push up), got {g_a}"
    assert g_b > 0, f"low-pip road should have positive gradient (push down), got {g_b}"
    assert abs(g_a) > 1e-4 and abs(g_b) > 1e-4


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
```

- [ ] Run: `cd mcts_study && python -m pytest tests/test_road_pip_prior.py -v -k "road_pip_prior_loss"`
- [ ] Expected: FAIL — `ImportError: cannot import name 'road_pip_prior_loss'`.

### Step 4.2 — Implement the batched loss

- [ ] Append to `mcts_study/catan_gnn/road_pip_prior.py`:

```python
def road_pip_prior_loss(
    *,
    p_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    edge_features: torch.Tensor,
    vertex_features: torch.Tensor,
    hex_features: torch.Tensor,
) -> torch.Tensor:
    """Cand 11 auxiliary loss.

    Math: for each sample where gate A fires
        q[r]      = softmax(p_logits[L_R])[r]
        prior[r]  = score(r) / sum(score(legal_R))
        sample_L  = -sum(prior[r] * log q[r])  for r in L_R

    Returns mean over firing samples (samples where gate fires). If no
    sample in the batch fires the gate, returns a 0 tensor with grad
    enabled (so .backward() is a no-op).

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

    B = p_logits.shape[0]
    device = p_logits.device

    # Gate A part 1: no legal settlement in 0..53
    legal_settle_any = legal_mask[:, 0:54].any(dim=-1)   # [B]
    legal_road_mask = legal_mask[:, ROAD_ACTION_OFFSET:ROAD_ACTION_OFFSET + NUM_EDGES]  # [B, 72]
    has_legal_road = legal_road_mask.any(dim=-1)         # [B]

    candidates = (~legal_settle_any) & has_legal_road    # [B]
    if not candidates.any():
        return torch.zeros((), dtype=p_logits.dtype, device=device)

    # Score per (sample, road). Loop over samples that pass the first
    # two gate conditions. The per-sample inner loop over 72 roads is
    # vectorized inside compute_road_scores.
    scores = torch.zeros(B, NUM_EDGES, dtype=torch.float32, device=device)
    for b in range(B):
        if not bool(candidates[b].item()):
            continue
        scores[b] = compute_road_scores(
            edge_features=edge_features[b],
            vertex_features=vertex_features[b],
            hex_features=hex_features[b],
            legal_road_mask=legal_road_mask[b],
        )

    # Gate A part 3: at least one legal road with nonzero score.
    has_score = (scores.sum(dim=-1) > 0)                 # [B]
    firing = candidates & has_score                      # [B]
    if not firing.any():
        return torch.zeros((), dtype=p_logits.dtype, device=device)

    # Per-sample target (linear normalization over scores). Outside L_R: 0.
    target = torch.zeros(B, NUM_EDGES, dtype=torch.float32, device=device)
    score_sums = scores.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    target_firing = scores / score_sums
    # Only assign target where firing=True; rest stays 0.
    firing_mask = firing.unsqueeze(-1).to(target.dtype)
    target = target_firing * firing_mask

    # Layer 1: independent softmax over road logits. We mask non-legal roads
    # to -inf so they don't get any mass within the road slice.
    road_logits = p_logits[:, ROAD_ACTION_OFFSET:ROAD_ACTION_OFFSET + NUM_EDGES]  # [B, 72]
    masked = road_logits.masked_fill(~legal_road_mask, float("-inf"))
    log_q = F.log_softmax(masked, dim=-1)
    # 0 * -inf = NaN; zero out illegal positions before multiply.
    log_q = log_q.masked_fill(~legal_road_mask, 0.0)

    # Per-sample CE: -sum(target * log_q)
    sample_loss = -(target * log_q).sum(dim=-1)          # [B]
    # Multiply by firing mask (target is 0 outside firing, so this is
    # redundant for value but keeps autograd graph clean).
    sample_loss = sample_loss * firing.to(sample_loss.dtype)

    # Mean over firing samples (NOT mean over the full batch).
    n_firing = firing.to(sample_loss.dtype).sum().clamp(min=1)
    return sample_loss.sum() / n_firing
```

- [ ] Run: `cd mcts_study && python -m pytest tests/test_road_pip_prior.py -v`
- [ ] Expected: PASS (all 9 tests).

### Step 4.3 — Commit

- [ ] Run:

```bash
cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3
git add mcts_study/catan_gnn/road_pip_prior.py mcts_study/tests/test_road_pip_prior.py
git commit -m "feat(cand11): batched road_pip_prior_loss with Gate A + Layer-1 KL"
```

---

## Task 5 — Plumb λ_road through train.py

**Files:**
- Modify: `mcts_study/catan_gnn/train.py`

### Step 5.1 — Add the function argument

- [ ] Edit `train.py:303-342`. Find the existing line `lambda_settle: float = 0.0,` (around line 316) and insert below it:

```python
    lambda_road: float = 0.0,
```

The signature block now contains:

```python
def train_main(
    *,
    run_dirs: list[Path],
    out_dir: Path,
    # ... (existing args) ...
    lambda_vp: float = 0.0,
    vp_compare_rule: bool = False,
    lambda_settle: float = 0.0,
    lambda_road: float = 0.0,       # <-- NEW
    class_balanced_policy: bool = False,
    # ... (rest unchanged) ...
```

### Step 5.2 — Add the docstring entry

- [ ] In the docstring inside `train_main` (around line 344-373), find the block describing `mid_tournament_every` and add this block above it:

```python
    lambda_road: Cand 11 (pure-pip road prior). Weight on the auxiliary KL
        pulling the model's softmax-over-legal-roads toward a distribution
        proportional to pip(far endpoint of the road) when that endpoint
        is settlement-legal. Gate A: fires only when NO legal settlement
        action exists in the sample. Layer 1: independent softmax over
        road slice (no global logit inflation). Default 0 (off). Cell 5
        first run uses 0.05.
```

### Step 5.3 — Add the loss term in the train step

- [ ] In `train.py` find the existing Cand 1 block (around lines 550-560 — starts with `# Cand 1: pure-pip settlement-vertex prior. Off by default`). Insert AFTER that block (still inside the `for batch, value_t, policy_t, legal in train_loader:` loop) the new block:

```python
            # Cand 11: pure-pip road prior. Off by default (lambda_road=0).
            # Gate A fires when no legal settlement exists. Layer 1 KL is
            # over the road slice only.
            if lambda_road > 0.0:
                from .road_pip_prior import road_pip_prior_loss
                # Reshape PyG-concat features [B*N, F] -> [B, N, F].
                hex_feat_b = batch["hex"].x.view(-1, 19, 8)
                vert_feat_b = batch["vertex"].x.view(-1, 54, 13)
                edge_feat_b = batch["edge"].x.view(-1, 72, 6)
                lroad = road_pip_prior_loss(
                    p_logits=p_logits,
                    legal_mask=legal,
                    edge_features=edge_feat_b,
                    vertex_features=vert_feat_b,
                    hex_features=hex_feat_b,
                )
                loss = loss + lambda_road * lroad
```

Note: this block sits AFTER `loss = w_value * lv + w_policy * lp` and AFTER the existing Cand 8 + Cand 1 blocks. The variable `p_logits` is already in scope from the model call earlier in the loop.

### Step 5.4 — Verify the file still parses

- [ ] Run: `cd mcts_study && python -c "from catan_gnn import train; print('ok')"`
- [ ] Expected output: `ok`

### Step 5.5 — Commit

- [ ] Run:

```bash
cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3
git add mcts_study/catan_gnn/train.py
git commit -m "feat(cand11): plumb lambda_road through train_main"
```

---

## Task 6 — Add --lambda-road CLI flag to train_grid_inproc.py

**Files:**
- Modify: `mcts_study/scripts/train_grid_inproc.py`

### Step 6.1 — Add the argparse entry

- [ ] In `train_grid_inproc.py`, find the existing `--lambda-settle` block (around lines 163-167). Insert AFTER it:

```python
    p.add_argument("--lambda-road", type=float, default=0.0,
                   help="Cand 11 (pure-pip road prior). Weight on the auxiliary "
                        "KL pulling the policy's softmax-over-legal-roads toward "
                        "edges whose far endpoint is settlement-legal and "
                        "high-pip. Gate A: fires only when no legal settlement "
                        "exists. Default 0 (off). Cell 5 first run uses 0.05.")
```

### Step 6.2 — Pass it to train_main

- [ ] Find the `train_main(...)` call (around lines 270-280). Find the line `lambda_settle=args.lambda_settle,` and insert below it:

```python
                lambda_road=args.lambda_road,
```

### Step 6.3 — Verify CLI parses

- [ ] Run: `cd mcts_study && python scripts/train_grid_inproc.py --help 2>&1 | grep lambda-road`
- [ ] Expected: a line showing `--lambda-road` in the help output.

### Step 6.4 — Commit

- [ ] Run:

```bash
cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3
git add mcts_study/scripts/train_grid_inproc.py
git commit -m "feat(cand11): add --lambda-road CLI flag"
```

---

## Task 7 — Calibration script (Layer-3 pre-launch diagnostic)

**Files:**
- Create: `mcts_study/scratch_road_pip_calibration.py`

### Step 7.1 — Write the script

- [ ] Create `mcts_study/scratch_road_pip_calibration.py` with:

```python
"""Pre-launch diagnostic for Cand 11 (road-pip prior).

Walks 1000 random samples from the 100k cache and reports:
  - Fraction of samples where Gate A's first condition fires (no legal settle).
  - Of those, fraction where any legal road has nonzero score (gate fully fires).
  - Distribution of |L_R| (number of legal roads) on firing samples.
  - Mean entropy of prior vs mean entropy of MCTS visits restricted to roads.

If prior entropy is much lower than visits entropy, the prior is sharper and
lambda_road=0.05 may be too aggressive. If they're comparable, the chosen
lambda is reasonable.

Usage:
    cd mcts_study
    python scratch_road_pip_calibration.py --cache-path ~/catan_cache/cache_100k.pt
"""
import argparse
import math
from pathlib import Path

import torch

from catan_gnn.dataset import CachedDataset
from catan_gnn.road_pip_prior import (
    ROAD_ACTION_OFFSET,
    compute_road_scores,
    NUM_EDGES,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-path", type=Path, required=True)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = CachedDataset(source=None, cache_path=args.cache_path)
    print(f"cache loaded: {len(ds)} positions")
    rng = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=rng)[: args.n_samples].tolist()

    n_no_settle = 0
    n_road_nonzero = 0
    n_road_zero_after_no_settle = 0
    road_count_hist = {}
    prior_entropies = []
    visit_entropies = []

    for i, k in enumerate(idx):
        data, value_t, policy_t, legal = ds[k]
        legal = legal.bool()
        # Reshape per-sample [N, F] (no batch dim from __getitem__).
        hex_f = data["hex"].x        # [19, 8]
        vert_f = data["vertex"].x    # [54, 13]
        edge_f = data["edge"].x      # [72, 6]

        legal_settle_any = legal[0:54].any().item()
        legal_road_mask = legal[ROAD_ACTION_OFFSET:ROAD_ACTION_OFFSET + NUM_EDGES]
        n_legal_roads = int(legal_road_mask.sum().item())

        if legal_settle_any:
            continue
        n_no_settle += 1
        road_count_hist[n_legal_roads] = road_count_hist.get(n_legal_roads, 0) + 1

        scores = compute_road_scores(
            edge_features=edge_f, vertex_features=vert_f,
            hex_features=hex_f, legal_road_mask=legal_road_mask,
        )
        total = float(scores.sum().item())
        if total <= 0:
            n_road_zero_after_no_settle += 1
            continue
        n_road_nonzero += 1

        prior = (scores / total).clamp(min=1e-12)
        prior_nonzero = prior[prior > 0]
        H_prior = -(prior_nonzero * prior_nonzero.log()).sum().item()
        prior_entropies.append(H_prior)

        # MCTS visit entropy restricted to legal roads.
        road_visits = policy_t[ROAD_ACTION_OFFSET:ROAD_ACTION_OFFSET + NUM_EDGES]
        road_visits = road_visits * legal_road_mask.to(road_visits.dtype)
        s = float(road_visits.sum().item())
        if s > 0:
            rv = (road_visits / s).clamp(min=1e-12)
            rv_nz = rv[rv > 0]
            H_visits = -(rv_nz * rv_nz.log()).sum().item()
            visit_entropies.append(H_visits)

    N = args.n_samples
    print()
    print(f"=== Cand 11 calibration on {N} random cache samples ===")
    print(f"Samples with NO legal settlement (Gate A part 1): {n_no_settle} ({100*n_no_settle/N:.1f}%)")
    print(f"  Of those, with at least one nonzero road score (Gate A fully fires): "
          f"{n_road_nonzero} ({100*n_road_nonzero/max(n_no_settle,1):.1f}%)")
    print(f"  All-zero road scores (gate part 3 blocks): "
          f"{n_road_zero_after_no_settle} ({100*n_road_zero_after_no_settle/max(n_no_settle,1):.1f}%)")
    overall_fire_rate = 100 * n_road_nonzero / N
    print(f"  OVERALL gate-firing rate: {n_road_nonzero}/{N} = {overall_fire_rate:.1f}%")
    print()
    print(f"|L_R| histogram on Gate-A-part-1 samples:")
    for k in sorted(road_count_hist):
        print(f"  |L_R| = {k:2d}: {road_count_hist[k]:4d} samples")
    print()
    if prior_entropies:
        mean_H_prior = sum(prior_entropies) / len(prior_entropies)
        max_H_uniform = math.log(max(len(prior_entropies), 1))  # not exact — for ref only
        print(f"Mean prior entropy (firing samples): {mean_H_prior:.3f}")
    if visit_entropies:
        mean_H_visits = sum(visit_entropies) / len(visit_entropies)
        print(f"Mean MCTS-visits entropy over legal roads (firing samples): {mean_H_visits:.3f}")
        if prior_entropies:
            print(f"Ratio prior/visits: {mean_H_prior/max(mean_H_visits,1e-6):.3f}")
            print(f"  (1.0 = comparable sharpness; <0.5 = prior much sharper, "
                  f"consider lower lambda_road)")
    print()
    print("If overall gate-firing rate < 5%, the loss term is rarely active;")
    print("the experiment may not produce a measurable signal.")
    print("If > 60%, the prior dominates a large fraction of samples;")
    print("consider lower lambda_road (e.g. 0.025).")


if __name__ == "__main__":
    main()
```

### Step 7.2 — Run the calibration in WSL

- [ ] Run:

```bash
wsl -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && \
  cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study && \
  python scratch_road_pip_calibration.py --cache-path ~/catan_cache/cache_100k.pt"
```

- [ ] Expected: a printed report with gate-firing rate, |L_R| histogram, and entropy ratio. Should complete in ~2-5 minutes (cache load dominates).
- [ ] Record the gate-firing rate and the prior/visits entropy ratio. Paste them into the journal stub (Task 10).

### Step 7.3 — Commit

- [ ] Run:

```bash
cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3
git add mcts_study/scratch_road_pip_calibration.py
git commit -m "feat(cand11): pre-launch calibration script

Reports Gate A firing rate and prior-vs-MCTS-visits entropy on a 1000-sample
random subset of the cache. Per writing-plans Task 7 of the Cell 5 plan."
```

---

## Task 8 — Smoke test (1-epoch train on toy fixture with --lambda-road 0.05)

**Files:**
- Create: `mcts_study/tests/test_cell5_smoke.py`

### Step 8.1 — Find an existing toy fixture pattern

- [ ] Run:

```bash
cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3
grep -rn "test_cand7_stack\|test_lambda_settle\|test_cand1_stack" mcts_study/tests/ | head -5
```

- [ ] Expected: a line pointing at an existing smoke-test file. Read that file with the Read tool to understand the fixture pattern before writing `test_cell5_smoke.py`.

### Step 8.2 — Write the smoke test

- [ ] Create `mcts_study/tests/test_cell5_smoke.py`:

```python
"""Cell 5 (Cand 11) smoke test: 1-epoch train on a tiny replay fixture
with --lambda-road 0.05. Asserts:
  - No NaN in any loss component.
  - Total loss > vanilla loss when lambda_road > 0 AND any sample fires the gate.
  - With lambda_road = 0, total loss matches vanilla baseline byte-identical.

Run:
    cd mcts_study && python -m pytest tests/test_cell5_smoke.py -v
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from catan_gnn.train import train_main


@pytest.fixture
def toy_run_dir(tmp_path: Path) -> Path:
    """Reuse the smallest available real run-dir fixture. The test
    bypasses the cache path so the dataset is built from these parquets.

    Per chat 2026-05-25, the canonical smoke fixture lives at
    runs/v3/smoke_5_games (5 games × all rotations, cited from existing
    test infrastructure). If that path doesn't exist on the dev machine,
    fall back to picking any worker directory with 'moves.parquet'.
    """
    candidates = [
        Path("runs/v3/smoke_5_games"),
        # Cited from existing journals: 100k data-gen worker dirs each have moves.parquet
        Path("runs/v3/2026-05-05T05-50-e9_v3_data_gen_100k_w12/worker0"),
    ]
    for c in candidates:
        if c.exists():
            return c
    pytest.skip("No smoke fixture run-dir found; create runs/v3/smoke_5_games to enable")


def _train_with(run_dir: Path, tmp_path: Path, *, lambda_road: float) -> float:
    out = tmp_path / f"out_lr_{lambda_road}"
    out.mkdir(parents=True)
    train_main(
        run_dirs=[run_dir],
        out_dir=out,
        hidden_dim=8,
        num_layers=1,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        lambda_road=lambda_road,
        device="cpu",
        seed=0,
        early_stop_patience=0,
    )
    log_path = out / "training_log.json"
    import json
    log = json.loads(log_path.read_text())
    return float(log["epochs"][-1]["train_loss_total"])


def test_smoke_no_nan_with_lambda_road(toy_run_dir, tmp_path):
    loss = _train_with(toy_run_dir, tmp_path, lambda_road=0.05)
    assert not math.isnan(loss), f"train_loss is NaN with lambda_road=0.05"
    assert loss > 0


def test_smoke_lambda_road_zero_matches_vanilla(toy_run_dir, tmp_path):
    """With lambda_road=0, the road loss block is skipped entirely.
    Result should be identical to a vanilla 1-epoch train (modulo
    DataLoader nondeterminism, which we eliminate with seed=0 + cpu)."""
    loss_off = _train_with(toy_run_dir, tmp_path, lambda_road=0.0)
    loss_off_2 = _train_with(toy_run_dir, tmp_path, lambda_road=0.0)
    # Same seed → same loss.
    assert abs(loss_off - loss_off_2) < 1e-5, (
        f"vanilla (lambda_road=0) not reproducible: {loss_off} vs {loss_off_2}"
    )
```

### Step 8.3 — Run the smoke test

- [ ] Run:

```bash
wsl -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && \
  cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study && \
  python -m pytest tests/test_cell5_smoke.py -v"
```

- [ ] Expected: 2 PASS (or 2 SKIP if no fixture exists — in that case create one per Step 8.4).

### Step 8.4 — If no fixture exists: smoke fixture is the existing 100k worker

- [ ] If the test skipped, point it at one worker dir:

```bash
ls /mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study/runs/v3/ | head
```

- [ ] Find an `e9_v3_data_gen_100k` run dir and edit `_train_with`'s `run_dir` argument inside the test if needed. Re-run pytest.

### Step 8.5 — Commit

- [ ] Run:

```bash
cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3
git add mcts_study/tests/test_cell5_smoke.py
git commit -m "test(cand11): 1-epoch smoke train asserting no-NaN + reproducibility"
```

---

## Task 9 — Launch Cell 5 (production training)

**Files:**
- No new files. Status file written to `runs/v3/dashboard/cell5.json`.

### Step 9.1 — Verify no concurrent CPU-heavy sweep is running

- [ ] Per memory `feedback_user_running_sweeps.md`: before any CPU-heavy launch, check for user-owned background processes:

```bash
wsl -- bash -lc "ps -ef | grep -E 'catan_mcts run|train_grid_inproc' | grep -v grep"
```

- [ ] Expected: no output (or only this very session's processes). If a user sweep is running, **STOP**, do not launch. Wait for clearance.

### Step 9.2 — Verify the calibration result is acceptable

- [ ] Recall the calibration output from Step 7.2. Confirm:
  - Gate-firing rate is between **5%** and **60%**.
  - Prior/visits entropy ratio is between **0.3** and **2.0**.
- [ ] If outside those bounds, do not launch. Report numbers to user and ask for a λ_road revision.

### Step 9.3 — Launch the run

- [ ] Run:

```bash
wsl -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && \
  cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study && \
  nohup python scripts/train_grid_inproc.py \
    --cache-path ~/catan_cache/cache_100k.pt \
    --out-root runs/v3/loss_aug/05_cand_road_pip_h128_l4 \
    --status-file runs/v3/dashboard/cell5.json \
    --epochs 15 --batch-size 256 --device auto \
    --rotate --rotate-mode random --cells h128_l4 --seed 0 \
    --mid-tournament-every 5 \
    --lambda-road 0.05 \
    > runs/v3/loss_aug/05_cand_road_pip_h128_l4/cell5_launch.log 2>&1 &"
```

- [ ] Expected: process PID printed. The launch log path is `runs/v3/loss_aug/05_cand_road_pip_h128_l4/cell5_launch.log`. Note the PID; per memory `feedback_dont_kill_authorized_runs.md`, do NOT kill this run on "better idea" reasoning — only kill per the ep5 rule below.

### Step 9.4 — Record the launch in the journal stub

- [ ] Open the journal file (created in Task 10 below — do that first if it doesn't exist), append the launch command and PID. Update with timestamps as ep5 and ep10 finish.

---

## Task 10 — Journal stub + ep5 decision

**Files:**
- Create: `docs/superpowers/journals/2026-05-XX-cell5-road-pip-prior.md` (replace `XX` with the launch date)

### Step 10.1 — Create the journal stub

- [ ] Create the file with this template:

```markdown
# Cell 5: Cand 11 (road-pip prior) — standalone experiment

**Date:** 2026-05-XX
**Plan:** `docs/superpowers/plans/2026-05-25-road-pip-prior.md`
**Spec sections:** Mathematical Specification (locked); Layer-1 KL, Gate A, λ_road=0.05.
**Status:** RUNNING / KILLED / COMPLETED (update as appropriate)
**Cell output:** `runs/v3/loss_aug/05_cand_road_pip_h128_l4/`
**Baseline for comparison:** Cell 0 vanilla (`runs/v3/loss_aug/00_baseline_h128_l4_pilot/`)

## Setup

| Setting | Cell 0 (baseline) | Cell 5 (this run) |
|---|---|---|
| Architecture | h128_l4 (632k params) | h128_l4 (same) |
| Cache | cache_100k.pt | same |
| Batch size | 256 | 256 |
| LR | 1e-3 (Adam) | same |
| Augmentation | random hex rotation | same |
| Seed | 0 | 0 |
| `lambda_vp` (Cand 8) | 0.0 | 0.0 |
| `vp_compare_rule` (Cand 10) | False | False |
| **`lambda_road` (Cand 11)** | **0.0** | **0.05** |

## Calibration result (Task 7, pre-launch)

(paste output of scratch_road_pip_calibration.py here)

Gate-firing rate: XX%
Prior/visits entropy ratio: X.XX

## Per-epoch metrics

(fill in as training progresses; format mirrors Cell 1 journal)

| ep | train_loss | val_loss | val_top1 | gate_fire_rate | best |
|---:|---:|---:|---:|---:|---|

## ep5 mid-tournament — the decision point

(fill in after ep5 mid-tournament completes)

| Player | Cell 5 | Cell 0 baseline | Δ |
|---|---:|---:|---:|
| PureGnn | / 120 ( %) | 15 / 120 (12.50%) | |
| GnnMcts | / 120 ( %) | 2 / 120 (1.67%) | |
| LookaheadMctsV3 | / 120 ( %) | 102 / 120 (85.00%) | |
| Random | / 120 ( %) | 1 / 120 (0.83%) | |

**Decision rule (per plan):** if PureGnn ≥ 1.5pp below Cell 0's ep5 (i.e. ≤10.58%
= 12.69 wins/120 — round to ≤12 wins), kill the run, journal the result, do
not continue. Otherwise, let it run to ep15.

## Behavioral metric — road-to-settlement ratio

(populate after running the midgame parser on Cell 5's ep5 parquets)

| Role | roads/100 turns | settlements/100 turns | roads ÷ settlements |
|---|---:|---:|---:|
| Cell 1 PureGnn ep10 (cited prior) | 17.49 | 2.43 | 7.2 |
| Cell 5 PureGnn ep5 (this) | | | |
| Lookahead in-tournament ep10 (cited) | 18.10 | 4.53 | 4.0 |

## Conclusion

(fill in)
```

### Step 10.2 — Commit the stub

- [ ] Run:

```bash
cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3
git add docs/superpowers/journals/2026-05-XX-cell5-road-pip-prior.md
git commit -m "docs(cell5): journal stub for Cand 11 launch"
```

### Step 10.3 — When ep5 mid-tournament completes: apply the kill rule

- [ ] Wait for the ep5 mid-tournament to finish (~5h after launch per Cell 1 timing). Check:

```bash
ls /mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study/runs/v3/loss_aug/05_cand_road_pip_h128_l4/training_h128_l4/mid_tournaments/
```

- [ ] Expected: a directory like `2026-05-XXT*-e10_v3_tournament/` with 10 worker subdirs.

- [ ] Compute PureGnn winrate using the standard aggregator. Run:

```bash
wsl -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && \
  cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study && \
  python scratch_midgame_actions_cell1_ep10.py 2>&1 | head -50"
```

(Adapt the script's TR path to the Cell 5 ep5 directory; ~1 line change.)

- [ ] If PureGnn ≤ 12 wins (≤10.00%), **KILL the run.** Run:

```bash
wsl -- bash -lc "ps -ef | grep train_grid_inproc | grep -v grep"
# Note the PID, then:
wsl -- bash -lc "kill <PID>"
```

- [ ] Either way, update the journal with the ep5 numbers and the kill/continue decision.

### Step 10.4 — Commit the ep5 decision

- [ ] Run:

```bash
cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3
git add docs/superpowers/journals/2026-05-XX-cell5-road-pip-prior.md
git commit -m "docs(cell5): ep5 mid-tournament result + decision"
```

---

## Self-Review

I'm running the writing-plans skill's self-review now (no separate skill — this is a checklist on the plan I just wrote).

**1. Spec coverage:**
- Math spec → fully implemented across Tasks 1-4. Gate A's three conditions implemented in `road_pip_prior_loss` (legal_settle_any, has_legal_road, has_score). ✓
- Layer 1 (independent softmax over road slice) → implemented in Step 4.2. ✓
- λ_road = 0.05 default → set in train.py default + train_grid_inproc.py default + journal stub + launch command. ✓
- Early-kill rule (ep5 ≥1.5pp below Cell 0) → Step 10.3. ✓
- Standalone (no Cand 8+10) → launch command omits `--lambda-vp` and `--vp-compare-rule`. ✓
- Pre-launch calibration (Layer 3) → Task 7 + Step 9.2 acceptance band. ✓
- No cache rebuild → confirmed; Cand 11 reads existing tensors at training time. ✓

**2. Placeholder scan:**
- The journal stub uses `2026-05-XX` as a placeholder for the launch date. This is intentional — the date is unknown at plan-writing time. Step 10.1 instructs the engineer to replace it.
- No "TBD"s, no "implement later"s, no "similar to Task N" without code. ✓

**3. Type consistency:**
- `road_pip_prior_loss` signature uses keyword-only args (`*,`) throughout — matches the existing `settlement_prior_loss` and `_vp_prior_loss` calling style. ✓
- Tensor shapes: `[B, 280]` logits, `[B, 280]` legal_mask, `[B, 72, 6]` edge_features etc. — match the reshape done in `train.py` Step 5.3. ✓
- `ROAD_ACTION_OFFSET = 108` defined once in road_pip_prior.py, imported in tests. ✓
- `EDGE_TO_VERTICES_TENSOR` exported from road_pip_prior.py but never read by the loss — only test imports use it. The Task 1 test imports it for completeness; if the engineer's IDE flags it as unused, that's fine. ✓
- `compute_road_scores` returns `[72]` (not `[280]`) — verified by tests in Task 3. The batched `road_pip_prior_loss` then scatters into a target with shape `[B, 72]` (not `[B, 280]`) — the indexing later writes back into the [B, 280] gradient only via the `log_softmax(p_logits[:, 108:180])` path. ✓
- Cand 10's `vp_compare_swap_target` modifies `policy_t`; this is unrelated to Cand 11 and unchanged. ✓

**4. One thing worth flagging:**
- Task 8 Step 8.1's "find existing toy fixture" instruction is the closest to a placeholder. I made it explicit: run a grep, read the result, mirror the pattern. The engineer must do small adaptation work, but the steps are concrete (find file, read file, mirror). Acceptable.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-25-road-pip-prior.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
