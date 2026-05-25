# Cand 11 Vectorization — Performance Improvement Spec

**Date:** 2026-05-25
**Status:** Improvement suggestion, NOT yet implemented
**Trigger:** Cell 5 launch (PID 569, 2026-05-25) ran 4h+ without completing
epoch 1, vs Cell 1 baseline of ~63 min/epoch. Root cause analysis points
to the per-sample Python loop in `compute_road_scores`.

## 1. Problem

`mcts_study/catan_gnn/road_pip_prior.py::road_pip_prior_loss` calls
`compute_road_scores` once per sample in a Python for-loop:

```python
# road_pip_prior_loss, current implementation
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
```

`compute_road_scores` itself contains two more Python loops:

```python
# compute_road_scores, current implementation
for e in range(NUM_EDGES):           # 72 iters per sample
    if not bool(legal_road_mask[e].item()):
        continue
    v_new = far_endpoint(edge_id=e, edge_features=edge_features)
    if v_new < 0:
        continue
    if not bool(settle_legal[v_new].item()):
        continue
    out[e] = vertex_pip[v_new]
```

And `settlement_legal_mask` adds another loop:

```python
# settlement_legal_mask, current implementation
for v in range(NUM_VERTICES):        # 54 iters per sample
    if not empty[v].item():
        continue
    nbrs = VERTEX_NEIGHBORS[v].to(vertex_features.device)
    if bool(empty[nbrs].all().item()):
        out[v] = True
```

Each `.item()` call forces a CUDA-CPU sync (or for CPU tensors, a
Python-level scalar extract). Per training batch (B=256):
- Outer loop: 256 sample iterations
- Per sample: 54 vertex iterations + 72 edge iterations
- Each iteration: 1-3 `.item()` calls

**Per-batch cost: ~256 × 130 × 2 ≈ 67,000 `.item()` calls.** On
CPU-tensor path this is ~10-100 µs each due to Python overhead, so
~7-67 ms per batch added on top of whatever the GNN forward+backward
costs. The unit tests don't surface this — they use B=1 or B=3 and
finish in milliseconds total.

**Observed impact:** 4× slowdown of full training cycle vs Cell 1
(vanilla) on the same h128_l4 / batch_size=256 config.

## 2. Fix: vectorize all three functions across the batch dimension

### 2a. `settlement_legal_mask` — batched

```python
def settlement_legal_mask_batched(vertex_features: torch.Tensor) -> torch.Tensor:
    """Batched version.

    Args:
        vertex_features: shape [B, 54, 13] OR [54, 13] (single sample).

    Returns:
        Bool tensor of shape [B, 54] (or [54] for single sample).
    """
    # empty: shape [..., 54]
    empty = (vertex_features[..., 0] >= 0.5)

    # Build a [54, max_nbrs] index tensor at module load time. Pad with -1
    # for vertices with fewer than max_nbrs neighbors. Pre-existing
    # VERTEX_NEIGHBORS is a list[Tensor]; we densify once at module init.
    # max_nbrs on the standard board is 3 (interior vertices); coastal
    # vertices have 2.
    #
    # PAD with -1 means "self-loop to a fake always-empty vertex." We
    # achieve this by appending one extra column of "always True" to empty:
    #   empty_padded: [..., 55] where empty_padded[..., 54] = True always.

    empty_padded = torch.cat(
        [empty, torch.ones_like(empty[..., :1])], dim=-1
    )  # [..., 55]

    # NBRS_PADDED is [54, max_nbrs=3] long, with -1 replaced by 54 (the
    # always-True sentinel). Precomputed at module load.
    # Gather: nbr_empty = empty_padded[..., NBRS_PADDED]
    # Shape: [..., 54, max_nbrs]
    nbr_empty = empty_padded[..., NBRS_PADDED]   # broadcasts over batch
    # All neighbors must be empty (sentinel is always True, so coastal
    # vertices with only 2 real neighbors still get True from the sentinel).
    all_nbrs_empty = nbr_empty.all(dim=-1)        # [..., 54]
    return empty & all_nbrs_empty                 # [..., 54]
```

Module-level setup (one-time cost):

```python
def _build_nbrs_padded() -> torch.Tensor:
    """Dense [54, 3] long tensor. Vertices with fewer than 3 real neighbors
    get the sentinel index 54 (which `empty_padded` makes always-True)."""
    max_nbrs = max(len(n) for n in VERTEX_NEIGHBORS)
    assert max_nbrs == 3, f"expected max 3 neighbors, got {max_nbrs}"
    pad = torch.full((NUM_VERTICES, max_nbrs), 54, dtype=torch.long)
    for v, nbrs in enumerate(VERTEX_NEIGHBORS):
        for i, u in enumerate(nbrs.tolist()):
            pad[v, i] = u
    return pad

NBRS_PADDED: torch.Tensor = _build_nbrs_padded()  # [54, 3]
```

### 2b. `far_endpoint` — batched

```python
def far_endpoint_batched(edge_features: torch.Tensor) -> torch.Tensor:
    """Batched version. Returns far-endpoint vertex per (sample, edge), or
    -1 if both/neither endpoint is in the viewer's frontier.

    Args:
        edge_features: shape [B, 72, 6].

    Returns:
        Long tensor of shape [B, 72]. -1 means "no clear far endpoint."
    """
    # viewer_owns: [B, 72] bool
    viewer_owns = (edge_features[..., 2] >= 0.5)

    # Frontier vertices per sample: gather endpoints of owned edges.
    # EDGE_TO_VERTICES_TENSOR is [72, 2] long. We need a per-sample
    # [54] bool frontier mask.
    # For each (b, e), if viewer_owns[b, e]: mark EDGE_TO_VERTICES_TENSOR[e]
    # as frontier in sample b.
    #
    # Vectorized: build a [B, 54] frontier mask via scatter-or.
    B = edge_features.shape[0]
    device = edge_features.device

    # Expand edge endpoints to [B, 72, 2] and the viewer_owns mask to gate them.
    ep = EDGE_TO_VERTICES_TENSOR.to(device).unsqueeze(0).expand(B, -1, -1)  # [B, 72, 2]
    owned_expand = viewer_owns.unsqueeze(-1)  # [B, 72, 1]
    # Set to -1 (invalid scatter index) where not owned, so they don't
    # contribute. We'll use scatter_add into a [B, 55] buffer and check >0.
    # Simpler: flatten and use index_put_ via boolean mask.
    frontier = torch.zeros(B, NUM_VERTICES, dtype=torch.bool, device=device)
    # For each sample, mark endpoints of owned edges as frontier.
    # We can use a sparse approach: flatten (B, 72) → (B*72), pick owned ones.
    flat_owned = viewer_owns.view(-1)                                     # [B*72]
    sample_idx = torch.arange(B, device=device).repeat_interleave(72)     # [B*72]
    edge_idx = torch.arange(72, device=device).repeat(B)                  # [B*72]
    ep_flat = EDGE_TO_VERTICES_TENSOR.to(device)[edge_idx]                # [B*72, 2]

    owned_samples = sample_idx[flat_owned]
    owned_eps = ep_flat[flat_owned]                                       # [N, 2]
    # Mark both endpoints
    frontier[owned_samples, owned_eps[:, 0]] = True
    frontier[owned_samples, owned_eps[:, 1]] = True

    # Now for each candidate (sample, edge), check whether v0 or v1 in frontier.
    v0 = ep[..., 0]                                                       # [B, 72]
    v1 = ep[..., 1]                                                       # [B, 72]
    v0_in = frontier.gather(1, v0)                                        # [B, 72]
    v1_in = frontier.gather(1, v1)                                        # [B, 72]

    # far = v1 if v0_in and not v1_in; v0 if v1_in and not v0_in; -1 otherwise.
    far = torch.full_like(v0, -1)
    far = torch.where(v0_in & ~v1_in, v1, far)
    far = torch.where(v1_in & ~v0_in, v0, far)
    return far                                                            # [B, 72]
```

### 2c. `compute_road_scores` — batched

```python
def compute_road_scores_batched(
    *,
    edge_features: torch.Tensor,      # [B, 72, 6]
    vertex_features: torch.Tensor,    # [B, 54, 13]
    hex_features: torch.Tensor,       # [B, 19, 8]
    legal_road_mask: torch.Tensor,    # [B, 72] bool
) -> torch.Tensor:
    """Batched version. Returns [B, 72] float."""
    # Pip per vertex per sample.
    hex_pip = hex_features_to_pip(hex_features)             # [B, 19]
    vertex_pip = compute_vertex_score(hex_pip)              # [B, 54]

    # Settlement-legal mask per (sample, vertex).
    settle_legal = settlement_legal_mask_batched(vertex_features)  # [B, 54]

    # Far endpoint per (sample, edge).
    far = far_endpoint_batched(edge_features)               # [B, 72]

    # Score per (sample, edge):
    #   score = vertex_pip[b, far[b, e]] if far >= 0 AND settle_legal[b, far[b, e]] AND legal_road_mask[b, e]
    #         else 0
    #
    # Use a sentinel: clamp far to >=0 (negative → 0 as a safe index),
    # gather, then mask out via the validity mask.
    far_safe = far.clamp(min=0)                             # [B, 72]
    gathered_pip = vertex_pip.gather(1, far_safe)            # [B, 72]
    gathered_legal = settle_legal.gather(1, far_safe)        # [B, 72]

    valid = (far >= 0) & gathered_legal & legal_road_mask    # [B, 72]
    return torch.where(valid, gathered_pip, torch.zeros_like(gathered_pip))
```

### 2d. `road_pip_prior_loss` — strip the outer for-loop

```python
def road_pip_prior_loss(*, p_logits, legal_mask, edge_features,
                        vertex_features, hex_features) -> torch.Tensor:
    """As before, but compute_road_scores is batched."""
    if legal_mask.dtype != torch.bool:
        legal_mask = legal_mask.bool()

    legal_settle_any = legal_mask[:, 0:54].any(dim=-1)
    legal_road_mask = legal_mask[:, ROAD_ACTION_OFFSET:ROAD_ACTION_OFFSET + NUM_EDGES]
    has_legal_road = legal_road_mask.any(dim=-1)

    candidates = (~legal_settle_any) & has_legal_road
    if not candidates.any():
        return torch.zeros((), dtype=p_logits.dtype, device=p_logits.device)

    # ONE call, all samples — no Python loop.
    scores = compute_road_scores_batched(
        edge_features=edge_features,
        vertex_features=vertex_features,
        hex_features=hex_features,
        legal_road_mask=legal_road_mask,
    )                                                       # [B, 72]
    # Zero out scores for non-candidate samples
    scores = scores * candidates.unsqueeze(-1).to(scores.dtype)

    # Rest of the function unchanged from current impl.
    has_score = (scores.sum(dim=-1) > 0)
    firing = candidates & has_score
    if not firing.any():
        return torch.zeros((), dtype=p_logits.dtype, device=p_logits.device)

    score_sums = scores.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    target_firing = scores / score_sums
    firing_mask = firing.unsqueeze(-1).to(target_firing.dtype)
    target = target_firing * firing_mask

    road_logits = p_logits[:, ROAD_ACTION_OFFSET:ROAD_ACTION_OFFSET + NUM_EDGES]
    masked = road_logits.masked_fill(~legal_road_mask, float("-inf"))
    log_q = F.log_softmax(masked, dim=-1)
    log_q = log_q.masked_fill(~legal_road_mask, 0.0)

    sample_loss = -(target * log_q).sum(dim=-1)
    sample_loss = sample_loss * firing.to(sample_loss.dtype)
    n_firing = firing.to(sample_loss.dtype).sum().clamp(min=1)
    return sample_loss.sum() / n_firing
```

## 3. Equivalence guarantees

The vectorized versions must produce **bit-identical results** to the
loop versions on the existing 12 unit tests in
`mcts_study/tests/test_road_pip_prior.py`. Specifically:

- All 5 `road_pip_prior_loss` tests (Gate A behavior, gradient
  isolation to road slice, prior favors high-pip road, mean-over-firing-
  samples) must pass byte-identically.
- All 3 `compute_road_scores` / `build_road_pip_target` tests must pass.
- All 3 topology + far_endpoint + settlement_legal_mask tests must pass.

**Validation plan (TDD):**

1. Keep the existing loop functions in the module under new names
   `compute_road_scores_loop`, `settlement_legal_mask_loop`, etc.
2. Add the batched versions alongside.
3. Add a new test `test_batched_matches_loop` that runs 100 random
   samples through both and asserts byte-identical outputs.
4. Switch `road_pip_prior_loss` to call the batched versions only after
   the equivalence test passes.
5. Delete the loop versions in a follow-up commit after one full epoch
   of training confirms no behavioral drift.

## 4. Performance expectation

The vectorized version replaces ~67k Python-level `.item()` calls per
batch with ~10 tensor ops (each operating on [B, 72] or [B, 54]
tensors). Expected per-batch speedup:

- Current (loop): unknown but ~4× vanilla per Cell 5 observation
- Vectorized: expected ~1.1-1.3× vanilla (the road-loss tensor ops
  themselves cost something, but should be small relative to the GNN
  forward+backward)

**If achieved:** ep1 drops from ~4h back to ~70 min. 15 epochs in
~17.5h, matching Cell 1's reference timeline.

## 5. Why this wasn't caught pre-launch

The smoke test (`test_cell5_smoke.py`) used:
- 2 games × 1 epoch
- batch_size = 4
- hidden_dim = 8, num_layers = 2

At this scale, ~8 Python iterations per batch × ~4 batches × 1 epoch
= ~32 iterations total. Finished in seconds. The overhead is real
per-batch but invisible at toy-fixture scale.

**The miss is consistent with the new memory entry
`feedback_training_observability.md`:** smoke tests should include a
production-batch-size timing assertion, not just a no-NaN assertion.

**Suggested smoke test addition:**

```python
@pytest.mark.slow
def test_cand11_per_batch_overhead_acceptable(tmp_path, e1_fixture):
    """Cand 11 should add <50% overhead vs vanilla on a single batch
    at production batch size. Fails loud if per-sample overhead leaks."""
    import time
    # ... run 10 batches at batch_size=256 with lambda_road=0 and =0.05
    # Assert: time_with_road / time_without_road < 1.5
```

## 6. Action items (when adopted)

1. Implement batched versions per §2.
2. Add equivalence test per §3.
3. Add timing-assertion smoke test per §5.
4. Re-run calibration (cheap, results should match prior 19.5% / 0.387).
5. Relaunch Cell 5 with vectorized impl.

## 7. Open question

The current Cell 5 run (PID 569) is in an extended ep1. Options:
- **Kill now**, vectorize, relaunch. Cost: ~5h of compute lost, ~2-3h
  for fix + relaunch, then ~17h to full result. Saves ~40h calendar.
- **Let it finish**, vectorize for Cell 6+. Cost: ~60h calendar for
  Cell 5; Cand 11 result lands ~Wednesday.
- **Let it finish but vectorize in parallel**, ready to deploy for any
  follow-up cell.

Decision deferred to user.
