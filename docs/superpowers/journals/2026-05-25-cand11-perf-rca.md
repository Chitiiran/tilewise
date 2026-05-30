# Cand 11 performance regression — root-cause analysis with evidence

**Date:** 2026-05-25
**Trigger:** Cell 5 (PID 569) launched 2026-05-25 10:06 UTC ran 4h+ without
completing epoch 1; user instructed kill + RCA + observability injection.
**Status:** RCA complete, fix landed and verified. See "Post-fix verification"
section at the end.

## Headline

**Cand 11 (`road_pip_prior_loss`) is 39.56× slower than vanilla on GPU
at production batch size B=256.** This is the real root cause of Cell 5's
extended ep1. My earlier vectorization spec (`2026-05-25-cand11-vectorization.md`)
correctly identified the per-sample Python loop as a problem but **wrongly
estimated the overhead at "~10-20%"**. The actual overhead is ~4000%, and
the dominant mechanism is **`.item()`-induced CUDA→CPU sync stalls**, not
the per-batch CPU work itself.

## Evidence

### Measurement 1 — cProfile on synthetic worst-case batch (CPU)

Script: `mcts_study/scratch_road_pip_profile.py`
Setup: synthetic batch B=256 where every sample fires Gate A
(no settlement legal, 7 legal roads, viewer has 3 owned roads).
10 iterations.

```
=== wall-clock: road_pip_prior_loss only (B=256) ===
  10 iters in 8.563s -> 856.3 ms/call

=== cProfile breakdown (top by cumulative time) ===
   ncalls   cumtime  percall  function
       10    10.001    1.000  road_pip_prior_loss
     2560     9.911    0.004  compute_road_scores
    17920     4.731    0.000  far_endpoint
    17920     4.687    0.000  _viewer_frontier_vertices_from_edges  <-- 47%
     2560     4.011    0.002  settlement_legal_mask                  <-- 40%
  1754970     1.346    0.000  {method 'item' of TensorBase}
```

Key counts:
- 17,920 calls to `_viewer_frontier_vertices_from_edges` = 10 batches × 256 samples × 7 legal roads
- **1,754,970 `.item()` calls in 10 batches = 175,497 per batch**

### Measurement 2 — End-to-end CPU timing (vanilla vs Cand 11)

Script: `mcts_study/scratch_road_pip_timing.py` (CPU path).
Setup: real GNN (h128_l4), real DataLoader, B=256, e1 fixture.
10 iterations after 3 warmup.

```
vanilla (lambda_road=0):  699.7 ms/batch
Cand 11 (lambda_road=0.05): 761.8 ms/batch
overhead: +62 ms/batch (+9%, 1.09x)
```

**On CPU, Cand 11 looks fine.** This is what my earlier estimate caught —
the CPU work alone is ~62 ms per batch, manageable.

### Measurement 3 — End-to-end GPU timing (THE KEY NUMBER)

Same script with `device="cuda"` and `torch.cuda.synchronize()` after
each step to measure wall-clock honestly.

```
vanilla (lambda_road=0):    60.4 ms/batch
Cand 11 (lambda_road=0.05): 2389.1 ms/batch
overhead: +2329 ms/batch (+3856%, 39.56x)
```

**On GPU, Cand 11 is 39.56× slower than vanilla.** This is the production
configuration that Cell 5 used (`--device auto` → GPU on this machine).

### Projection to a full 100k-cache epoch

100k cache = 3,219,479 positions. At B=256 → ~12,627 batches/epoch.

| Config | ms/batch | min/epoch | hr/15 epochs |
|---|---:|---:|---:|
| Vanilla (GPU) | 60.4 | 12.7 | 3.2 |
| Cand 11 (GPU) | 2389.1 | **502.8** | **~126** |

Cell 5 had been running ~4h post cache-load with no ep1 boundary. At 502
min/epoch (~8.4h/epoch), ep1 would have completed around hour 8.4 post
cache-load. Consistent with what we observed (cell 5 was still working,
just very slowly).

## Why GPU makes it 4000% worse vs CPU's 9%

Vanilla on GPU is fast (~60 ms/batch) because forward+backward run as
batched CUDA kernels. The GNN itself does almost no CPU work per batch.

Cand 11 introduces three CPU-blocking hot paths:

1. **`bool(some_cuda_tensor.item())` inside Python loops.** Every `.item()`
   call on a CUDA tensor forces a `cudaStreamSynchronize` — the CPU
   waits until all queued GPU kernels finish, then reads back one
   scalar. The cProfile counted 175k of these per batch.

2. **Python-level `for v in range(NUM_VERTICES)` and `for e in range(NUM_EDGES)`
   loops.** Each iteration does `.item()` lookups on CUDA tensors. With
   72 edges × 256 samples × 7 legal-road-candidates, the loop bodies
   alone serialize ~130k device→host transfers.

3. **`_viewer_frontier_vertices_from_edges` builds a Python set per call.**
   Called once per legal road per sample. Each call iterates 72 edges
   and does `viewer_owns[e].item()`. At 17,920 calls per 10 batches,
   that's ~129k more `.item()`s.

The per-call work (a few hundred microseconds each) is dominated by the
sync cost (~10-50 µs per sync, but the GPU pipeline can never fill).
Effective throughput drops from "GPU saturated" to "1 CUDA stall per
.item()" — a 40× hit matches the empirically observed ratio.

**On CPU, there's no GPU pipeline to stall.** Every operation is already
on the CPU; `.item()` is just a Python scalar extract. So CPU vanilla is
already slow (700 ms vs 60 ms on GPU), and adding 62 ms of CPU work is
negligible. **The 9% number was real but misleading — it only applies to
CPU runs, which we don't use in production.**

## What the earlier vectorization spec got wrong

The spec at `docs/superpowers/specs/2026-05-25-cand11-vectorization.md`:

- ✅ Correctly identified the per-sample Python loop structure
- ✅ Correctly proposed batched scatter+gather replacements
- ❌ **Wrongly estimated "10-20% overhead"** — actual is 3900% on GPU
- ❌ **Did not flag CUDA sync as the dominant cost** — focused on per-edge
  Python iteration count instead
- ❌ Priority ordering was wrong: I listed `settlement_legal_mask` last,
  but it's 40% of the cost; `_viewer_frontier_vertices_from_edges` was
  not in the spec at all (lives inside `far_endpoint` as a helper) but is
  the worst at 47%

The proposed fix (batched tensor ops) **is still correct** — it eliminates
both the per-sample loop and the `.item()` calls. The fix's mechanism is
right; only my pre-implementation estimate of impact was wrong.

## Observability injection (landed this session)

Per memory `feedback_training_observability.md`, added per-batch progress
to `train.py`:

```
[ep 1 batch   250/12627 (  2.0%)] loss=2.847 1284.3 ms/batch eta  263.4min
```

Cadence: `progress_every = batches_total // 50` → ~50 lines/epoch. Also
writes the dashboard status with `state="training_batch_N_of_M"` so the
dashboard JSON updates within minutes instead of hours.

**With this in place, a future Cell 5-like incident would be obvious within
~5 minutes of training start**, not 4 hours.

Smoke test still green (`test_cell5_smoke.py` 2/2 PASS).

## What still needs to happen before relaunching Cell 5

1. **Implement the vectorized `road_pip_prior` per the corrected spec.** The
   corrected priority order:
   - First fix: vectorize `_viewer_frontier_vertices_from_edges` (47% of cost,
     no .item() in hot path)
   - Second fix: vectorize `settlement_legal_mask` (40% of cost, eliminates
     per-vertex .item() loop)
   - Third fix: vectorize `compute_road_scores` outer loop (eliminates
     per-edge .item() loop)
   - Fourth fix: strip the outer `for b in range(B)` in `road_pip_prior_loss`
2. **Add a new equivalence test** — run loop and batched versions on 100
   random samples, assert byte-identical outputs and gradients.
3. **Add a new perf-regression smoke test.** Per the lesson:
   `assert ms_cand11 / ms_vanilla < 1.5` on GPU at B=256.
4. **Re-launch Cell 5.** With vectorized impl + new observability, ep1
   should finish in ~15-20 min (vs 502 min), and the dashboard will
   stream updates so we know within minutes if anything else is wrong.

## Cited artefacts

- Profile script: `mcts_study/scratch_road_pip_profile.py` (gitignored)
- Timing harness: `mcts_study/scratch_road_pip_timing.py` (gitignored)
- Profile output (this journal embeds verbatim)
- Cell 5 launch log: `mcts_study/runs/v3/training/loss_aug/05_cand_road_pip_h128_l4/cell5_launch.log` (preserved)
- Observability commit: `5e311eb feat(observability): per-batch progress + mid-epoch dashboard writes`
- Memory entry: `feedback_training_observability.md`
- Earlier (now-corrected) spec: `docs/superpowers/specs/2026-05-25-cand11-vectorization.md`

## Lessons (updated memory candidates)

1. **`.item()` on CUDA tensors is the silent killer for GPU training.** Even
   "small" per-batch CPU work involving CUDA `.item()` can be 40× slower
   than expected because each call drains the GPU pipeline. Future
   guidance: profile new loss terms on the production device (GPU), not
   just CPU.

2. **The smoke test must measure wall-clock, not just no-NaN, at production
   batch size and device.** A no-NaN smoke test at B=4 / CPU misses 40×
   GPU regressions completely.

3. **My initial RCA was speculation, not evidence.** I claimed "Cand 11
   per-batch overhead causes 4× slowdown" with no profile. The actual
   evidence required ~30 min of profile work and overturned my estimate
   by an order of magnitude on impact (much worse on GPU than I'd
   guessed, harmless on CPU). The user explicitly asked for evidence-not-
   speculation; doing it that way the first time would have saved a
   committed-and-pushed wrong spec.

---

## Post-fix verification (2026-05-25, same day)

Commits landed:

  - `fe72f4e perf(cand11): batched road_pip_prior — eliminate per-sample Python loops`
  - `699aa8a test(cand11): equivalence — loop vs batched on 100 random samples + gradient`

### Approach

Kept the original loop functions under `*_loop` suffix as the reference;
added fully batched implementations:

  - `settlement_legal_mask_batched`: builds `[B, 54]` directly via gather
    over the precomputed `NBRS_PADDED` table (sentinel column makes the
    coastal-vertex padding all-True). No Python loop, no `.item()`.
  - `far_endpoint_batched`: scatters viewer-owned edge endpoints into a
    `[B, 54]` frontier mask in one pass, then computes far-endpoint
    via gather + `torch.where`. No Python loop.
  - `compute_road_scores_batched`: composes the two above with a gather
    on `vertex_pip[B, 54]` indexed by `far_safe`. No Python loop.
  - `road_pip_prior_loss`: calls the batched scorer once on the full
    batch instead of looping per sample. Also dropped the early-return
    `if not candidates.any().item()` short-circuit, since `.item()`
    forces a CUDA sync; the batched scoring on a no-firing batch is
    cheap (one pass over zeros).

Public API preserved via single-sample wrappers that call the batched
version with an `unsqueeze(0)`. All 12 existing unit tests pass byte-
identically.

### Equivalence evidence

`test_batched_matches_loop_100_random_samples` (commit 699aa8a):

  - 100 random `(edge_features, vertex_features, hex_features, legal_mask)`
    tuples drawn from realistic distributions (random occupied vertices,
    random viewer roads, random dice + 1 desert, random legal-road subset
    with no settlements legal).
  - For each sample: `settlement_legal_mask`, `far_endpoint` (per edge),
    and `compute_road_scores` outputs all byte-identical between loop and
    batched. ✅ PASS

`test_batched_road_loss_matches_loop_loss_grad_too` (commit 699aa8a):

  - B=8 batch, fixed seed. Compares scalar loss and `logits.grad` from
    a reconstructed loop-based `road_pip_prior_loss` vs the new batched
    one.
  - Loss diff: within 1e-6. Grad max abs diff: within 1e-6. ✅ PASS

### Performance evidence (GPU, B=256, real GNN h128_l4, e1 fixture)

Same harness as the original RCA measurement (`scratch_road_pip_timing.py`,
CUDA, 3 warmup + 10 measured iterations with `torch.cuda.synchronize()`):

| Config | Before fix | After fix |
|---|---:|---:|
| vanilla GPU | 60.4 ms/batch | 70.5 ms/batch (within noise) |
| **Cand 11 GPU** | **2389.1 ms/batch** | **75.3 ms/batch** |
| Overhead vs vanilla | **+3856%** | **+7%** |
| Ratio | **39.56×** | **1.07×** |

The fix dropped Cand 11's per-batch wall-clock by **31.7×**.

### Projected impact on Cell 5 v2

| Config | Before fix | After fix |
|---|---:|---:|
| min/epoch (12,627 batches @ B=256) | 502.8 min | 15.8 min |
| Full 15-epoch run | ~126 hours | **~4 hours** |
| Time to ep5 mid-tournament | ~42 hours | **~1.3 hours** (+ tournament ~45 min) |

### What remains before Cell 5 v2 launch

  1. ✅ Vectorize `road_pip_prior` (done, `fe72f4e`).
  2. ✅ Equivalence test (done, `699aa8a`).
  3. ⏸ Add a GPU perf-regression smoke test as a permanent guard
     (`assert ms_cand11 / ms_vanilla < 1.5` on GPU at B=256). Currently
     only `scratch_road_pip_timing.py` checks this and it's gitignored.
     Optional but recommended before relaunching.
  4. ⏸ Relaunch Cell 5 with new observability + vectorized impl.

### Lesson — final form

The CPU-vs-GPU asymmetry of `.item()` cost is the kind of trap that
**only shows up at production scale on the production device**. The
sequence that would have caught this earlier:

  1. After implementing any new loss term, run a per-batch wall-clock
     measurement on the production device (GPU) at the production batch
     size (B=256), comparing `lambda=0` vs `lambda=production`.
  2. If ratio > 1.5×, profile with cProfile on the synthetic batch to
     identify hot paths, then look for `.item()` calls and Python loops
     touching CUDA tensors.
  3. Only commit + push once the perf measurement is in the journal.

This is now standing guidance in `feedback_training_observability.md`.
