# B1 inference-server spike — NO-GO (IPC overhead dominates)

**Date:** 2026-06-13 (autonomous, on idle GPU after iter-1)
**Verdict:** ❌ **NO-GO.** The central-GPU-inference-server + N-CPU-worker
architecture is *worse* than the current independent-process design on this
hardware. Do not build it.

## Measurements (GTX 1650, h128/L4, full-Catan observation)

| config | aggregate evals/s | mean batch | latency |
|---|---:|---:|---|
| baseline (in-proc batch=1, 1 proc) | **36** | 1 | 28 ms/eval |
| baseline × 5 procs (current design) | ~180 | 1 | — |
| **server + 10 CPU clients** | **10** | **2.0** | **p50 237ms / p95 314ms** |

The server is **18× slower** than the 5-proc baseline and **3.6× slower** than
even a single in-proc process.

## Why it fails

The go bar was ≥2× the 5-proc aggregate. It came in at ~1/18×. Root cause:
- **IPC serialization dominates.** Each eval ships a PyG `Data` object over a
  `multiprocessing.Queue` (pickle round-trip) and waits for the reply. That
  round-trip is ~237ms — **8× the 28ms the actual GPU eval takes**. The work
  per eval is too small to amortize cross-process transport.
- **Batches never fill** (mean_batch=2.0 of a 64 cap): clients spend almost all
  their time blocked on the queue round-trip, so few are ever simultaneously
  parked for the batcher to collect. The thing the server exists to do —
  batch — doesn't happen.

This is the mirror image of the in-process `BatchedGnnEvaluator`, which works
(mean_batch≈16 in the arena) precisely because there's no serialization: async
coroutines hand the batcher a Python object by reference.

## What this rules in

The throughput wall is **not** "GPU underused, needs cross-process batching."
It's "per-eval CPU work (state_to_pyg + tree management) is the cost, and the
GPU eval is already cheap (28ms)." So the levers that actually matter:

1. **Vectorize / speed up `state_to_pyg`** — it's CPU and on the hot path of
   every leaf. Cutting it speeds up the in-proc path with zero IPC risk.
2. **Rust-side self-play / MCTS** (the spec's milestone-2 lever) — moves tree
   management + state encoding off the Python/CPU hot path entirely. This is
   the real 10-100× option, justified now that B1 is ruled out.
3. **More independent processes** — the current design scales ~linearly with
   procs until VRAM (≈7 procs on 4GB) or CPU cores (6c/12t) saturate. Cheap,
   already works, no new infra.

For iteration 2 right now: **stay with independent processes** (5-7), which is
the measured-best option on this box. The throughput-engineering budget should
go to state_to_pyg / Rust, not an inference server.

## Artifacts
- `scripts/spike_inference_server.py` (fixed `state._engine.observation()` API;
  kept for the record + re-measurement on future hardware, e.g. a 2nd GPU where
  the IPC math could change).
