# Pipeline Build — Autonomous Run Summary (for review)

**Date:** 2026-06-18 (autonomous, ~10h delegation)
**Directive:** "use all the learning to make the pipeline the best and maxed-out
for this machine, with a pausable and highly-observable GNN training pipeline.
observability first, then pausable, then max out threads+GPU. n_concurrent can be
512+. run small batches, verify, improve. dashboard must work well."
**Branch:** `az-difficulty-bots` (all pushed). Spec:
`docs/superpowers/specs/2026-06-18-az-pipeline-maxed-pausable-observable.md`.

---

## What was built (all 3 phases done + verified)

### Phase 1 — Observability (done)
- `catan_az/sampler.py`: background GPU/CPU/RAM sampler → `resources.jsonl`
  (nvidia-smi util/power/mem + loadavg + free RAM), SIGTERM-stoppable. Wired into
  `daily.run_cycle` so it runs for every iteration.
- `catan_gnn/train.py`: streaming `train_progress.jsonl` per batch (loss, value,
  policy, grad-norm, ms/batch) — the live loss curve (per the per-batch
  observability HARD RULE).
- `catan_az/analytics.py`: `_tail_jsonl`, `resources_live`, `train_progress_live`.
- `catan_az/dashboard/`: two new endpoints (`/api/resources-live`,
  `/api/train-progress-live`) + two new cards in `index.html` (Machine GPU/CPU
  with util+power sparkline; Training live loss curve) using the existing
  vanilla-JS tick() pattern + an inline-SVG `spark()` chart.
- **Verified:** 12 unit tests; `dash_smoke.py` serves populated live panels
  end-to-end; the micro self-play run actually wrote `resources.jsonl` (32 lines)
  while the GPU was engaged.

### Phase 2 — Pausability (done, all stages)
- **Self-play:** `self_play_rust` processes seeds in `n_concurrent`-sized chunks,
  flushing + marking done per chunk, checking a `PAUSE` sentinel between chunks.
  Resume skips done seeds → byte-identical (per-game RNG). 3 tests.
- **Training:** `PAUSE` checked at the epoch boundary (after `checkpoint.pt` with
  `next_epoch` is written); `resume_from` continues. Per-epoch-seeded DataLoader
  shuffle so order depends only on (seed, epoch). 2 tests. **Honest contract:**
  training is NOT byte-reproducible on GPU (cuDNN/Adam ~1e-6 even un-paused), so
  pause/resume is a VALID continuation (same epochs, matching loss), not
  byte-identical. Game replay is the bit-exact requirement, not training replay.
- **Arena:** `run_arena_games` (explicit (rot,seed) list) lets `_run_arena_rust`
  chunk the plan, write results.jsonl per chunk, check PAUSE between chunks.
  Resume skips done seeds. 3 tests.
- One control surface: a `PAUSE` file in the iter dir / loop root pauses any
  running stage; `STOP` (existing) ends after the current iteration.

### Phase 3 — Max out threads + GPU (done)
- **Architecture settled by measurement, not assumption.** A timed profile
  (CUDA-sync around the bare forward) showed `forward_is` = **84.7%** of
  wall-clock; ALL parallelizable work (marshal+extract+CPU) = ~13.5%. So the
  multi-threaded scheduler would win ≤13.5% — **NOT built** (the "scheduler core
  at 95%" was a red herring: busy ≠ bottleneck). We are genuinely
  GPU-forward-bound.
- **The realized win:** retire the N-process self-play launch (right only for the
  old 1-core asyncio engine) for **ONE high-concurrency batched process** — one
  fat GPU batch ~93% full. `daily._launch_selfplay_procs` forces n_procs=1 for
  the rust engine; `n_concurrent` default 24→**256** (measured knee), max_batch→32.
- **Concurrency data (production net, sims=200, deterministic CUDA):** mean GPU
  batch 10.7 (16 games) → 21.5 (64) → 29.9/32 (512). Throughput: ~16× over the
  old per-leaf path at the production config. Beyond batch fill, the only levers
  are forward COST/COUNT (tree reuse, fp16) — documented follow-ups, NOT built.
- 15 tests green (launch model, config, module switch, one-process assertion).

---

## Key measurements (this run)
- Timed profile: forward 84.7% / marshal 4.3% / extract 1.9% / CPU 7.3%.
- Concurrency → batch fill: 16g=10.7, 64g=21.5, 256g≈mid, 512g=29.9/32 (93%).
- Raw GNN forward ceiling: ~9,400 states/s at B=2048 (plateauing; knee B≈128-256);
  VRAM never the limit (638MB/4GB at B=2048).
- End-to-end games/min: B=32/32g 3.26 → B=64/64g 3.85 (diminishing past B=32).
- Recommended production: **engine=rust, n_concurrent≈256, B_MAX(max_batch)=32**.

## Reproducibility posture (deliberate)
- **Game replay = bit-exact** (self-play + arena reproducible run-to-run via
  deterministic CUDA + fixed batch composition). This is the contract that
  matters ("go back to the match" without storing full games).
- **Training = not byte-reproducible on GPU** (accepted; you don't replay
  training). Non-deterministic CUDA (2.6× forward) stays REJECTED — it would
  break game replay.

## NOT done (need your decision / out of scope)
- Did NOT resume the paused AZ loop or merge to main (standing rules).
- Did NOT build: multi-threaded scheduler (data says ≤13.5%), tree reuse, fp16
  inference — these are the only remaining throughput levers and are
  forward-cost/count changes, deferred follow-ups.

## End-to-end integration smoke (verified this run)
Ran the PRODUCTION self-play path (rust engine, real az_iter_1 128×4 net, CUDA,
chunked/pausable) directly:
```
[self_play_rust] chunk done: 2/2 games
[self_play_rust] done: 2 games -> .../self_play_rust-p46991   exit=0
records: games.*.parquet + moves.*.parquet written; done.txt=2
GPU during run: 38% util / 28.5W (engaged, was idle 3.7W)
sampler: resources.jsonl 125 lines written alongside
```
So the full stack runs together: Rust batched engine + CUDA + chunked self-play +
SelfPlayRecorder parquet + live resource sampler. (Test harness note: a chunk =
n_concurrent games that finish together, so a chunk flushes all-at-once; an
earlier 6-game smoke produced 0 records only because a 3-min sampler kill cut it
off before the all-or-nothing chunk finished — not a bug. For finer flush
granularity in production, set n_concurrent to the desired flush size.)

## Final verification (all green)
- Plain-pytest pipeline sweep: **45 passed, 1 skipped** (GPU arena test gated).
- GPU arena-match (via pytest_mctsrs.sh): **1 passed** (369s) — rust arena on
  CUDA matches the Python arena per-game + winrate, confirming the device-matched
  .ts export fix. (This bug — CPU-traced .ts crashing on CUDA — would have
  silently broken production arena; now fixed + verified.)
- Integration smoke: real self-play produces records, GPU engaged, sampler runs.

## Recommended next step
The infra is done and much faster + correct. The highest-value action now is to
**run a real AZ iteration** on this engine (naturally-terminated self-play →
train → finishes-naturally arena) and watch the value head start learning from
true outcomes on the live dashboard. The throughput work bought the ability to do
this in reasonable time; the science win is in iterations, not more tuning.

## Files changed (high level)
- New: `catan_az/sampler.py`; tests `test_sampler`, `test_observability_analytics`,
  `test_dashboard_live`, `test_selfplay_pause`, `test_train_pause`,
  `test_arena_pause`; scripts (sampler runners, timed/concurrency profilers,
  micro smoke).
- Modified: `catan_az/{daily,arena,config,analytics,resources}.py`,
  `catan_az/dashboard/{server.py,static/index.html}`, `catan_gnn/train.py`,
  `catan_mcts/experiments/self_play_rust.py`, `catan_mcts_rs/src/{python,evaluator,
  selfplay}.rs`.
