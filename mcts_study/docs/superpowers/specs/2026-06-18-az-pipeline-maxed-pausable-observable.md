# AZ Training Pipeline — Maxed-out, Pausable, Observable

**Date:** 2026-06-18
**Status:** Draft (architecture pending the timed-profile measurement)
**Owner:** taking ownership per user directive 2026-06-18.
**Builds on:** the Rust-MCTS + TorchScript rewrite (Phases 0-10, done) and the
existing `catan_az` loop (daily.py, dashboard, journal, ladder, status, STOP).

---

## 0. Goal

Turn the now-correct AZ loop into the **best pipeline this machine can run**:
maximum self-play/arena throughput, **pausable** at any point with near-zero
loss, and **highly observable** while running. Build order (user): **(1)
observability → (2) pausability → (3) max out threads + GPU.** Everything
validated by small-batch experiments — run small, verify, improve.

Machine: 12 cores, 54 GB RAM, GTX 1650 4 GB. Engine: `catan_mcts_rs` (Rust MCTS
+ TorchScript GNN, deterministic CUDA, reproducible). Existing observability +
pause exist in basic form; this hardens them, then maxes throughput.

## 1. Why this order

Observability first because we can't safely tune or pause-test what we can't
see — we need live mean-batch, games/min, GPU util/power, CPU per-core, training
loss/grad-norm BEFORE changing the launch model, so every throughput change is
measured, not guessed (the lesson from chasing the moving bottleneck). Pause
second so long max-throughput runs are safe to interrupt. Max-out last, on top of
instrumentation that proves each change helps.

---

## 2. Phase 1 — Observability (build first)

**What exists:** `dashboard/server.py` + `index.html`, `status.py`
(StatusWriter → status.json), `journal.csv`, `progress.py`. Stage-level, polled.

**Gap:** no live INTRA-stage metrics. While a stage runs (self-play for ~hours,
training for ~minutes, arena for ~an hour) you can't see games/min, mean batch
fill, GPU util/power, per-core CPU, or training loss/grad-norm streaming.

**Build:**
- **Metrics emit from the Rust engine.** `run_selfplay`/`run_arena` periodically
  write a small `progress.jsonl` line (games done, leaves/s, mean batch, elapsed)
  — cheap, append-only, crash-safe. (The profiler already computes these; expose
  them on the production path behind a flag.)
- **GPU/host sampler.** A lightweight background sampler (nvidia-smi util/power/
  mem + /proc per-core CPU + RSS) appends to `resources.jsonl` every ~2s during a
  stage.
- **Training metrics.** `catan_gnn.train` emits per-batch (or every ~30-60s) loss
  / lr / grad-norm / throughput to `train_progress.jsonl` (the
  `feedback_training_observability` HARD RULE: per-batch, not per-epoch).
- **Dashboard surfaces them live.** Extend the existing dashboard to read the new
  jsonl streams and show: current stage + ETA; self-play games/min, mean-batch,
  GPU util/power; training loss curve + grad-norm; arena winrate-so-far + CI +
  timeout/draw rates. (SSE or short-poll — reuse the existing server.)
- **Verify (small batch):** run a tiny self-play (8 games) + 1 train epoch on a
  toy cache + a 8-game arena; confirm every metric streams and the dashboard
  renders them. This is the acceptance test for Phase 1.

## 3. Phase 2 — Pausability (build second)

**What exists:** STOP sentinel halts BETWEEN iterations; resume via
done-markers + manifest + per-game parquet flush (loses ≤1 game).

**Gap:** can't pause cleanly MID-stage (mid self-play wave, mid training, mid
arena) and resume exactly.

**Build (pause = fast, safe, any point; resume = continue ~where left off):**
- **Self-play:** a `PAUSE` sentinel checked between batched waves; on pause,
  finish the in-flight forward, flush completed games (already per-game), write a
  resume cursor (which seeds are done). Resume skips done seeds (exists). Target:
  pause within one wave (~seconds), lose ≤ in-flight games.
- **Training:** checkpoint every N steps (not just per-epoch) + write an
  optimizer/scheduler/epoch/step resume file; on PAUSE finish the current step,
  checkpoint, exit. Resume loads it and continues mid-epoch.
- **Arena:** already resumable per-game via results.jsonl; add a PAUSE check
  between games. Resume skips played (rot,seed).
- **One control surface:** PAUSE (graceful, resumable), STOP (end after current
  iteration), and a status field showing paused/running. Reuse the sentinel-file
  pattern (works across the WSL/subprocess boundary).
- **Verify:** start each stage, PAUSE mid-stage, confirm clean exit + a resume
  cursor; resume and confirm it continues and the final records are
  byte-identical to an un-paused run (reproducibility under pause — critical).

## 4. Phase 3 — Max out threads + GPU (build last)

**What exists:** `daily._launch_selfplay_procs` launches **N self-play
PROCESSES** — the right model for the OLD 1-core asyncio engine, the WRONG model
for the Rust batched engine (N processes = N small GPU batchers fragmenting the
4 GB card vs one fat batch).

**Architecture decision — SETTLED by the timed profile (2026-06-18).** Splitting
the GPU phase with a CUDA-sync around the bare forward (G=64, B_MAX=32, sims=50):
`forward_is`=84.7%, marshal=4.3%, extract=1.9%, CPU provide+advance=7.3%. **The
bare GPU forward is 84.7% of wall-clock; ALL parallelizable work is ~13.5%.** So
the multi-threaded scheduler (candidate A) can win AT MOST ~13.5% — NOT worth the
complexity or the reproducibility risk. **Decision: candidate B — we are
genuinely GPU-forward-bound; do NOT build the threaded scheduler.** The "single
core at 95%" was a red herring (busy ≠ bottleneck).

**So "max out GPU" reduces to two things, both already in hand + one new lever:**
1. **Fill the batch (DONE):** concurrency G≥256 → mean batch ~30/32 (93%). Adopt
   it as the production launch model: collapse the N-process self-play (right for
   the old asyncio engine) into ONE high-concurrency batched process. This is the
   single biggest realized win and a config/launch-model change, reproducible.
2. **Tune B_MAX + n_concurrent to the knee** (data: B_MAX≈32-64, n_concurrent
   ≥256; raw GNN keeps scaling but end-to-end flattens). Pick by measured
   leaves/s on the dashboard. Small B_MAX bump (32→64) is +18% e2e for 2× VRAM.
3. **The only lever beyond batch fill is forward COST/COUNT** (documented, NOT
   built in this spec — they are follow-ups):
   - tree reuse across moves → fewer leaves/game → fewer forwards (biggest honest
     remaining win; ~same GPU per forward but far fewer of them).
   - cheaper net: fp16 inference / smaller hidden dim (must re-validate strength).
   - non-deterministic CUDA (2.6× forward) stays REJECTED — breaks replay.

**This phase's concrete deliverable:** swap the self-play launch model to ONE
batched process at tuned n_concurrent/B_MAX (Rust engine), with a short sweep to
pick the values, A/B'd against the current N-process model on leaves/s + games/min
+ GPU util, reproducibility intact. Training/arena tuning (DataLoader workers,
arena n_concurrent) as smaller tune-ups. Tree reuse + fp16 are explicitly
deferred follow-ups, not this phase.

- **Constraint:** every change keeps the reproducibility/replay contract.

## 5. Out of scope
- Net architecture changes (separate concern).
- Non-deterministic CUDA (breaks replay).
- Multi-GPU / multi-machine.

## 6. Success criteria
- **Observability:** every stage streams live metrics to the dashboard; you can
  watch games/min, mean-batch, GPU util, training loss, arena winrate in real time.
- **Pausability:** PAUSE at any point exits cleanly within seconds, loses ≤
  in-flight work, and resume produces byte-identical final records vs un-paused.
- **Throughput:** measured leaves/s (and games/min) materially above the current
  single-scheduler ceiling (~2,525 leaves/s), with the GPU better utilized,
  reproducibility intact. A recommended production config (n_concurrent, B_MAX,
  process/thread model) backed by the sweep.

## 7. Method (standing)
Run small batches, verify, improve — never a big run on an unmeasured change.
Each phase has a small-batch acceptance test before it's trusted. Bottleneck may
move again; re-measure after each change and stop when it moves to something that
doesn't matter (the lesson from this session).
