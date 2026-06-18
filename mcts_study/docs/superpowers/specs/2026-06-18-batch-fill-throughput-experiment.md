# Batch-Fill Throughput Experiment — Design Spec

**Date:** 2026-06-18
**Status:** Approved design, ready for implementation planning
**Context:** Follow-up to Task 10 of the Rust-MCTS + TorchScript-GNN rewrite
(`2026-06-17-rust-mcts-torchscript-gnn.md`). The batched self-play engine works
and is reproducible; this spec improves its THROUGHPUT.

---

## 1. Problem (data-grounded)

Phase profiling of `play_games_batched` (production net 128×4, sims=200,
deterministic CUDA) found the bottleneck is **batch under-fill**, NOT the CPU:

| config | mean batch | GPU% | CPU% | leaves/s |
|---|---|---|---|---|
| 16 games, B_MAX=32 | 10.7 / 32 | 96.8% | 3.0% | 1298 |
| 64 games, B_MAX=32 | 21.5 / 32 | 91.7% | 8.0% | 2281 |

The GPU forward is 92–97% of wall-clock, but each call carries a large FIXED
~20ms deterministic-scatter latency that is paid **regardless of batch size**
(see the batch-size sweep: 19.9ms@B=1 → 24.1ms@B=128). So a half-full batch
wastes ~half the GPU. CPU work (tree/expand/advance) is only 3–8% — optimizing
it is pointless. **The lever is mean batch FILL.**

Two structural causes of under-fill, and a lever for each:
- **Too few concurrent games.** With G games and B_MAX=32, mean batch ≤ G when
  G<32, and rises with G. Confirmed: 16g→10.7, 64g→21.5. **Lever A: run more
  concurrent games.**
- **The un-parked fraction.** Even at 64 games, mean batch (21.5) is well below
  the 32 cap. At any forward, ~1/3 of games are NOT parked on a leaf — they are
  inside `advance_to_search` resolving a run of chance / single-legal moves, or
  starting a new move's search. Those games contribute no leaf to the batch.
  **Lever B: keep more games parked — advance the non-parked games concurrently
  so they rejoin the batch sooner.**

## 2. Goal

Raise mean batch fill (→ closer to B_MAX) and thus leaves/sec and games/min,
**without breaking reproducibility** (same seeds + config → byte-identical game
records, every run — the "go back to the match" contract). Quantify each lever
with a controlled experiment.

## 3. The two levers

### Lever A — concurrency (config; confirmed)
Run G ≫ B_MAX games through one batcher. `play_games_batched` already supports
any G; this is a tuning question: what G maximizes leaves/sec without
over-subscribing CPU/VRAM? Pure config — no algorithm change, trivially
reproducible (each game's NpRng is independent of G).

### Lever B — keep games parked (scheduler change)
A game in `advance_to_search` (resolving chance/single-legal runs) is "dark" —
not parked, not contributing a leaf. Today the scheduler advances a finished
game **inline, serially**, between forwards. Lever B advances dark games so they
rejoin the parked set faster, lifting mean fill toward B_MAX.

**The reproducibility constraint (critical):** batch COMPOSITION must be a
deterministic function of (seeds, B_MAX, config) — never of wall-clock timing.
The float reassociation in the batched forward means a leaf's value depends on
which other leaves share its batch; if overlap changes that grouping
nondeterministically, the replay contract breaks. So Lever B MUST preserve a
deterministic parked-set ordering: games are batched in a fixed order (by slot
index), and a game's parked leaf enters the batch at a position determined only
by its slot, not by when its advance happened to finish.

Given that constraint, Lever B is **not** true async overlap (that introduces
timing nondeterminism). It is a **deterministic two-phase wave**: each round,
(1) advance ALL dark games to their next parked leaf (or done), in slot order;
(2) batch ALL currently-parked games' leaves (in slot order, chunked by B_MAX);
(3) feed results back. This already maximizes fill for a given game set — the
current loop is close to this but advances games one-at-a-time as they finish a
move mid-wave, which can leave the next chunk under-filled. The change is to
**collect the full parked set before chunking**, so every chunk except the last
is exactly B_MAX. (Investigate: does the current `active.chunks(b_max)` already
do this? If mean-B<cap is purely a G<effective-parked artifact, Lever B reduces
to A. The experiment will tell.)

## 4. Experiment design

A controlled, reproducible benchmark producing a table, plus a parity check.

### 4.1 Concurrency sweep (Lever A)
For G ∈ {16, 32, 64, 128, 256} at B_MAX=32, sims=200, fixed seed set, CUDA
deterministic: measure mean batch, leaves/sec, games/min, CPU% vs GPU%, peak
VRAM. Find the G that maximizes leaves/sec (the knee — beyond it, CPU/VRAM or
the un-parked fraction caps the gain). Output: a table + recommended
`n_concurrent`.

### 4.2 Scheduler A/B (Lever B)
Implement the "collect-full-parked-set-then-chunk" scheduler
(`play_games_batched_v2`). At the best G from 4.1, compare v1 vs v2: mean batch,
leaves/sec, games/min. Expect v2 mean-B ≥ v1 mean-B.

### 4.3 Reproducibility + parity gate (NON-NEGOTIABLE)
- **Self-consistency:** each path (v1, v2, every G) run TWICE → byte-identical
  game records. The replay contract.
- **v1↔v2 equivalence:** v2 must produce records that are either (a)
  byte-identical to v1 at the same (seeds, B_MAX) IF the batch composition is
  unchanged, or (b) if v2 changes composition, it is STILL a valid reproducible
  engine and its records match the existing Phase-6-style statistics
  (length/winner/value-target distributions) — documented explicitly which.
- A concurrency change (Lever A) must NOT change any single game's record
  beyond the batch-composition float reassociation already accepted in Task 10;
  assert records are identical across G for the SAME batch-composition, or
  document the reassociation if G changes grouping.

### 4.4 Metrics harness
Extend the existing `profile_batched` (already attributes GPU/CPU/advance +
mean batch + leaves) to emit a machine-readable row per (G, scheduler) so the
sweep produces a clean table. Reuse `cpu_profile`-style ignored tests driven by
env vars (TP_GAMES, TP_BMAX, TP_SIMS).

## 5. Out of scope
- Non-deterministic CUDA (2.6× faster forward) — rejected: breaks the replay
  contract the user requires. Noted as a known option only.
- CPU micro-optimization (engine-clone trimming, incremental observation) — only
  3–8% of time; not worth it until batch fill is solved.
- Tree reuse across moves (fewer leaves/game) — a separate, larger optimization;
  out of scope for this experiment.

## 6. Success criteria
- A reproducible sweep table (G × scheduler → mean-B, leaves/s, games/min).
- A recommended production `n_concurrent` (+ B_MAX) with evidence.
- If v2 helps: mean-B and leaves/s measurably higher than v1 at matched G, with
  the reproducibility gate green. If v2 does NOT beat v1 (i.e. the current loop
  already fills optimally and under-fill is purely G-limited), document that
  Lever B is unnecessary and ship Lever A's tuning recommendation alone.

## 7. Verification mandate (carried over)
Reproducibility is non-negotiable (the replay contract). Every scheduler variant
must pass: run-twice byte-identical, and the documented v1/v2 relationship.
Determinism (deterministic CUDA + fixed slot-order batching) is preserved.
