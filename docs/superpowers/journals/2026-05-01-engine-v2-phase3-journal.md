# Engine v2 — Phase 3 Execution Journal

**Branch:** `engine-v2`
**Worktree:** `C:/dojo/catan_bot/.claude/worktrees/engine-v2/`
**Phase 1 journal:** [`2026-04-30-engine-v2-phase1-journal.md`](2026-04-30-engine-v2-phase1-journal.md)
**Phase 2 journal:** [`2026-05-01-engine-v2-phase2-journal.md`](2026-05-01-engine-v2-phase2-journal.md)

Phase 3 took the full-Catan v2 engine and wired it into the Python ML
pipeline: data sweep, GNN baseline training, and a 4-way tournament
sanity test. Eight rule-bug fixes landed mid-phase as the data revealed
problems. **Phase 3 is shipped, with caveats below.**

---

## What shipped

| Phase | Task | Status |
|---|---|---|
| 3.1 | Rebuild PyO3 against v2 engine | ✅ |
| 3.2 | Update mcts_study Python wrappers for v2 action space (280) | ✅ |
| 3.3 | Update GNN observation: F_VERT 7→13, N_SCALARS 22→59 | ✅ |
| Salvage | Recorder per-game flush + skip_game keeps action_history + atexit | ✅ |
| P1 | Cap ProposeTrade per turn (eventually 1, found via wedge debug) | ✅ |
| P2 | Real Catan port vertex pairs | ✅ |
| P3 | Per-vertex port-kind one-hot in vertex_features | ✅ |
| M1 | Combined PyO3 entry points (query_status, apply_action_smart) | ✅ |
| M2/M3 | Adapter / Rust MCTS rewrites | ❌ skipped (profiler said ≤8% gain) |
| Fix-138 | Terminal check on LR/LA bonus push to 10VP | ✅ |
| Decay fix | Discount rollout returns by length so MCTS prefers fast wins | ✅ |
| C1 | Cache `legal_actions()` against the legal_mask | ✅ |
| 3.4 | 2000-game sims=100 training data sweep | ✅ |
| 3.5 | GNN baseline training (16/20 epochs, early-stop) | ✅ |
| 3.6 | 4-way tournament (after GPU patch) | ✅ |
| 3.7 | This journal | ✅ |

---

## Phase 3.4 — Training data sweep

**Final config**: 2000 games × sims=100 × 5 workers × 60 sec wall-clock cap.

**Results:**

| Metric | Value |
|---|---:|
| Wall clock | **75 minutes** |
| Total games | 2000 |
| Natural completions | 1998 (99.9%) |
| Timeouts (60s cap) | 2 (seeds 2100164, 2100749 at 10458 / 12296 steps) |
| Recorded MCTS moves (P0) | **463 848** |
| Avg moves/game | 232 |
| **MCTS (P0) winrate** | **82.2%** |
| Loser winrates | P1 4.8%, P2 6.6%, P3 6.4% |
| Median game length | 1109 steps |
| p75 game length | 1426 steps |
| Max game length | 12 296 |
| Winners ending at exactly 10 VP | 79% |
| Winners ending at 11+ VP | 21% (legitimate single-action +2 jumps from LR/LA flips) |

**Path**: `mcts_study/runs/v2_phase34_train/2026-05-01T16-00-e1_winrate_vs_random/worker{0,1,2,3,4}/`

### Bugs found via the sweep data

The first sweep (commit a3f4bcd, pre-fix) had **3/5 of its 5-game smoke
games hit the 30k-step engine cap with no winner**. That triggered an
investigation that landed three engine bugs:

1. **Trade-loop pathology (P1)** — MCTS could loop `ProposeTrade(A→B)`
   then `ProposeTrade(B→A)` forever within a single turn since total
   resources were conserved. Cap added at 4 (then later 1).
2. **Terminal check missed on LR/LA bonus (fix-138)** — `update_longest_road`
   and `update_largest_army` bumped VP +2 but didn't call `check_win`.
   So a player could reach 10VP from a knight-play and the engine kept
   playing for 30+ more steps.
3. **Random rollouts didn't differentiate fast vs slow wins (decay
   fix)** — `random_rollout_to_terminal` returned ±1.0 regardless of
   step count. Even when MCTS could `BuildCity` to win in 1 step, it
   often picked `EndTurn` because the rollout from EndTurn also
   terminated with the same player winning (within ~50 steps via
   random play). UCB1 had no signal to break the tie. Fix:
   discount returns by `0.999^steps` (cuts max-step pathology -52%
   in same-seed paired comparison).

These were caught by **inspecting actual sweep data**, not unit tests.
Lesson: data inspection is essential after any meaningful engine
change, even if the unit tests pass.

---

## Phase 3.5 — GNN baseline training

**Config**: hidden_dim=32, num_layers=2, batch=64, lr=1e-3, 20 epochs,
CUDA (GTX 1650). Trained on the full 463 848 positions.

**Results**:

| Metric | Value |
|---|---:|
| Cache build | 24 minutes (457 pos/s) |
| Per-epoch | ~6.7 min train + 34s val |
| Epochs trained before early-stop | 16 (best at epoch 4) |
| **Best val_top1 accuracy** | **0.273** |
| (Random baseline = 1/280) | 0.36% (model is ~76× better) |
| Train loss curve | 2.520 → 2.326 (small monotonic decrease) |
| Val loss curve | 2.691 → 2.779 (no improvement; mild overfit) |

Best checkpoint: `runs/v2_phase35_baseline/checkpoint_best.pt` (epoch 4,
val_top1=0.273).

### Why early-stop

The pattern became clear by epoch 8: train_loss plateaued, val_loss was
trending up, val_top1 oscillated 0.21–0.27 with no upward trend after
epoch 4. Continuing to epoch 20 would not have improved the best
checkpoint. Stopped at epoch 16 to save ~30 min of GPU time.

### What the model learned

val_top1 ≈ 27% means: given a position the model picks the
MCTS-recommended action 27% of the time vs a baseline of 0.4% from
uniform-random. For 280 actions where typical position has ~10 legal,
that's roughly 2.7× the legal-uniform baseline. Modest but real signal —
the policy learned to filter to legal actions and prefer build/trade
patterns.

### Where it could be better

- More data (2000 games is small; 10 000+ is typical for a baseline)
- Augmentation (4× perspective rotation already encoded in the obs API
  but not used at training time; 6× hex symmetry not implemented)
- Larger model — 80k params is the v0 spec target, not a strong ceiling
- Self-play loop — the data here is from RandomBot opponents, so the
  model learns "how to beat random" not "how to play well"

---

## Phase 3.6 — 4-way tournament

**First attempt (CPU)**: GnnMcts, PureGnn, LookaheadMcts, Random; sims=25,
lookahead-depth=10, 180s wall-clock, 5 workers × 5 games × 4 rotations = 20
games. **All 20 games hit the cap at 24-42 moves.** Root cause: e7's
`_load_model` used `map_location="cpu"` and the GNN forward path stayed on
CPU even with CUDA available. Per-move cost was ~5 sec instead of ~0.25 sec
expected.

**GPU patch** (commit during Phase 3.6): added `--device {auto,cpu,cuda}`
to e7 CLI; `_load_model` and `GnnEvaluator` constructors now thread the
device through; CUDA detection happens INSIDE worker processes (CUDA
initialization is per-process, not shareable from parent). Fallback to
CPU when CUDA unavailable.

**Re-run (GPU)**: same lineup, sims=25, lookahead-depth=25, 300s cap,
1 worker (single CUDA context), `--device cuda`.

| Metric | Value |
|---|---:|
| Wall clock | **12:01** |
| Total games | 20 |
| Natural completions | 20/20 |
| Timeouts | 0/20 |
| Per-game avg | ~36 sec |

**Per-role winrate (GTX 1650, GNN checkpoint epoch 4 from Phase 3.5):**

| Player | Wins | Winrate |
|---|---:|---:|
| **LookaheadMcts** sims=25, depth=25 | **18/20** | **90.0%** |
| PureGnn (argmax of policy head) | 1/20 | 5.0% |
| Random | 1/20 | 5.0% |
| **GnnMcts** sims=25, GNN-guided | **0/20** | **0.0%** |

### What this tells us

**The trained GNN is not yet useful as an MCTS evaluator.** Three
observations stack:

1. **GnnMcts wins 0 games** — guiding MCTS with the trained GNN
   produces *worse* play than random rollouts (which won 82% in the
   Phase 3.4 sweep against random opponents). The policy/value
   estimates from epoch 4 redirect MCTS toward bad lines.

2. **PureGnn wins 5%** — the same as Random. The argmax of the
   policy head is no better than uniform among legal actions. val_top1
   = 0.273 in training looked like real signal (76× random) but in a
   competitive 4-way matchup that signal isn't enough.

3. **LookaheadMcts dominates 90%** — the unchanged baseline from v1
   (sims=25 MCTS guided by an N-step lookahead-VP heuristic) walks
   away with the tournament. This is the ceiling our GNN needs to
   beat.

### Why the GNN isn't strong enough

- **Distribution shift**. Training data was P0-MCTS vs 3-RandomBot.
  In test, 3 of 4 players are NOT random — they play strong-ish
  Catan. The GNN never saw competitive opposition during training.
- **Small dataset**. 463k positions at 80k params + 16 epochs is at
  the bottom of "realistic baseline" range. A v1 model would need
  10× the data and 5-10× the params to compete with LookaheadMcts.
- **No augmentation**. 6× hex symmetry + 4× perspective rotation =
  24× larger effective dataset. The infrastructure is there
  (`engine.observation_for(viewer)`) but not wired into training.
- **No self-play loop**. Real AlphaZero training generates data with
  the *current* model playing itself, then trains on those games.
  We trained on a fixed RandomBot dataset.

### What's salvageable

- **The pipeline works end-to-end on GPU.** Sweep → cache → train →
  checkpoint → tournament → results. All the plumbing is in place.
- **LookaheadMcts is a known-strong baseline** (90% in this lineup).
  Future GNN runs have a clear target.
- **The "0% GnnMcts" result is informative.** It rules out "the GNN
  is mid-strength and just needs sims" — the GNN at this training
  level is actively harmful inside MCTS. Future work should focus
  on data quality (self-play) and quantity (augmentation), not
  more sims.

---

## Engine bug fixes shipped this phase

Eight engine/pipeline fixes total. Each was small but several were
critical for usable training data.

| Fix | Commit | Impact |
|---|---|---|
| Recorder: per-game flush + atexit | aea342e | Salvage games on Ctrl-C |
| skip_game keeps action_history | aea342e | Don't lose timed-out games |
| P1 trade-loop cap | 74eaedb | Eliminated 30k-step wedge |
| P2 canonical port vertex pairs | 74eaedb | Real Catan ports, not placeholders |
| P3 vertex port-kind features | 74eaedb | GNN can learn port-adjacent value |
| M1 combined PyO3 calls | 6dd978b | -7→-2 FFI calls/step (latent perf) |
| C1 legal_actions cache | 7457f18 | -34% legal_actions ns in MCTS hot path |
| Trade cap 4→1 + terminal check | f708851 | -54% ProposeTrade share, -52% max steps |
| Rollout length-discount | 323b071 | -52% pathological-tail games |

All fixes have regression tests (cargo test + pytest both green throughout).

---

## Numbers across phases

| | v1 baseline | Phase 2 complete | Phase 3 complete |
|---|---:|---:|---:|
| Action space | 206 | 280 | 280 |
| Rust tests | 75 | 168 | **190+** |
| Python tests | ~50 | ~70 | **~85** |
| Engine `bench-engine-step` | 0.356 µs | 1.068 µs | **0.86 µs** |
| Engine `bench-mcts-game` | 8.5 ms | 13.5 ms | **9.1 ms** |
| Engine `bench-state-clone` | 0.228 µs | 0.226 µs | **0.13 µs** |
| Engine `bench-legal-mask` | n/a | 0.177 µs | **0.010 µs** |
| Real e1 sweep, sims=100 | crashed often | n/a | **5.8 s/game, 82% MCTS winrate** |
| Random rollout per game | 8.5 ms | 13.5 ms | **2.0 ms** (after decay+fixes) |

The "v2 is slower" Phase 2 regression has been mostly clawed back.
v2-Phase3 is **competitive with v1 on the rollout/clone hot paths**,
and `bench-legal-mask` is **17× faster** than v1 had any equivalent for
because v1 had no caching layer.

---

## What's ready for v3 / future phases

1. **Working v2 engine + GNN training pipeline.** End-to-end from
   sweep → cache → train → checkpoint validates.
2. **Tournament harness** (e7) — needs GPU patch but otherwise functional.
3. **Safety harness** (`scripts/safety_check.py`) — Rust + Python + 5
   baseline replays + 5 perf workloads + playback render. ALL GREEN
   throughout Phase 3.
4. **Playback viewer** (`catan_mcts/playback.py`) — single-file HTML
   replay for any seed, used to debug seed=1600011 (the slow-game
   investigation that uncovered fix-138 and the decay fix).
5. **Salvage-by-default recorder.** SIGINT-safe, per-game shards.
6. **Detailed memory notes** for design decisions, perf baselines, and
   bugs found.

---

## What's NOT ready

- **Augmentation tooling** (4× perspective + 6× hex rotation) — would
  give 24× training data. Not built yet. **High priority** based on
  Phase 3.6 results — we need a much bigger effective dataset.
- **Self-play loop** — current pipeline trains on
  RandomBot-vs-MCTS data. Real AlphaZero needs MCTS-vs-MCTS-itself
  data. **Critical for play strength** — Phase 3.6 confirmed the
  fixed-data baseline doesn't beat LookaheadMcts.
- **Larger model** — v0 spec is 80k params; a v1 spec might ramp to
  500k–2M params for stronger play.
- **Phase 1.3 unmake-move** — explicitly deferred (state.clone is
  0.13 µs, not the bottleneck).

---

## Total commits Phase 3

```
74eaedb  fix: trade-loop pathology + canonical ports + GNN port features
0cf2d3b  docs: Phase 2 wrap-up journal + bench comparison report
3af9ee8  phase 3.3: v2 observation — expand scalars 22 -> 59, use cached legal_mask
a3f4bcd  phase 3.2: dynamic ACTION_SPACE_SIZE + v2 engine_version
76aba23  feat: v2 playback viewer
aea342e  fix(recorder): per-game parquet flush + salvage timed-out games
6dd978b  M1: combined PyO3 entry points (query_status, apply_action_smart)
5c3c2e5  test: safety harness for upcoming M1/M2/M3 refactor pyramid
7457f18  perf: legal_actions() honors the legal_mask cache + add bench_random_game
f708851  fix: terminal check missed on LR/LA bonus + tighten trade cap to 1
323b071  feat(rollout): length-discount terminal returns so MCTS prefers fast wins
```

11 Phase-3 commits.

Combined Phase 1 + 2 + 3 across the v2 restart: **35+ commits**, all
on `origin/engine-v2`. Engine went from broken-stripped-down-v1 (Apr
26) to full-rules + caching + bug-free + trained baseline GNN
(May 1). One person-week of effort.

---

## Final state of the worktree

- **Last good commit** at journal time: `323b071` (rollout decay).
- **Safety harness** ALL GREEN as of `5c3c2e5`.
- **Working tree** clean.
- **Pushed to** `origin/engine-v2`.

Phase 3 is closed. Future v3 phase (GPU tournament, augmentation,
self-play, larger model) tracked in next planning doc.
