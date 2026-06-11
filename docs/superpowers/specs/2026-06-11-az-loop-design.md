# Catan AZ Loop — "grow the game, grow the bot" — design

**Date:** 2026-06-11
**Status:** approved (user, 2026-06-11 planning session)
**Branch:** `az-difficulty-bots` (PR #3)
**Supersedes:** the one-shot distillation direction (plan 2026-06-10-az-difficulty-bots.md Tasks 3–5); Tasks 0–2 of that plan (difficulty presets, data-gen) are retained — the running data-gen IS iteration 1.

## 1. North star & framing (user decisions, verbatim intent)

- **End goal: an unbeatable Catan bot.** Beating LookV3 is a milestone, not the goal.
- **Faithful AlphaZero loop as the engine of progress**, warm-started from Cell 6
  (AlphaGo-Master-style: canonical machinery, pragmatic seed). User accepts the
  behavior-cloned seed.
- **Full Catan rules (vp=10, bonuses=on) throughout.** No 5-VP sprint: the
  tournament-inversion result showed simplified rules actively mis-train.
- **The engine grows toward real Catan while the models grow** — small fidelity
  increments per ~few iterations so self-play can *learn* strategies (real
  trading, strategic discards) instead of us scripting them. Keep state small;
  introduce fidelity gradually.
- **Weaker checkpoints stay playable** in the web difficulty ladder.
- **Verify throughput feasibility and project time-to-strength from measured
  data**, not folklore.

## 2. The loop (core artifact)

New package `mcts_study/catan_az/`:

```
loop.py        orchestrator: run_iteration(cfg) + run_forever(cfg)
config.py      AzConfig dataclass (all knobs, JSON round-trip)
buffer.py      sliding-window replay buffer over parquet run dirs
arena.py       candidate-vs-champion gate (async, shared seeds)
ladder.py      Elo ladder + champion registry (CHAMPION.txt + elo.json)
status.py      status.json writer (dashboard-consumable, per-stage heartbeat)
```

Iteration N:

| Stage | What | Source of truth |
|---|---|---|
| SELF-PLAY | champion net + AsyncMcts (sims=200, Dirichlet α=0.8 ε=0.25, τ=1 first 30 moves → 0), full Catan, `games_per_iter` games across `n_procs` worker processes (architecture per §3) | existing `self_play_async.py`, parameterized |
| BUFFER | window = last `window_games` games across iterations (canonical-AZ sliding window; fixes the all-history deviation) | `buffer.py` manifest over run dirs |
| TRAIN | continue champion on window: lr=2e-4, epochs≤4, early-stop on val loss (May-31 lesson: warm-started nets overfit by ep2-3); `--policy-sharpen 1.0` (canonical visit targets — sharpening stays available as a flagged experiment, default off) | existing `train_main` |
| ARENA | GnnMcts(candidate) vs GnnMcts(champion), 2+2 seats, 4 rotations × 30 games = 120, shared seeds, <5% timeout rule; **promote if candidate >55%** (AGZ-style gate — protects the warm seed from regression) | `arena.py` (async harness, e10e machinery) |
| PUBLISH | promoted champion → `checkpoints/az_iter_<N>.pt` + Elo update + web ladder entry; every ~5 iterations: 60-game LookV3 anchor match for absolute calibration | `ladder.py` |
| JOURNAL | append one row: iter, games, positions, train/val loss, arena result, Elo, anchor result, wall-clock per stage | `runs/v3/az_loop/journal.csv` + status.json |

**Resumability:** every stage idempotent + checkpointed (self-play via done.txt;
train via checkpoint files; arena via its own done.txt). `run_forever` resumes
mid-iteration after any crash. Per-batch/30-60 s observability in every stage.

**Iteration 1 salvage:** the 5-process run launched 00:42 tonight (champion =
Cell 6 = `round0_Cell6.pt`, exploration on, full Catan, seeds 21M–25M) is
iteration 1's SELF-PLAY stage, already ~150 games/h aggregate (measured
01:55–02:05). The loop's first action is to consume its output.

## 3. Throughput: verify-then-commit

Measured tonight: 5 independent GPU procs ≈ **150 games/h aggregate** (51 games
by ~80 min, waves landing; refine at data-gen end). Old single-worker baseline:
~0.8 games/min × workers.

Order of work:
1. **Loop v1 runs on the current 5-proc architecture** — no new infra blocks the loop.
2. **B1 inference-server spike** (timeboxed ~half day): one GPU process owning
   the model; N CPU search workers send observation batches over
   `multiprocessing` connections; server batches across workers (window+cap, same
   flush logic as BatchedGnnEvaluator). Measure: round-trip latency, aggregate
   games/h at 10 workers. **Go: ≥2× the 5-proc aggregate. No-go fallback:**
   independent GPU procs at VRAM-fit max (6–7 × ~535 MiB on the 4 GB card).
3. **Milestone-2 lever (not now):** Rust-side self-play (engine-native MCTS +
   inference via tch/ort or candle) — the 10-100× option, justified only if the
   measured Elo-vs-games slope demands millions of games.

**Projection methodology (the user's feasibility question):** the journal IS the
projection instrument. After ~10 iterations, fit Elo-vs-cumulative-games; report
projected games (and wall-clock at measured games/day) to reach: (a) PureGnn
deploy beats LookV3 head-to-head, (b) champion GnnMcts >65% vs LookV3,
(c) >95% vs LookV3 ("unbeatable by the old baseline"). Scenario table in the
first loop journal entry; updated every 5 iterations. No promises before slope
data exists.

## 4. Engine fidelity curriculum

One increment per ~few iterations, each: its own mini-spec → TDD → flag-gated
(default off) → enabled in loop config → models continue training through the
change. Observation layout stays stable throughout; **one** action-space
expansion event total.

| Stage | Change | Actions | State-size approach |
|---|---|---|---|
| F1 | Robber steal-victim becomes a decision (spec approved 2026-06-01) | reuse pattern, no new ids | masked decision phase; victim options ≤3 |
| F2 | Strategic discard on 7 (player chooses, replacing instant-random) | `Discard(r)` ids 199–203 already exist | sequential discard decisions; hand-size bounded |
| F3 | Trade accept/reject — opponents decide, ending 100% auto-accept | +2 ids (AcceptTrade, RejectTrade) | response phase, one bit per opponent queried in seat order |
| F4 | Multi-resource trades (2:1, 1:2, 2:2 grids) | ~+40 ids | bounded offer grid, masked |

**Action-space policy:** single expansion 280 → ~322 superset before F3 lands.
Policy-head transfer: old logits keep trained weights, new logits random-init,
illegal-masked until their stage activates. After expansion the head never
changes again.

F1–F4 are *engine-fidelity* work (rules of the game humans play), explicitly not
bot-strategy scripting — strategies emerge from self-play on the truer game.

## 5. Web ladder integration

Difficulty presets (shipped tonight) gain a dynamic tier: promoted champions
appear as `AZ iter-N` entries (bot_registry reads the ladder registry). Beginner
→ Expert static tiers remain. Humans always have opponents at every strength.

## 6. Testing

- `catan_az` unit tests: config round-trip; buffer window selection (drops
  oldest, respects cap, mixed run dirs); arena seating/rotation/scoring;
  promote/hold logic incl. timeout-rate guard; ladder Elo math; status writer.
- Integration: one micro-iteration end-to-end (games_per_iter=4, sims=8, 1
  epoch, arena 8 games) on CPU in CI-tolerable time — proves the plumbing, not
  strength.
- The running iteration-1 data doubles as the real-data fixture.

## 7. Config defaults (v1)

```python
AzConfig(
  games_per_iter=400,          # ≈2.5-3h self-play at measured 150 games/h
  window_games=1200,           # ~3 iterations
  sims=200, dirichlet_alpha=0.8, dirichlet_eps=0.25, temp_moves=30,
  lr=2e-4, max_epochs=4, early_stop=True, policy_sharpen=1.0,
  arena_games=120, promote_threshold=0.55, arena_timeout_rate_max=0.05,
  anchor_every=5, anchor_games=60,
  vp_target=10, bonuses=True,
  n_procs=5,                   # until B1 spike verdict
)
```

## 8. Out of scope (this spec)

- F1–F4 implementations (each gets its own mini-spec; F1's exists).
- Rust self-play port (milestone-2, slope-gated).
- Dashboard UI beyond status.json (existing dashboard patterns can consume it).
