# 9-hour compounding AlphaZero loop — beat LookV3 with PureGnn

**Date:** 2026-05-31 ~21:40 EDT → ~07:00 EDT 2026-06-01 (~9.4h)
**Goal (user):** maximize compute (10 workers), run autonomously to 7am, chase
the HARD target: **PureGnn (raw policy, no search) beats LookV3.**

## Why the 1-round AZ test failed, and how this fixes it

The 1-round AZ-exploration test (117 games) DEGRADED the net: exploratory targets
are noisier, and 117 games is too few for that variance to average into
improvement. Naively repeating it would compound degradation. Fixes:

1. **ACCUMULATE the corpus** (user choice). Each checkpoint trains on the FULL
   growing pool, not a fresh 117. Round 1 ~150 games, mid-run ~500, end ~900+.
   Growing data averages out exploration noise → learning can compound.
2. **ARENA-GATE every candidate** — only promote a net that beats the champion.
   Champion starts = Cell6. No regression spirals.
3. **Train from Cell6 (fixed strong init) on the accumulated pool** each time —
   avoids the warm-start-a-degrading-net death spiral. (The corpus improves via
   accumulation + champion-sourced self-play, not via re-init.)
4. **Continuous self-play** (user choice): 10 workers generate NON-STOP for 9h
   (data is the bottleneck); periodic train/arena snapshots measure progress.

## Architecture

```
CORPUS_DIR = runs/v3/az9h_corpus/   (one accumulating dir, all workers write here)
champion   = Cell6  (until something beats it)

CONTINUOUS: 10 self-play workers, --self-play (Dirichlet+temperature), from the
  current champion checkpoint, sims=160, 2h caps, relaunched as they expire.
  Distinct seed-bases so no collision; same out-root so corpus accumulates.

EVERY ~90 min (checkpoint k = 1..~6):
  1. snapshot: count games in CORPUS_DIR.
  2. train candidate_k: from Cell6, on the FULL CORPUS_DIR, epochs=8,
     early-stop-patience=2, lr 5e-4, rotate. -> az9h_train_k/checkpoint_best.pt
  3. arena_k: e10e_async, candidate_k as PureGnn(A) + GnnMcts(B), champion(C),
     LookV3(D). 80 games. seed-base 80M + k*1M.
  4. measure: candidate_PureGnn vs LookV3 (THE goal metric), candidate_GnnMcts
     vs LookV3, candidate vs champion.
  5. PROMOTE if candidate beats champion (head-to-head or higher vs LookV3):
     champion <- candidate_k; save rl_checkpoints/az9h_round{k}.pt; switch
     self-play source to the new champion.
  6. journal the checkpoint row. commit.

STOP at 07:00 EDT, OR if candidate_PureGnn beats LookV3 convincingly (>~5pp over
  80 games) -> GOAL MET, final report.
```

## The compounding hypothesis (what success looks like)

Track candidate_PureGnn-vs-LookV3 across checkpoints k=1..6:
- **Compounding:** the number climbs (3.8% → 8 → 15 → 25 → ... → >LookV3). The
  growing accumulated corpus + champion improvement let the policy progressively
  absorb search's behavior.
- **Plateau:** it flattens well below LookV3 → strong evidence the policy can't
  encode LookV3-level play at h128 regardless of data; search stays irreducible.

Either is a real finding. The divergence diagnostic (trade-override rate) is the
mechanistic companion — it should keep dropping if compounding is real.

## Resource plan

- 10 workers, single-exec harness-tracked (NOT nohup/&+wait — both die). ~12
  cores, 10 single-core self-play procs fit. ~2-3 games/min combined →
  ~150-180 games/hr → ~900-1200 games over 9h if continuous.
- Per-game persistence; corpus accumulates; resumable.
- Train (~5-10 min) + arena (~40-60 min) run alongside self-play (share cores;
  self-play slows but doesn't stop).

## Risks + mitigations

- **Self-play workers die / WSL flake:** harness-tracked single-exec survives;
  relaunch on each checkpoint; per-game persistence salvages all finished games.
- **Arena slow (GnnMcts seat):** 80 games ~50 min; acceptable at ~90-min cadence.
- **No compounding (plateau):** that IS the answer — document it; GnnMcts (53.8%
  vs LookV3) remains deployable.
- **Context windows:** compact as needed; the champion pointer (CHAMPION.txt) +
  this plan + per-checkpoint journals make the loop resumable from any window.

## Stop criteria
07:00 EDT, OR PureGnn beats LookV3 >~5pp. Commit every checkpoint. Report the
PureGnn-vs-LookV3 trajectory at the end regardless of outcome.
