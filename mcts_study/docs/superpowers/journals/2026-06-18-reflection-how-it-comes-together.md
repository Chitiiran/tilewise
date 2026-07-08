# Reflection: How All This Comes Together to Make the GNN Better

**Date:** 2026-06-18
**Scope:** Connecting the arena-timeout investigation → the Rust-MCTS + TorchScript
rewrite → batching/throughput work, back to the ONE goal: a stronger Catan GNN.

---

## 0. The one goal (never lost, sometimes buried)

We are training a Catan bot with **AlphaZero**: a GNN that, through repeated
rounds of **self-play → train → evaluate (arena) → promote**, gets stronger.
Everything below is in service of turning that loop faster and more correctly.
The loop only improves the net if each stage works AND if it can run enough
iterations in human time.

```
        ┌─────────────┐     game records      ┌──────────┐
        │  SELF-PLAY  │ ───(states, π, z)───▶ │  TRAIN   │
        │ (MCTS+GNN)  │                        │ (PyTorch)│
        └─────────────┘                        └────┬─────┘
              ▲                                      │ candidate net
              │ promoted champion                    ▼
        ┌─────┴───────┐    winrate verdict     ┌──────────┐
        │   PROMOTE   │ ◀────────────────────  │  ARENA   │
        │  (ladder)   │                         │ cand vs  │
        └─────────────┘                         │ champion │
                                                └──────────┘
```

---

## 1. Where it started: the arena couldn't finish, and "wins" were fake

The trigger (memory `project_arena_latency_bound_2026_06_17`): **arena games
100% timed out** — 0/48 finished naturally at concurrency=16. Because no game
reached the real win condition (**10 VP, first player to it**), the verdict fell
back to a **VP-margin tiebreak at a wall-clock cap**: whoever was *ahead on
points when the clock ran out* was declared the winner.

That is a corrupt evaluator. Two problems, both fatal to the AZ loop:
- **Wrong signal:** a bot ahead on VP at minute 25 of an unfinished game is not
  the bot that would have *won the game*. The arena was selecting for
  "accumulates VP fast early," not "wins Catan." Promotions based on that would
  drift the net toward the wrong objective.
- **Too slow to iterate:** even when games did finish (c=8), they took ~1200s
  median; a 300-game arena = many hours. With ~10h/iteration already, the loop
  couldn't run enough rounds to make the net meaningfully better.

So the very thing that decides "is the new net better?" — the arena — was both
**slow** and **measuring the wrong thing**. That is why we stopped and dug in.

## 2. Why it was slow: latency, not compute (the diagnosis that reframed everything)

The arena wasn't GPU-bound — the GPU sat at **28% util, 4W of 75W, 3% VRAM**.
It was **latency-bound**: every MCTS node crossed Python → PyO3 → Rust engine →
back, and Python → PyTorch → back, *millions of times per game*. The GPU
finished each tiny forward in microseconds, then idled waiting for the next leaf
to crawl through the asyncio/PyO3 path. Arena was worse than self-play because it
ran **two** evaluators (cand + champ) whose leaves couldn't batch together.

Crucially: **the same bottleneck throttles self-play** (the data *generator*).
So fixing it speeds up BOTH halves of the loop — more data per hour AND faster,
correct verdicts.

## 3. What we built: MCTS into Rust, GNN via TorchScript, zero per-node crossings

The rewrite (`catan_mcts_rs`): move the entire MCTS — tree, UCB selection,
expansion, `apply_action`, chance handling — into Rust calling the engine
directly (no PyO3 per node), and run the GNN in-process via `tch-rs` loading a
**TorchScript-exported** copy of the trained net. Training stays in PyTorch,
untouched.

The non-negotiable: **bit-exact to the Python oracle**. A faster loop that
silently searches differently would poison every future iteration invisibly —
you'd train on subtly-wrong data forever and never know. So every unit was
double-verified (golden + differential), culminating in the **24/24 self-play
gate** and **production-net cross-check (2/2 bit-exact at sims=200)**. The Rust
engine produces the *identical* games the Python path would — just faster.

This is the deep reason the whole effort is legitimate: **we changed the engine
without changing the science.** The net trains on the same distribution of
self-play games; the arena measures the same way — only now it actually finishes.

## 4. The bottleneck kept moving — and each move taught us where the loop's real cost is

This is the arc worth remembering. We didn't fix "the" bottleneck once; we
chased it as it relocated, and each location is a fact about the AZ loop:

1. **Per-node PyO3/asyncio crossings** (original). Killed by moving MCTS to Rust.
   → 2.8× faster single-threaded, games finish naturally.
2. **GPU starvation / one-leaf-at-a-time.** Killed by cross-game leaf batching
   (Task 10) → ~16× at the production config (B=32, 32 games) on CUDA. The GPU
   went from idle (3.7W) to working (29.6W).
3. **Batch under-fill.** Profiling (the surprise) showed the CPU was only 3-8%;
   the real loss was half-empty GPU batches (mean B=10.7/32 at 16 games). Fixed
   by **concurrency** — more simultaneous games fill the batch: G=512 → mean
   B=29.9/32 (93%).
4. **Diminishing returns + the GPU forward itself.** Past ~G=256 the batch is
   ~full and we're genuinely GPU-forward-bound (~2,525 leaves/s) — the
   deterministic-scatter kernel (~20ms/call), well under the raw 9,400 states/s
   ceiling.
5. **The single-threaded scheduler** (current frontier). At G=256-512 one core
   is pegged at 93-95% while 11 sit idle — the scheduler serializes per-game CPU
   work AND host-side tensor marshaling between forwards. Multi-threading it
   (deterministic barrier; in design) is the next lever; whether it's worth it
   hinges on the marshal-vs-forward split we're measuring now.

The lesson (already in `feedback_experiment_design_discipline` territory):
**optimize what you measure, re-measure after each fix, and stop when the
bottleneck moves to something that doesn't matter.** We were about to "optimize
the CPU" on a theory; the profiler said CPU was 8% and saved that effort.

## 5. The replayability constraint — why this is about MORE than speed

A recurring hard requirement (user): the loop must be **reproducible** — same
seeds + net + config → byte-identical games, every run. Reason: **we don't store
full games**, so to debug "why did the net do X in that match," we must be able
to *replay it exactly* from the seed. This constraint shaped every speed choice:
- It's why the RNG had to be a bit-exact NumPy PCG64 replica in Rust.
- It's why batching uses **deterministic CUDA** (the scatter kernel is otherwise
  nondeterministic) and a **fixed batch composition** (slot order), even at a
  ~2.6× speed cost vs non-deterministic mode — which we explicitly rejected.

Reproducibility is not a nicety here; it is what makes the generated data and
the arena verdicts *trustworthy and auditable*. A faster loop you can't replay
would be a faster way to produce results you can't debug.

## 6. How this makes the GNN better — the payoff

Tie it back to the four stages:

- **SELF-PLAY (data generation).** Faster, batched, GPU-fed self-play means
  **more games per hour** → more (state, policy-target, value-target) training
  examples per iteration → the net sees more of the game tree and trains on a
  richer distribution. And because games now **finish naturally** (10-VP wins,
  not wall-clock cuts), the **value target z is the true game outcome**, not a
  censored VP-margin. The value head — the exact thing
  `project_e10e_gnnmcts_worse_than_puregnn` identified as the missing piece (the
  behavior-cloned net's value head was never trained on real outcomes) — finally
  gets clean labels. That is the direct path to MCTS *helping* instead of
  hurting.

- **TRAIN.** Unchanged on purpose — but it now receives better, more, and
  uncensored data. Same training code, better inputs → better net.

- **ARENA (evaluation).** Now finishes games and decides by the **real win
  condition**, fast enough to run a full 300-game gate per iteration. Promotions
  reflect "which net actually wins Catan," not "which net is ahead when the clock
  stops." The ladder finally selects for the right objective.

- **PROMOTE + iterate.** Faster correct iterations → the loop can actually run
  the *many* rounds AlphaZero needs to climb. The original standing objective
  (`project_iter7plus_run_to_promotion`) — "run until a champion is promoted" —
  was blocked precisely because each iteration was too slow and the arena
  verdict was untrustworthy. This whole effort unblocks it.

**In one sentence:** we rebuilt the loop's engine so it generates more,
correctly-terminated, replayable games per hour and evaluates them by the true
win condition — which gives the value head the honest outcome labels and the
ladder the honest verdicts that AlphaZero needs to actually make the GNN
stronger.

## 7. Where we are, and the honest next decision

- The rewrite is **done and verified** (Phases 0-10, bit-exact, pushed). The
  engine default is `rust`. Self-play and arena both finish naturally.
- Throughput: ~16× over the old per-leaf path at production config; ~2,525
  leaves/s, GPU-bound, with a single-threaded scheduler as the current ceiling.
- **The remaining optimization (multi-threaded scheduler) is an
  efficiency play, not a correctness one.** The loop is already correct and
  much faster. The open question is only *how much further* throughput is worth
  pushing before just running the loop.

**The reflection's takeaway for what to do next:** the infrastructure goal is
essentially met. The highest-value next action is arguably not more
optimization — it is to **run the actual AZ loop** on this fixed engine: generate
a real iteration of naturally-terminated self-play, train, run a real
finishes-naturally arena, and see the value head start to learn from true
outcomes. The throughput work has bought us the ability to do that in reasonable
time; the science win is in spending that time on iterations, not on shaving the
next 2× off a loop that already runs.
