# Batched GNN Evaluator + Async Self-Play Engine — design

**Date:** 2026-05-30
**Status:** design approved, pre-implementation
**Author:** brainstormed with Claude (superpowers:brainstorming)

## Motivation

The current `GnnEvaluator` (`catan_mcts/gnn_evaluator.py`) runs **one batch=1
forward pass per MCTS leaf**. At sims=200 over deep full-Catan games this is the
dominant cost: the 2026-05-29 e10e diagnostic measured ~4.3 min/game for a single
GnnMcts seat, and the 10-worker tournament suffered a 32% wall-clock-timeout rate
under GPU contention (cited: `project_gnn_mcts_game_cost_2026_05_29`,
`project_e10e_gnnmcts_worse_than_puregnn_2026_05_29`).

GPUs are throughput devices: one tiny h128 graph costs nearly the same wall-clock
as a batch of 64, because per-call overhead (Python→CUDA handoff, host↔device
transfer, kernel launch) dominates over the actual matmul. Feeding the GPU one
leaf at a time wastes ~98% of its capacity (observed: 89% "util" doing mostly
overhead).

**This matters for RL, not just diagnostics.** An AlphaZero-style self-play loop
needs thousands–tens of thousands of games per training iteration. At 4 min/game
that is infeasible (~28 GPU-days for 10k games). Batched evaluation is the
enabling infrastructure on the critical path for RL regardless of which RL flavor
is chosen.

## Goal & scope

**In scope (this spec):**
- `BatchedGnnEvaluator` — async evaluator that batches leaf evals across many
  concurrent games into single forward passes.
- `AsyncMcts` — a minimal (~150-line) async MCTS whose leaf evaluation is an
  `await` suspension point.
- `self_play_async.py` — orchestrator running N game-coroutines, reusing the
  existing `SelfPlayRecorder` to write e9-schema parquets.
- A clean re-run of the e10e diagnostic on this stack as the correctness gate.

**Out of scope (future specs):**
- The AlphaZero training loop (replay buffer, train step, net replacement, eval
  gating).
- Multi-GPU / multi-process inference server (architecture "B1").
- Virtual-loss within-search (intra-game) leaf batching.
- Tree reuse / transposition tables across moves.

## Primary consumer

**Self-play data generation for RL.** Throughput across many concurrent games is
what matters; individual game latency does not. This is what makes
**batch-across-games** the right architecture and lets us leave OpenSpiel's MCTS
untouched (we replace it for self-play with our own async MCTS, but don't fork
its C++ internals).

## Architectural decisions (and rejected alternatives)

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Batching axis | across concurrent games | within one search (virtual loss) | within-search needs replacing OpenSpiel MCTS; across-games saturates the GPU at N=64 already |
| Concurrency model | single-process **asyncio** (architecture "B2") | multi-process + inference server ("B1") | one GTX 1650; IPC server is YAGNI until a 2nd GPU exists. asyncio = no IPC serialization, single GPU context |
| MCTS driver | **minimal async MCTS in Python** | thread-per-game wrapping OpenSpiel; PureGnn-only self-play | threads reintroduce GIL contention + blocking-bridge complexity; PureGnn drops search quality. Custom async MCTS gives full control + a clean `await`-at-leaf |
| Step-boundary batching | rejected | — | with blocking MCTS, games desync immediately and batches stay tiny (~2); does not deliver the batch |

## Architecture

Single-process asyncio engine, four cooperating components:

```
self_play_async.py (orchestrator)
  spawns N game-coroutines, each driving one AsyncMcts to play a full game
      │  each game awaits leaf evals
      ▼
  AsyncMcts (one per game-coroutine)
    select(UCB c=1.4) → expand → `await evaluator.eval(leaf)` → backup
    ~200 sims/move; suspends at each leaf
      │  (state) → Future
      ▼
  BatchedGnnEvaluator (one, shared)
    parks each request on a pending queue; batcher coroutine drains when
    batch full OR window elapsed OR all-live-games-parked → ONE forward pass
    → resolves each caller's Future
      │
      ▼
  GnnModel (GPU)

  finished games → SelfPlayRecorder (reused) → games/moves parquet (e9 schema)
```

**Component isolation:** the evaluator is testable with synthetic await-storms
(no MCTS); the MCTS is testable against a trivial sync evaluator (no batching).
They meet only at the `async eval(state) -> (value, policy)` interface.

## Component 1 — BatchedGnnEvaluator

Interface: `async eval(state) -> (value: np.ndarray[4], policy_logits: np.ndarray)`.

Per call:
1. `features = state_to_pyg(state.observation())` (CPU, cheap)
2. `fut = loop.create_future()`
3. append `(features, fut)` to `pending_queue`; signal batcher
4. `return await fut` — **coroutine suspends here**, letting other games' leaves
   accumulate.

Batcher coroutine:
```
loop:
  wait until flush condition
  drain a batch of (features, fut) pairs
  batch = Batch.from_data_list([f for f,_ in drained]).to(device)
  with torch.no_grad(): values, logits = model(batch)
  move to CPU
  for i,(_,fut) in enumerate(drained): fut.set_result((values[i], logits[i]))
```

**Flush condition (the deadlock guard):**
```
flush_now = (len(pending) >= MAX_BATCH)
         or (len(pending) >= active_game_count)   # everyone who can ask, has
         or window_elapsed_since_first_pending
```
The middle clause is critical: it means "all live games are already parked — do
not wait for more." Without the `window_elapsed` guard the run hangs as games
drain below MAX_BATCH near end-of-run; without the middle clause every late-game
batch silently degrades to a timeout-fired partial.

**Chance / terminal short-circuit:** chance nodes (dice) and terminal states
resolve without the model and never enter the batch queue (mirrors the current
`prior()` chance-node branch).

**OOM handling:** cap `MAX_BATCH` conservatively (default 64); on CUDA OOM in the
batcher, halve the batch, retry once, then fail loud logging the breaking size.

## Component 2 — AsyncMcts

Standard PUCT/UCB MCTS; the only `await` is the leaf eval.

```
async def search(root_state, n_sims):
    root = Node(root_state)
    for _ in range(n_sims):
        node, path = root, [root]
        while node.is_expanded and not node.state.is_terminal():
            node = select_ucb_child(node, c=1.4)
            path.append(node)
        if node.state.is_terminal():
            value = terminal_returns(node.state)        # length-discounted
        elif node.state.is_chance_node():
            value = await handle_chance(node)            # expand outcomes, no GPU
        else:
            value, priors = await evaluator.eval(node.state)   # SUSPEND
            node.expand(priors)
        backup(path, value)
    return visit_counts(root)        # policy target = normalized visit counts
```

**Semantics that MUST match OpenSpiel's MCTSBot** (silent strength changes if wrong):

| Aspect | Decision | Source |
|---|---|---|
| Selection | UCB, `c=1.4`, priors from policy head | every prior tournament |
| Value perspective | 4-player: backup per-seat value vector, select on acting player's component | Catan is 4-player |
| Terminal returns | length-discounted `DECAY^steps` | `project_v2_rollout_decay` |
| Chance nodes | expand by `chance_outcomes()`, sample on descent; no GPU | engine |
| Final move | argmax visit count (recorded as policy target) | standard |

## Hardening (folded into v1 — each defends a previously-hit failure mode)

**1. Stuck-game watchdog.** Every window assert `parked + running == active_games`.
If `running > 0` persists across many windows while batches stay tiny, a game is
stuck on something other than an eval — log its **seed + last action**. Converts a
silent ~2× slowdown into a loud, located failure. (Defends: silent throughput
collapse.)

**2. Per-game RNG ownership + distributional equivalence.** Each game-coroutine
owns its RNG, seeded from the game seed; never draw from shared/batcher state.
This makes a game's *decisions* reproducible given its seed even though wall-clock
interleaving and batched-forward float-reduction order are nondeterministic.
**Consequence:** the equivalence bar is **distributional (winrate within CI), not
bit-identical** — async batching makes exact action-for-action repro with
OpenSpiel impossible by construction, so we do not chase it. (Defends: wasting a
day chasing an impossible repro.)

**3. Memory-budget backpressure.** Estimate per-game tree RAM from a probe; cap
`N = min(desired, ram_budget / per_game_tree)`; **log the cap** (no silent caps).
(Defends: WSL OOM crash — a prior failure on this box; cited
`crash-and-cleanup-journal`.)

## Tuning knobs

| Knob | Too low | Too high | Default |
|---|---|---|---|
| `MAX_BATCH` | small batches, less speedup | VRAM pressure (4GB) | 64 |
| `WINDOW_MS` | fires before batch fills | latency, idle games | 5 |
| `N_GAMES` | batch never fills | RAM for N trees | 64–128 (capped by budget) |

Constraint: `N_GAMES >= MAX_BATCH` so there are always enough suspended games to
fill a batch.

## Error handling & resumability

| Failure | Guard |
|---|---|
| Batcher deadlock at end-of-run | window timeout + all-parked flush clause |
| A game-coroutine raises | per-coroutine try/except; log seed+traceback; record skipped; other N−1 continue (`gather(return_exceptions=True)`) |
| GPU OOM | conservative MAX_BATCH; halve-and-retry-once then fail loud |
| Slow/runaway game | per-game wall-clock cap AND step cap; record skipped, move on |
| Whole-run crash | per-game flush to parquet; `done.txt` → resumable |

Resumability reuses `SelfPlayRecorder` (cell-flush + done-tracking) unchanged.
Single-process → one `done.txt` (simpler than the 10-worker tournament layout).

**Observability** (per-batch, not per-epoch): log every ~30–60s — games done / N,
**mean achieved batch size** (the headline health metric), games/sec, GPU util.
Batch size ≈ 64 → winning; ≈ 2 → batching failed and the run is no faster than
before.

## Testing strategy (TDD — tests first)

**Unit:**
- `test_batcher_fills_to_max` — 64 concurrent evals → exactly one forward pass; all futures correct.
- `test_batcher_window_fires_partial` — 3 calls + window → partial batch resolves (no hang).
- `test_batcher_flush_when_all_parked` — pending == active → immediate flush.
- `test_chance_and_terminal_skip_gpu` — those leaves never enter the queue.
- `test_watchdog_flags_stuck_game` — stalled coroutine logged within K windows.
- `test_async_mcts_matches_sync_on_toy` — visit counts vs reference on a tiny game.
- `test_per_game_rng_reproducible` — same seed → same action sequence.
- `test_memory_budget_caps_N` — tight budget → N capped and logged.

**Integration:**
- `test_self_play_writes_valid_parquet` — 4 async games → e9-schema parquets load into `CatanReplayDataset`.
- `test_resume_skips_done_seeds` — kill after 2 games, restart, skips them.

## Acceptance criteria (definition of done) — two gates

**Gate 1 — Throughput.** Report mean achieved batch size and games/sec.
Baseline = the single-worker probe rate of ~4.3 min/game (~256 s/game) measured
2026-05-29 (`project_gnn_mcts_game_cost_2026_05_29`), NOT the contended 10-worker
tournament rate. **Pass:** mean batch ≥ 32 (of 64) AND per-game wall-clock ≤ 24 s
(≥10× faster than the 256 s/game baseline). Batch collapse to ~2 ⇒ design failed;
diagnose before proceeding.

**Gate 2 — Correctness (clean e10e re-run).** Re-run the e10e diagnostic
(Cell6 PureGnn / Cell6 GnnMcts / Cell1 PureGnn / LookV3, full Catan) on the
async/batched stack with timeout rate < 5% (vs old 32%).
**Pass:** GnnMcts winrate consistent (within CI) with the old bias-corrected
finding (GNN+MCTS ≤ PureGnn). This simultaneously proves the async MCTS plays
correctly AND gives the clean number that settles the GNN+MCTS question without
the contention artifact.

If Gate 2 *contradicts* the old result (GnnMcts suddenly competitive), that is a
real finding: either the old result was contention-driven, or the async MCTS
diverges from OpenSpiel. Both are diagnosable and worth knowing before RL.

## Cited memory / artefacts

- `project_gnn_mcts_game_cost_2026_05_29` — 4.3 min/game cost baseline.
- `project_e10e_gnnmcts_worse_than_puregnn_2026_05_29` — the result this validates.
- `project_e5_winrate_was_wallclock_artefact` — why timeout rate must be <5%.
- `project_v2_rollout_decay_2026_05_01` — length-discounted terminal returns.
- `feedback_salvage_compute_and_time`, `feedback_training_observability`,
  `feedback_experiment_design_discipline` — hardening rationale.
- `crash-and-cleanup-journal` (2026-05-09) — the WSL OOM that motivates backpressure.
- Current code: `catan_mcts/gnn_evaluator.py`, `catan_mcts/bots_gnn.py`,
  `catan_mcts/experiments/e10e_gnn_mcts.py`, `catan_mcts/recorder.py`.
