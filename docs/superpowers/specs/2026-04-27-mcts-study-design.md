# MCTS Study — Design Spec

**Date:** 2026-04-27
**Status:** Draft — awaiting user review
**Project:** `catan_bot/mcts_study/` — first project building on top of the v1 engine
**Builds on:** [`2026-04-27-catan-engine-design.md`](2026-04-27-catan-engine-design.md)

## 1. Goal

Use OpenSpiel's MCTS implementation, driven by our Rust Catan engine, to:

1. **Produce an experimental study** of how MCTS performs on Catan — covering win-rate vs. random at multiple simulation budgets, sensitivity to the UCB exploration constant `c`, random vs. heuristic rollout policies, and a small tournament against a hand-coded greedy baseline.
2. **Generate a self-play dataset** suitable for training a GNN policy/value network in the next project.

The deliverable is *understanding*, not a maximally strong bot. This project is a learning ground for "how to use MCTS as a tool" — strength is a measurement output, not a target.

### Non-goals

- Beating a human, or beating Catanatron's bots.
- Writing our own MCTS — we use `open_spiel.python.algorithms.mcts`.
- Training a neural network — that's the next project.
- Engine changes beyond the minimum required to expose the APIs MCTS needs.

### Success criteria

1. `MCTSBot` (OpenSpiel) plays a full Catan game against `RandomBot` end-to-end, deterministically given a seed, no exceptions.
2. The four experiments run from a single command (`python -m catan_mcts run all`) and write results to a versioned `runs/<timestamp>-<name>/` directory.
3. A short markdown writeup (~5 pages) summarising results, with plots, and a `learnings.md` containing **at least 3 concrete reusable learnings about MCTS as a tool**.
4. **A self-play dataset on disk in a documented schema** that the next (GNN) project can load directly. Includes per-move MCTS visit counts, the action taken, and the game's final per-player outcomes.

## 1.5 Self-play dataset (bridge to GNN project)

This is the bridge to the next project. Designing it now costs little; retrofitting it later costs a full re-run of all experiments.

### Per-move row schema (`moves.parquet`)

| Column | Type | Notes |
|---|---|---|
| `seed` | int64 | Engine seed for this game |
| `move_index` | int32 | 0-based index of this MCTS-decided move within the game |
| `current_player` | int8 | The MCTS player making this move |
| `legal_action_mask` | list<bool>[205] | Matches engine's action space |
| `mcts_visit_counts` | list<int32>[205] | Raw root visit counts — the policy target |
| `action_taken` | int32 | The action MCTS played |
| `mcts_root_value` | float32 | MCTS's value estimate at the root |
| `schema_version` | int32 | Hardcoded constant, bumped on schema change |

### Per-game row schema (`games.parquet`)

| Column | Type | Notes |
|---|---|---|
| `seed` | int64 | |
| `winner` | int8 | -1 if no winner |
| `final_vp` | list<int32>[4] | |
| `length_in_moves` | int32 | Total moves in game (incl. non-MCTS) |
| `mcts_config_id` | string | FK to `config.json` for this run |
| `schema_version` | int32 | |

### Storage decisions

- **Parquet**, columnar. Pandas / PyTorch / Polars all read it natively.
- **One pair of parquets per experiment run.** ~tens of MB per 1k self-play games.
- **No observations stored.** The engine is deterministic (v1 spec §9 with hash regression tests), so the GNN project will replay `(seed, action_history_up_to_move_index)` to materialize observation tensors on demand. This avoids freezing the v1-TBD observation schema (dice encoding, plane order) into our dataset, and gives the GNN project full control over feature engineering.
- **Only MCTS moves are recorded.** Opponent (random / greedy) moves and chance-node steps are part of the game state but are not training targets.
- Schema versioned via the `schema_version` column. Bump explicitly on any field change. The GNN project's loader asserts the version it expects.

## 2. Architecture

A new top-level `mcts_study/` directory inside this repo. Imports `catan_bot._engine` directly. Has its own `pyproject.toml` and venv so OpenSpiel's deps don't pollute the engine's environment.

### 2.1 Layout

```
catan_bot/
  catan_engine/                # unchanged — Rust crate (Phase 0 adds APIs; see §3.7)
  python/catan_bot/            # unchanged — engine bindings
  mcts_study/                  # NEW
    pyproject.toml             # deps: open_spiel, pandas, pyarrow, matplotlib, hypothesis, pytest
    catan_mcts/
      __init__.py
      adapter.py               # CatanGame, CatanState — OpenSpiel interface
      bots.py                  # GreedyBaselineBot, heuristic_rollout
      recorder.py              # SelfPlayRecorder — writes parquet
      experiments/
        __init__.py
        common.py              # shared CLI args, seed handling, output paths
        e1_winrate_vs_random.py
        e2_ucb_c_sweep.py
        e3_rollout_policy.py
        e4_tournament.py
      cli.py                   # `python -m catan_mcts run e1` etc.
    notebooks/
      analysis.ipynb           # plots from runs/ — read-only consumer of parquet
    tests/
      test_adapter.py
      test_recorder.py
      test_smoke_mcts_vs_random.py
      test_determinism.py
    runs/                      # gitignored; one subdir per run, timestamped
    docs/
      writeup.md               # ~5-page deliverable
      learnings.md             # ≥3 reusable MCTS learnings
```

`runs/`, the venv, `__pycache__`, parquet files in `runs/`, and notebook outputs (via `nbstripout`) are gitignored.

### 2.2 Boundaries

- **`adapter.py` is the only module that knows about `catan_bot._engine`.** Everything downstream (bots, experiments, recorder) talks only to OpenSpiel types. This is the abstraction that makes future "swap MCTS for CFR / AlphaZero" cheap — the same adapter serves any OpenSpiel algorithm.
- **`recorder.py` observes games, doesn't drive them.** Decoupled so experiments can opt-in / opt-out.
- **Experiments are scripts, not a framework.** Each `eN_*.py` is a self-contained `main()` that takes a seed/run-name and writes one `runs/<name>/` directory. A `Makefile` or `cli.py` chains them.
- **`notebooks/analysis.ipynb` is read-only over `runs/`.** It never re-runs experiments. Experiment runs are slow; plotting iterates fast.

| Concern | Module |
|---|---|
| "How does Catan map to OpenSpiel?" | `adapter.py` only |
| "How do we run MCTS?" | `open_spiel` — we don't write this |
| "What experiments do we run?" | `experiments/eN_*.py` |
| "What do we record for the GNN project?" | `recorder.py` |
| "What does it look like?" | `notebooks/analysis.ipynb` |

## 3. Components

### 3.1 `adapter.py` — `CatanGame` / `CatanState` *(load-bearing)*

Implements OpenSpiel's `Game` and `State` Python interfaces by wrapping `catan_bot._engine.Engine`. Translates between OpenSpiel concepts (player IDs, action ints, chance nodes, returns) and our engine's concepts (action IDs 0-204, dice rolls via RNG, terminal reward).

OpenSpiel algorithms (`MCTSBot`, `RandomBot`, future `AlphaZero`) consume `CatanGame.new_initial_state()` and call `state.legal_actions()`, `state.apply_action(a)`, `state.is_terminal()`, `state.returns()`, `state.is_chance_node()`, `state.chance_outcomes()`, `state.clone()`, `state.serialize()`. The adapter translates each call.

**Hard parts:**

- **Chance nodes for dice and steal.** OpenSpiel models stochasticity by separating chance from player decisions: `is_chance_node()` returns True at chance points, `chance_outcomes()` returns `[(outcome, probability), ...]`, and MCTS uses these probabilities to weight branches correctly. Today's engine rolls dice internally during `Roll` and auto-steals during the steal phase. We split both into chance points (Phase 0; see §3.7). Without this, MCTS would treat dice as a player decision — the tree would be biased and UCB statistics would be meaningless. See §3.8.
- **`clone()`.** MCTS clones states heavily during simulation. Our `GameState` is `#[derive(Clone)]` and uses `Arc<Board>` so topology isn't copied. Need a PyO3 method `Engine.clone()` that returns a deep-copy `Engine`. Phase 0.
- **`serialize()`.** OpenSpiel uses serialized state strings as transposition-table keys. Doesn't have to be human-readable — just stable and unique. Implementation: hex-encoded `(seed, action_history)`. Action history is tracked Python-side in the adapter; no engine change needed.
- **`returns()`.** OpenSpiel expects `[r0, r1, r2, r3]`. Engine produces +1/-1 winner/losers; non-terminal returns `[0,0,0,0]`.
- **Action space.** OpenSpiel wants a fixed `num_distinct_actions()`. We have 205. Engine's `legal_actions()` already returns IDs in this space.

**Risk:** OpenSpiel may have additional contract surface (`information_state_string`, `observation_tensor`, etc.) that some algorithms require. MCTSBot itself is forgiving — needs legal actions, apply, clone, returns, terminal, chance. We implement that subset and add to it only when a future algorithm demands it.

### 3.2 `bots.py` — baselines and rollout policies

- `GreedyBaselineBot` — picks the highest-VP-yielding legal action; for non-build moves picks first legal. Hand-coded, no learning. *Purpose:* be a non-trivial opponent in Experiment 4, so we have *something* between "random" and "MCTS."
- `heuristic_rollout(state) -> action` — same logic, exposed as a callable for MCTS to use as a rollout policy in Experiment 3.

Depends on `pyspiel` types only — never reaches through to the engine. (This is what the adapter abstraction buys us.)

### 3.3 `recorder.py` — `SelfPlayRecorder`

Wraps a single game-play loop and records, per move, the schema in §1.5. Buffers in memory, flushes one parquet append on game end.

```python
with recorder.game(seed=..., config=...) as rec:
    while not state.is_terminal():
        ...
        rec.record_move(state, mcts_search_result)   # only after MCTS moves
        ...
    # recorder auto-finalizes on context exit, writes one row to games.parquet
```

Reads visit counts from MCTS's returned policy. Depends on `pyarrow` and the OpenSpiel MCTS result type. Not the engine.

Schema versioning: `SCHEMA_VERSION` constant in `recorder.py`, asserted in tests. Bumps explicitly when fields change.

### 3.4 `experiments/eN_*.py` — the four experiments

Each is a `main()` script taking `--seed`, `--num-games`, `--out-dir`, plus its own knobs.

- **e1 — `winrate_vs_random.py`.** Sweeps `sims_per_move ∈ {50, 200, 1000, 4000}`. For each, plays N games of MCTS-with-random-rollouts vs. 3 random opponents. Records win-rate and game-length distribution.
- **e2 — `ucb_c_sweep.py`.** Fixes sims/move at a moderate level (e.g. 200), sweeps the UCB exploration constant `c ∈ {0.5, 1.0, 1.4, 2.0, 4.0}`. Same opponent setup as e1.
- **e3 — `rollout_policy.py`.** Fixes c and sims/move. Compares MCTS-with-random-rollouts vs. MCTS-with-heuristic-rollouts. Plot win-rate vs. sims-budget for both. *The* classic MCTS lesson — heuristic rollouts often win at low budgets, lose at high budgets.
- **e4 — `tournament.py`.** Round-robin: MCTS (best config from e1-e3), GreedyBaselineBot, RandomBot, with all 4-seat permutations to control for seat advantage. Output: win-rate matrix.

`experiments/common.py` is a thin layer of shared seed-handling, output-directory creation, config-json dumping, parquet writer wiring, `tqdm` progress reporting. Not a framework.

### 3.5 `cli.py`

`python -m catan_mcts run e1 --num-games 200 --seed 42` and similar. Argparse → dispatch to the experiment's `main()`. The "single command runs everything" success criterion is `python -m catan_mcts run all`, which runs the four experiments with sensible defaults.

### 3.6 `notebooks/analysis.ipynb`

Read-only over `runs/`. Pandas-based. One section per experiment, producing the plots that go in `docs/writeup.md`. Notebook is committed; outputs stripped before commit (`nbstripout`).

### 3.7 Phase 0 — Engine-side prerequisites

These are small, contained changes to `catan_bot/catan_engine/` that this project requires. They live in this project's plan, not in v1's. v1 is closed.

1. **Expose dice rolls as chance points.** Some `engine.is_chance_pending()` / `engine.chance_outcomes()` queries plus `engine.apply_chance_outcome(value)`. The current internal roll path moves into the chance-outcome handler.
2. **Expose robber-steal target as a chance point** (currently auto-random). Same pattern.
3. **PyO3 `Engine.clone()`** returning an independent deep-copy `Engine`.
4. **(No new API needed for `serialize()`)** — adapter tracks action history Python-side.

### 3.8 Why Phase 0 is non-negotiable

Without chance nodes, MCTS treats dice as a player decision:

- The tree branches on dice outcomes as if the agent chose them, biasing search toward optimistic-luck paths.
- UCB statistics become meaningless as expected-return estimates. Experiment 2 (the `c` sweep) would measure noise.
- Experiment 3 (rollout policy) is more contaminated — rollouts also see deterministic dice in the search.
- Experiment 1 (win-rate vs. random) is least affected — at high enough sims/move, even biased MCTS beats random — but it's the only experiment that survives the bias.

Adding chance nodes later means rerunning every experiment and regenerating the self-play dataset (visit counts collected without chance nodes are bad GNN training targets). Doing it now is cheaper than doing it later.

## 4. Data flow

A single game from "experiment script starts" to "row in parquet":

```
experiment script
  ├─ creates CatanGame(seed)
  ├─ creates MCTSBot(game, sims=N, c=...) and opponents
  ├─ creates SelfPlayRecorder(out_dir, config)
  └─ loop:
        state = game.new_initial_state()
        with recorder.game(seed) as rec:
            while not state.is_terminal():
                if state.is_chance_node():
                    outcome = sample(state.chance_outcomes())   # dice / steal
                    state.apply_action(outcome)                  # not recorded
                else:
                    bot = bots[state.current_player()]
                    if bot is mcts_bot:
                        result = bot.step_with_policy(state)     # action, visits, root_value
                        rec.record_move(state, result)
                        state.apply_action(result.action)
                    else:
                        action = bot.step(state)                  # not recorded
                        state.apply_action(action)
            rec.finalize(returns=state.returns())
```

Notes:

- Only MCTS moves are recorded. Opponent moves are part of game state but not training targets.
- Chance-node steps are not recorded.
- Visit counts come from MCTS's per-move return, not estimated.
- One parquet append per game (in-memory buffering keeps row-group sizes sensible).
- Determinism end-to-end: same seed + same MCTS config + same opponent policies → same parquet bytes. Regression-tested.

There's no streaming, no parallelism, no shared mutable state. Experiments are embarrassingly parallel at the *game* level — if throughput becomes a problem later, run multiple processes with different seed ranges and concatenate parquets.

## 5. Error handling & failure modes

**Fail-fast** (let the exception kill the run):
- Any OpenSpiel contract violation (`apply_action` with illegal action, etc.). These are bugs in `adapter.py`, not runtime conditions.
- Engine panics propagated through PyO3 — we want the stack trace.
- Schema mismatch when reading parquet (the GNN project's loader asserts version).
- Determinism regression — if the regression test sees a different hash for a fixed seed/config, the run halts.

**Handle gracefully** (log + continue, but only at experiment-batch granularity):
- A single game in an N-game batch raises → log to `runs/<ts>/errors.log` (seed + traceback), increment failure counter, continue. A 2-hour run shouldn't die on game 873 of 1000.
- Parquet flush failure (disk full, permission) → halt the experiment, surface the error.

**Never silently degrade:**
- MCTS returns no visit counts → raise.
- Recorded `legal_action_mask[action_taken]` is False → raise (adapter and recorder disagree about legality).

**Reproducibility guard.** Every run dumps `config.json` with engine git SHA, OpenSpiel version, MCTS config, seed range, schema version. First thing to check when two runs disagree.

No retry logic, no fallback paths, no defensive `try/except` around experiment loops — those hide bugs we want to find.

## 6. Testing strategy

Scaled to risk. Heavy on the adapter, light on the experiments.

### 6.1 Adapter tests (`tests/test_adapter.py`) — load-bearing

- OpenSpiel contract: `legal_actions()` non-empty unless terminal; `apply_action(a)` for `a` in `legal_actions()` never raises; `clone()` produces an independent state (mutations don't leak).
- Chance-node contract: `is_chance_node()` is True exactly when the engine is at `Roll` or steal-target points; `chance_outcomes()` probabilities sum to 1.0.
- Round-trip: `state.serialize()` → reconstruct → same `legal_actions()`, same `is_terminal()`.
- Property test (`hypothesis`, ~256 cases): random legal-action sequences from initial state never raise.

### 6.2 Recorder tests (`tests/test_recorder.py`)

- Schema: written parquet has all expected columns with expected dtypes.
- Round-trip: write 10 fake games, read back, row count matches.
- `SCHEMA_VERSION` asserted in test (a bump can't slip in unnoticed).

### 6.3 Smoke test (`tests/test_smoke_mcts_vs_random.py`)

- One full game: MCTS (50 sims) vs. 3 random bots, fixed seed, completes without exception, terminal returns sum to 0 (one winner).

### 6.4 Determinism regression (`tests/test_determinism.py`)

- Fixed seed + fixed MCTS config + fixed opponent seeds → byte-identical `moves.parquet`. One stored hash. Catches both engine-side and adapter-side non-determinism.

### 6.5 What we deliberately don't test

- The experiments. They're scripts; the notebook is the truth. Testing "does e1 produce a plot" is unit-testing a deliverable.
- OpenSpiel's MCTS itself.
- Plot rendering / writeup content.

### 6.6 TDD discipline

- **Adapter:** tests-first. Writing the OpenSpiel contract as tests *is* the design clarification step — most adapter bugs are misunderstanding what `chance_outcomes()` should return, and tests force clarity.
- **Recorder:** tests-first.
- **Experiments:** no — exploratory by nature.

## 7. Versioning commitments

- **`moves.parquet` schema:** versioned. New fields bump the version; the GNN project's loader asserts.
- **`games.parquet` schema:** same.
- **Engine chance-node contract** (introduced in Phase 0): stable from this project's start. *When* a chance node fires (which engine phases) is part of the contract — adding new chance points later is a breaking change for the dataset and requires a re-run.

## 8. Open items deferred to implementation

- Exact OpenSpiel version pin (and `pyarrow` / `pandas` pins, since they touch the parquet schema).
- Exact `chance_outcomes()` distribution for the steal — uniform over opponents-with-cards is probably right but worth verifying against rules.
- Whether `mcts_root_value` is computed by OpenSpiel directly or has to be derived from root child stats.
- Whether `tqdm` progress is per-game or per-experiment-step.
