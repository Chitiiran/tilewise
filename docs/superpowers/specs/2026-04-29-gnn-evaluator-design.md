# GNN-v0 Evaluator — Design Spec

**Date:** 2026-04-29
**Status:** Draft — awaiting user review
**Project:** `catan_bot/mcts_study/` — second project on the v1 engine
**Builds on:** [`2026-04-27-mcts-study-design.md`](2026-04-27-mcts-study-design.md), MCTS-v2 hardening (branch `mcts-v2-hardening`)

## 1. Goal

Build a Graph Neural Network that replaces the random-rollout and lookahead-VP evaluators in OpenSpiel's `MCTSBot`, trained AlphaZero-style on the existing self-play parquets.

The deliverable is **the pipeline and the lessons** — what graph encoding, what model size, what loss weights, what training data sizes — not a strong bot. Bot strength is a measurement output.

### Non-goals

- Beating Catanatron's hand-tuned `AlphaBetaPlayer`.
- The full AlphaZero iteration loop (self-play → retrain → self-play). v0 trains once; iteration is a follow-on spec.
- Custom MCTS — we keep using `open_spiel.python.algorithms.mcts.MCTSBot`.
- Engine changes beyond a recorder schema bump.

### Success criteria

1. **Pipeline:** PyTorch + PyG model trains end-to-end on existing `runs/*/moves.parquet` and produces a checkpoint.
2. **Integration:** `GnnEvaluator` plugs into `MCTSBot` exactly the way `LookaheadVpEvaluator` does. No MCTS code changes.
3. **Strength bar (v0 pass):** `MCTSBot(GnnEvaluator)` at sims=400 reaches ≥60% winrate vs. 3 random opponents. At sims=100, ≥30%.
4. **Reproducibility:** every checkpoint carries `config.json` with hyperparameters, dataset run-dirs, git SHA. Loss curves are per-epoch JSON.
5. **Writeup-ready:** the four benchmarks (Bench-1 through Bench-4) produce numeric outputs that can be tabulated against `mcts-v2-hardening`'s e5 baseline.

## 2. Architecture overview

```
                    ┌─────────────────────────────────────┐
                    │   PyG HeteroData                    │
                    │   ├─ hex     [19,  F_HEX=8]         │
   engine.observation()  ├─ vertex  [54,  F_VERT=7]       │
                    │   ├─ edge    [72,  F_EDGE=6]        │
                    │   └─ scalars [22]                   │
                    │   + edge_index dicts for each type  │
                    └──────────────┬──────────────────────┘
                                   ▼
                  ┌─────────────────────────────────────┐
                  │  GnnBody (heterogeneous PyG)        │
                  │  - input projection per node type   │
                  │  - 2 × HeteroConv(hidden_dim=32)    │
                  │  - mean-pool over each node type    │
                  │  - concat with scalars              │
                  │  → embedding [128]                  │
                  └─────────┬─────────────┬─────────────┘
                            ▼             ▼
                   ┌─────────────┐  ┌─────────────────┐
                   │ ValueHead   │  │ PolicyHead      │
                   │ MLP→[4]     │  │ MLP→[206]       │
                   │ tanh→[-1,1] │  │ + legal mask    │
                   └─────────────┘  └─────────────────┘
```

**Three components, three files:**
- `state_to_pyg.py` — pure function: engine observation dict → `HeteroData`. No model.
- `gnn_model.py` — the `nn.Module`. Body + two heads.
- `gnn_evaluator.py` — wraps a trained model in OpenSpiel's `Evaluator`.

**Model size at v0:** hidden_dim=32, 2 message-passing layers, ~30-60k params. Latency target ≤5ms per `evaluate()` on CPU. Scale up (hidden_dim=64, 4 layers) only on evidence of saturation.

## 3. Heterogeneous graph wiring

**Node types (counts fixed by Catan geometry):**

| Node type | Count | Features per node |
|---|---:|---|
| `hex` | 19 | 8 (resource one-hot[5], dice-norm, robber, desert) |
| `vertex` | 54 | 7 (empty, settle, city, owner-perspective[4]) |
| `edge` | 72 | 6 (empty, road, owner-perspective[4]) |

Features come straight from the engine's existing `observation()`, no Python-side feature engineering.

**Edge types (only 2):**
- `hex ↔ vertex` — vertex-touches-hex adjacency. Each vertex has 1-3 hexes.
- `vertex ↔ edge` — vertex-connects-edge adjacency. Each edge has exactly 2 vertices.

We deliberately **do not** add `vertex ↔ vertex` or `hex ↔ hex` edge types — vertex-to-vertex info already flows via `vertex → edge → vertex` in two layers.

Adjacency tables are static. They live in the engine's `board.rs` (`vertices_of_hex`, `edges_of_vertex`); we expose them once via PyO3 (or hardcode in `state_to_pyg.py`) and reuse for every game.

**Scalars (22 floats)** — VPs, hand sizes, turn, phase one-hot — are **not** part of the graph. They concatenate to the pooled graph embedding before the heads.

**Forward-pass shapes (B = batch size):**

```
hex     [B, 19, 8]    vertex  [B, 54, 7]    edge    [B, 72, 6]
scalars [B, 22]
        │
        ├── per-type linear projection to 32 dim
        ▼
hex   [B, 19, 32]   vertex [B, 54, 32]   edge   [B, 72, 32]
        │
        ▼   HeteroConv layer 1 (one round of message passing)
hex   [B, 19, 32]   vertex [B, 54, 32]   edge   [B, 72, 32]
        │
        ▼   HeteroConv layer 2
hex   [B, 19, 32]   vertex [B, 54, 32]   edge   [B, 72, 32]
        │
        ├── mean-pool over each node type → 3 × [B, 32]
        ├── concat with scalars            → [B, 32+32+32+22] = [B, 118]
        └── final linear projection        → [B, 128]
        ▼
embedding [B, 128]
        │
        ├── ValueHead:  Linear(128→64) → ReLU → Linear(64→4) → tanh   = [B, 4]
        └── PolicyHead: Linear(128→206) → mask illegal → log_softmax  = [B, 206]
```

## 4. Training loop & loss

### Data source

Existing `runs/*-e*_*/worker*/moves.*.parquet` (per-MCTS-move) and `games.*.parquet` (per-game terminal). Each `moves` row → one labeled training position.

**Training targets per position:**

| Source field | Becomes |
|---|---|
| `mcts_visit_counts[206]` | **policy target** = visit_counts / sum (normalized to a distribution). Illegal actions are 0 by construction. |
| `games.winner` (joined by seed) | **value target** = 4-vector. Engine `observation()` is rotated to the *current player's* perspective, so the value target uses the same rotation: index `i` of value target = `+1` if `(current_player + i) % 4 == winner` else `-1`. `[0, 0, 0, 0]` if `winner == -1`. |
| `legal_action_mask[206]` | mask for the policy head; illegal logits → `-inf` before softmax. |

### Replay step

Parquet doesn't store the full state, only the seed and move index. To get features at move `i` of game `seed`:
1. Open a fresh `Engine(seed)`.
2. Replay the recorded action history up to move `i`.
3. Call `engine.observation()` → graph features.

**Pre-requisite recorder change (separate small PR before GNN training):** bump `SCHEMA_VERSION` from 1 to 2 and add an `action_history: list<int32>` column to `games.parquet`. Without this, replay needs to re-run `play_one_game` from scratch every `__getitem__`, which is too slow.

The recorder PR is one file change in `recorder.py` plus a v2 column in the schema; ~10 lines. Spec assumes it's in place when GNN-v0 training kicks off.

### Loss

```
L = w_v · L_value  +  w_p · L_policy

L_value  = MSE(value_pred, value_target)
L_policy = cross_entropy(policy_logits, policy_target)   # masked: illegal logits → -inf
```

Default `w_v = 1.0`, `w_p = 1.0` (AlphaZero original). Loss-weight sweep (`w_v / w_p ∈ {0.25, 1.0, 4.0}`) is one of the v0 experiments.

### Hyperparameters

| Knob | Default | Notes |
|---|---|---|
| Optimizer | Adam | lr=1e-3 |
| Batch size | 64 | small model, small dataset |
| Epochs | 20 | early stopping on val loss |
| Train/val split | **90/10 by seed** | whole games go to one side; prevents same-game leakage |
| Augmentation | none | board-rotation symmetry deferred to follow-on |
| Device | CPU | matches MCTS deployment; GPU is for future scale-up |

### Per-run output

```
gnn_v0/
├── checkpoint.pt        # final model weights
├── training_log.json    # loss curves per epoch
└── config.json          # hyperparams, run dirs, dataset size, git SHA
```

## 5. Evaluator integration

`GnnEvaluator` slots into `MCTSBot` exactly like `LookaheadVpEvaluator`. Same interface, no MCTS code changes.

```python
class GnnEvaluator(os_mcts.Evaluator):
    def __init__(self, model: GnnModel, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device
        self._cache_key = None
        self._cache_value = None
        self._cache_policy = None

    @torch.no_grad()
    def _forward(self, state):
        # Cache by engine action_history (deterministic state ID).
        # MCTS calls evaluate() and prior() back-to-back on the same state.
        key = tuple(state._engine.action_history())
        if key == self._cache_key:
            return self._cache_value, self._cache_policy
        obs = state._engine.observation()
        data = state_to_pyg(obs).to(self.device)
        v, logits = self.model(data)
        self._cache_key = key
        self._cache_value = v.squeeze(0).cpu().numpy()
        self._cache_policy = logits.squeeze(0).cpu().numpy()
        return self._cache_value, self._cache_policy

    def evaluate(self, state):
        v, _ = self._forward(state)
        return v   # np.ndarray shape [4], float32

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        _, logits = self._forward(state)
        legal = state.legal_actions(state.current_player())
        legal_logits = logits[legal]
        probs = scipy.special.softmax(legal_logits)
        return [(int(a), float(p)) for a, p in zip(legal, probs)]
```

**Wiring example:**

```python
# was: evaluator = LookaheadVpEvaluator(depth=25, base_seed=seed)
evaluator = GnnEvaluator(load_checkpoint("gnn_v0/checkpoint.pt"))
mcts_bot  = os_mcts.MCTSBot(game=game, uct_c=1.4, max_simulations=400,
                            evaluator=evaluator, ...)
```

**Out of scope for v0:** batched leaf evaluation. OpenSpiel's `MCTSBot` calls `evaluate` one leaf at a time; at sims=400 that's 400 sequential forward passes per move. If this becomes the bottleneck (likely at sims≥1000 with a larger GNN), the fix is a custom batched search loop. Flag for the writeup; defer the implementation.

## 6. Training data prep

`mcts_study/catan_gnn/dataset.py`:

```python
class CatanReplayDataset(torch.utils.data.Dataset):
    def __init__(self, run_dirs: list[Path]):
        # Glob all moves.*.parquet across run_dirs.
        # Filter rows where games.parquet has schema_version >= 2 (action_history present).
        # Build index list: (seed, move_index, parquet_row).
        # Build seed→(winner, action_history) map from games.parquet.
        ...

    def __len__(self): return len(self._index)

    def __getitem__(self, i):
        seed, move_index, row = self._index[i]
        # Replay engine to position. action_history records EVERY engine action
        # (chance and decisions, in order). move_index counts only MCTS-decided
        # moves. Walk action_history forward, counting MCTS decisions, until we
        # hit move_index — that's the prefix length to apply.
        full_history = self._action_history[seed]
        prefix_end = self._mcts_move_to_history_index(full_history, move_index)
        engine = Engine(seed)
        for action_id in full_history[:prefix_end]:
            if action_id & 0x8000_0000:  # chance bit
                engine.apply_chance_outcome(action_id & 0x7FFF_FFFF)
            else:
                engine.step(action_id)
        # Build inputs + targets, return (HeteroData, value_target, policy_target, legal_mask).
        ...
```

Globs `runs/*-e*_*/worker*/moves.*.parquet` and matching `games.*.parquet` at construction, concatenates, indexes. No state cache — replay from scratch each `__getitem__`. Engine release-mode replay is ~1ms per game; bottleneck during training will be the GNN forward pass, not replay.

## 7. Benchmarks & evaluation

Four benchmarks, run in this order. Each produces numeric output the writeup cites.

**Bench-1: training curves** (free, ~minutes per training run)

Per epoch in `gnn_v0/training_log.json`:
- `train_loss_total`, `train_loss_value`, `train_loss_policy`
- `val_loss_total`, `val_loss_value`, `val_loss_policy`
- `val_value_mae` — raw MAE on the value head, easier to read than MSE.
- `val_policy_top1_acc` — fraction of val moves where argmax(policy) matches `action_taken`.

**Pass:** validation loss decreases for ≥5 consecutive epochs and ends below 80% of epoch-1.

**Bench-2: static-position match against lookahead-VP** (~1 min)

Sample 1,000 val positions. For each:
1. Replay engine.
2. Run `LookaheadVpEvaluator(depth=25)` → reference value.
3. Run `GnnEvaluator` → predicted value + policy.
4. Record per-position MAE (value) and KL-divergence (policy vs visit-counts).

Outputs: `bench2_value_mae`, `bench2_policy_kl`. Lets us compare two GNN configurations apples-to-apples without spinning up MCTS.

**Bench-3: MCTS-with-GNN winrate (the headline)** (~30-60 min)

New experiment file: `mcts_study/catan_mcts/experiments/e6_mcts_gnn_winrate.py`. Mirrors e5's structure:

| Evaluator | Sims | Games | Notes |
|---|---:|---:|---|
| `LookaheadVpEvaluator(depth=25)` | 400 | 50 | baseline (e5 v2 best cell) |
| `GnnEvaluator(model=v0)` | 400 | 50 | the headline |
| `GnnEvaluator(model=v0)` | 100 | 50 | does GNN buy strength at lower sim budget? |

3 random opponents per seat. Standard e5 plumbing (workers, max_seconds, checkpoints, done.txt).

**Pass:** GNN-v0 at sims=400 ≥60% winrate vs. random; at sims=100 ≥30%.

**Bench-4: head-to-head vs lookahead-VP** (optional, ~30 min)

2-player MCTS tournament: `MCTSBot(GnnEvaluator)` vs `MCTSBot(LookaheadVpEvaluator)`, both at sims=400. Two random opponents fill the other seats. 25 games × 2 rotations = 50 games. Measurement, not pass/fail.

**Bench outputs:**
- `gnn_v0/bench1.json`, `gnn_v0/bench2.json` — auto-generated from training and a static-eval script.
- `runs/<ts>-e6_*/` — full self-play parquets, same schema as e5.
- `gnn_v0_summary.md` — short notebook-generated artifact for the writeup.

## 8. Component & file layout

```
mcts_study/
├── catan_gnn/                    # new package
│   ├── __init__.py
│   ├── state_to_pyg.py           # observation → HeteroData
│   ├── gnn_model.py              # GnnBody + ValueHead + PolicyHead
│   ├── dataset.py                # CatanReplayDataset
│   ├── train.py                  # training loop + checkpointing
│   └── benchmark.py              # bench-1, bench-2 entry points
├── catan_mcts/
│   ├── gnn_evaluator.py          # NEW: OpenSpiel evaluator wrapper
│   ├── recorder.py               # MODIFIED: SCHEMA_VERSION=2, add action_history
│   └── experiments/
│       └── e6_mcts_gnn_winrate.py  # NEW: bench-3 experiment
└── tests/
    ├── test_state_to_pyg.py
    ├── test_gnn_model.py
    ├── test_dataset.py
    ├── test_gnn_evaluator.py
    └── test_e6_runs.py            # e6 smoke test
```

Each file has one clear responsibility. Trained model weights and bench outputs land in a sibling `runs/gnn_v0/` (same parent as the MCTS run dirs).

## 9. Study experiments (planned, post-v0)

Once v0 passes its benchmarks, these are the comparable experiments — one knob varied at a time, baseline preserved.

| ID | Knob | Values | What it tells us |
|----|---|---|---|
| g1 | hidden_dim | 16 / 32 / 64 / 128 | Width sensitivity |
| g2 | layers | 1 / 2 / 4 / 8 | Depth / oversmoothing |
| g3 | loss weights `w_v / w_p` | 0.25 / 1.0 / 4.0 | Value-vs-policy tradeoff |
| g4 | dataset size | 1k / 5k / 20k positions | Sample-efficiency curve |
| g5 | encoding | hex+vertex+edge / vertex-only / vertex+edge | Encoding ablation |

Each experiment is one training run + one bench-1/2/3 pass. Sequence is g1, g2, g3, g4, g5 — width and depth before sample size before encoding ablation.

These are out of scope for the v0 spec; they become the next plan after v0 ships and we know the baseline numbers.

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Existing parquet has SCHEMA_VERSION=1 (no action_history). | Bump recorder, run a quick re-recording sweep with SCHEMA_VERSION=2 before training. ~1 hour at depth=25/sims=400 for 100 games. |
| GNN forward pass exceeds 5ms on CPU. | Drop `hidden_dim` to 16, drop layers to 1, or quantize. Width is the lowest-cost lever. |
| Sequential `evaluate()` calls dominate MCTS wall-clock at sims=1000+. | Out of scope for v0. Custom batched search is a follow-on. |
| `mcts_visit_counts` distributions are very peaked (one action gets 90% of visits). | Use a temperature on the policy target during training (e.g. `target = visits ** 0.5 / sum`). Becomes a knob for g3. |
| PyG `HeteroConv` batching has sharp edges. | Test thoroughly in `test_gnn_model.py` with batch sizes 1, 4, 64. |

## 11. Open questions (deferred, not blockers)

- **Symmetry augmentation:** the Catan board has 6-fold rotational + 1 reflection = 12 symmetries. Augmenting training data 12× is "free" data. Becomes g6 once g1-g5 are done.
- **Iteration loop (true AlphaZero):** train v0 → generate fresh self-play with `MCTSBot(GnnEvaluator)` → retrain. Becomes its own spec once v0 numbers are in hand.
- **Curriculum:** train value head first (cheap, signal-rich), then add policy. Could improve early-epoch stability. Defer; not load-bearing.
