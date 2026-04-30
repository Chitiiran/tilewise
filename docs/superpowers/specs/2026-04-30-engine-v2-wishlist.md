# Engine v2 — wishlist

**Status:** deferred. Captured 2026-04-30 while building the replay viewer's all-hands display. None of these are implemented in v1; they are the batched scope for the next engine pass.

**Why batch:** every engine change requires `cargo build` + `maturin develop`, which touches the Python venv and is unsafe while a long-running training process is using the same package. Doing them as one PR amortizes the cost and avoids interrupting active runs.

---

## 1. Rule additions (gameplay)

### 1a. Building inventory caps

Catan rules cap each player's lifetime building inventory:

- 5 settlements
- 4 cities
- 15 roads

The v1 engine **does not enforce these.** Filter `legal_actions()` accordingly:

- `BuildSettlement(v)` only legal if `state.players[p].settlements_built < 5`
- `BuildCity(v)` only legal if `state.players[p].cities_built < 4`
- `BuildRoad(e)` only legal if `state.players[p].roads_built < 15`

Counts already exist in `GameStats.players[p].{settlements,cities,roads}_built`. ~10 lines of Rust in `legal_actions()`.

**Impact:** changes the strategic problem (capped supply forces upgrades, longest road, etc.). Existing v1/v2 GNN training data is **invalidated** for v2-rules training.

### 1b. Longest Road tracking + VP

- Maintain a per-player `longest_road_length` recomputed when a `BuildRoad` event fires (or when a road is "broken" by an opponent's settlement on it — this is the tricky case).
- Award +2 VP to the player with the *strict* current max ≥ 5 roads. If two players tie, neither gets it. If a third player's road grows past the holder's, the title transfers and VP swings ±2.
- Add `state.longest_road_holder: Option<u8>` and `state.players[p].longest_road_length: u32`.
- Surface in `GameStats` and `engine.stats()`.

**Algorithmic note:** longest road on a graph with cycles is NP-hard in general but trivial on a Catan-sized graph (≤72 edges). DFS from each owned road endpoint is fine. Test against known patterns.

### 1c. Largest Army (deferred until dev cards)

Punted to Tier-2 because it requires dev cards. Mentioned here only so it's not forgotten.

---

## 2. Replay / inspection APIs (no rule change)

These don't change game behavior — they expose information that the engine already tracks but doesn't surface to Python. Required to build proper replay/inspection tooling without re-implementing engine state in Python.

### 2a. `engine.bank() -> [u8; 5]`

Returns the bank's remaining resources `[wood, brick, sheep, wheat, ore]`. Currently must be derived as `19 - sum(all_hands)` from the Python side, which requires knowing all hands.

**Implementation:** the bank is `state.bank_resources` already; just expose via PyO3.

### 2b. `engine.all_hands() -> [[u8; 5]; 4]`

Returns all 4 players' resource hands in **absolute seat order** (NOT perspective-rotated). Currently `engine.observation()` only reveals the current player's breakdown plus opponents' totals.

**Why needed:** every replay/visualization tool wants this. Reconstructing it in Python from action_history is correct but fragile (every chance-action encoding edge case becomes ours to maintain).

**Implementation:** ~5 lines of PyO3 wrapping `state.hands`.

### 2c. `engine.observation_for(viewer: u8)`

Like `engine.observation()` but takes an explicit viewer arg, so we can get any player's perspective on demand. Useful for:
- Multi-perspective training data (4× the effective signal per game)
- Per-player VP / position evaluation in tooling

**Implementation:** the perspective-rotation already happens internally; just thread the viewer arg through instead of using `state.current_player`.

### 2d. Missing `engine.stats()` PyO3 fields

The Rust `GameStats` already tracks these but the PyO3 binding (`lib.rs::stats()`) doesn't surface them:

- `resources_gained: [u32; 5]` per player
- `resources_gained_from_robber: [u32; 5]` per player
- `resources_lost_to_robber: [u32; 5]` per player
- `resources_lost_to_discard: [u32; 5]` per player

Replay summary panels want these (e.g., "P0 produced 23 wood, lost 3 sheep to robber"). Already in the struct (`stats.rs:18-21`); just wrap.

---

## 3. Plug-and-play replacement plan

Today the replay viewer (`scratch_replay_lite.py`) reconstructs all-player hands and bank in Python. That work goes through a `CardTracker` interface:

```python
class CardTracker:
    def snapshot_at(self, step_idx: int) -> {"hands": [[5]; 4], "bank": [5]}: ...
```

Two implementations:

- **Today (`PythonCardTracker`):** replays action_history with a Python-side state machine. Self-contained, no engine changes.
- **After v2 engine (`EngineCardTracker`):** thin wrapper around `engine.all_hands()` + `engine.bank()`. Drop-in replacement.

Switching is a one-line import change in the viewer.

---

## 4. Migration consequences

**v1/v2 GNN training data invalidation.** Once §1a (inventory caps) lands, the engine plays a different game. All existing parquets become stale for training purposes. Plan a fresh data sweep after the v2 engine merges.

**Stats schema bump.** §1b adds `longest_road_holder` and per-player `longest_road_length`. §2d exposes existing fields. Bump `STATS_SCHEMA_VERSION` from 1 → 2.

**Action space stable.** None of these add or remove actions; the 206-action space stays. GNN architectures don't need retraining for the v2 engine *if* we hold off on §1a (inventory caps) — but the model trained on v1-rules data won't transfer well to v2-rules play because reachable states differ.

---

## 4a. Balanced map generation

**Why:** Today the engine generates the board using its seed-driven RNG with no fairness guarantees, so two seeds can produce wildly different competitive boards (red numbers clustered, resource monocultures, etc.). Real Catan play uses a balanced procedure. To make our trained models comparable to humans — and to give MCTS / GNN a board distribution that matches the "real game" — we should ship a balanced generator.

### What "balanced" means in real Catan

The only **hard rule in the official rulebook** is: **no two red numbers (6, 8) may be adjacent.** Everything else is community convention layered on top.

### Two implementation tiers

**Tier 1 — official "ABC" method (recommended for v2):**

The standard 19-hex Catan board has number tokens labeled A through R on the back. The official balanced procedure:

1. Randomize hex **resources** (the 19-hex shuffle: 4 wood, 3 brick, 4 sheep, 4 wheat, 3 ore, 1 desert).
2. Place number tokens **in fixed alphabetical order** starting from a corner hex, spiraling counterclockwise inward. Skip the desert.
3. The token sequence is fixed by Klaus Teuber's design: `A=5, B=2, C=6, D=3, E=8, F=10, G=9, H=12, I=11, J=4, K=8, L=10, M=9, N=4, O=5, P=6, Q=3, R=11`.

The ABC sequence is constructed so that **no two red numbers (6, 8) are ever adjacent by construction** — it's not a separate constraint, it falls out of the spiral order. Same for "low" reds (2, 12).

This is what Colonist.io uses (per their blog: "the original ABC rule/number distribution"). It's also what the official rulebook recommends as the standard balanced setup.

**Implementation cost:** ~30 lines of Rust. Adds:
- `BoardLayout::generate_abc(rng) -> Self` — shuffle resources, place tokens via fixed ABC sequence on the canonical spiral.
- A way to expose the spiral path (corner-start choice, 6 corner options that all give equivalent boards by symmetry — pick deterministically e.g. always corner 0).

**Tier 2 — rejection sampling with multi-criteria scoring (deferred):**

What community generators (catanboard.com, etc.) do: generate random hex+token placements, score against constraints, keep the best of N tries (typical N = 100–1000). Constraints stack:

- No adjacent 6/8 (the official rule)
- No adjacent 2/12 (the symmetric "low red" rule)
- No same-resource clusters (e.g. 3+ wheat touching)
- No same-number adjacency (two 9s never touch)
- Production-pip-sum balance across regions (no corner-of-the-board has 2× the dot-density of another)

This produces **diverse-but-fair boards**, which is what the user wants for training data augmentation — the ABC method gives the same board topology for every game (only resource permutation changes), so it's not a great data-diversity source.

**Cost:** ~150 lines Rust. Defer until we have a concrete need (e.g. "v3 model training data needs board diversity beyond resource permutation").

### What to build for v2

Just Tier 1. It's:
- the canonical "fair Catan board" everyone trains and plays on,
- a small, well-defined Rust addition,
- enough to give MCTS / GNN a realistic board distribution.

Tier 2 lives in this wishlist for v3+.

### Notes from research (2026-04-30 web search)

- **Official Catan rulebook (Klaus Teuber, 2020):** the rule book offers two setup modes — a fixed beginner board, and a randomized board with the ABC procedure. The rulebook also offers "fully random with no-adjacent-reds enforcement" as a fallback for players who don't want the spiral.
- **Colonist.io:** uses ABC for maps; their published algorithm work is on **dice fairness** (a 36-card "Dice Deck" with reshuffle at 12 cards remaining, plus 30% repeat-roll suppression — orthogonal to map balance).
- **catanboard.com / community generators:** layer Tier 2 constraints, often evaluating ~600 candidates and returning the best by score. Not what the official game does.
- **Important:** the ABC method's "balance" is a property of the **token sequence + spiral path together**. Don't shuffle the tokens; only shuffle the resources. If we randomize tokens we lose the no-adjacent-reds guarantee and have to fall back to Tier 2 rejection.

### Wishlist entries this enables

- `engine.board_seed` and `engine.balanced_layout(method = "abc" | "fixed")` selector — current engine should keep its raw-random mode for backward compat with existing parquets, but new training data should use ABC.
- A separate config field on the recorder (`board_method: "raw" | "abc" | "fixed"`) so we can later filter parquets by board method.

---

## 4b. Training pipeline improvements (separate from engine)

These aren't engine work but were promised here so they don't get lost. Apply on the next training run, not the in-flight one.

### 4b.1. Per-epoch checkpoint saving in `train.py`

Today `train.py` writes one checkpoint at the very end (`checkpoint.pt` after the final epoch). This forecloses on natural diagnostics like "compare GNN_e10 vs GNN_e20 vs GNN_e60" — we can only ever evaluate the fully-trained model.

**Fix:** save `checkpoint_epoch{N:02d}.pt` at a configurable cadence (e.g. every 5 epochs, or on a list like `[10, 20, 30, 60]`). Keep the final `checkpoint.pt` symlink for backward compat with downstream scripts.

**Cost:** ~67 KB per checkpoint × 12 saves = trivial disk. No training-time cost.

**Why this matters:** with v1's policy collapse (always same settlement spot), the right diagnostic is "did the model overfit, and at which epoch was it actually best?" — which requires comparing intermediate snapshots. Without this we have to guess and re-run.

### 4b.2. Resume-from-checkpoint in `train.py`

Today killing a training run loses everything. The cache survives; the model state doesn't. Add a `--resume <path>` flag that loads weights + optimizer state and continues from the recorded epoch.

Lower priority than 4b.1 — useful but only when iterating.

### 4b.3. Two-GNN tournament mode in `e7_gnn_tournament.py`

Today e7 plays one GNN checkpoint against PureGnn / LookaheadMcts / Random. Add a mode that plays **two GNN checkpoints head-to-head** (e.g. GnnMcts(ckpt_a) vs GnnMcts(ckpt_b) vs LookaheadMcts vs Random). This is what we'd use to answer "did epoch 10 or epoch 20 play better?" once 4b.1 lands.

### 4b.4. Best-epoch tracking + `checkpoint_best.pt`

Track `best_val_top1` (or `best_val_loss`, configurable) across epochs and write `checkpoint_best.pt` whenever a new best lands. Means even on long noisy runs we keep the actually-best snapshot, not just the final one.

Pairs with 4b.1: together they solve the "we trained to ep60 but the real best was ep7" problem the v2_d25_w0w1 run hit.

**Implementation:** ~15 lines in the epoch loop. Track `best_metric` and `best_epoch`; copy state_dict to disk when current epoch beats it.

### 4b.5. Per-game val variance reporting

The 20-game val set gives noisy val_top1 swings (40% → 61% → 36%) because the *effective sample size* is the 20 game seeds, not 50k positions. Compute and log per-seed val_top1 alongside the aggregate mean: report mean, stdev, and quartiles. Then we can tell at a glance whether "ep7 = 0.611" is a real signal or "1 of the 20 val games happened to fit well this epoch."

**Implementation:** ~25 lines. Group val positions by seed, compute per-game top1, log min/p25/median/p75/max. Add to `EpochStats`.

### 4b.6. Round-robin tournament loop order

The current e8 implementation iterates `for perm in permutations: for game in range(N):` — every worker finishes all N games for permutation 0 before touching permutation 1. **Result:** any mid-run kill produces a single-permutation biased subset, not a balanced sample.

**Fix:** swap loop order. `for round in range(N): for perm in permutations:`. Each round produces `len(permutations)` games covering all seatings exactly once. After K rounds, you have K games per permutation — kill-safe and balanced.

Discovered the hard way during the 1200-game kill at 36 games (all from permutation 0). Adds maybe 30 lines of refactoring in `_run_cell` / `main`. Cheap and high-value.

### 4b.7. Throughput probe before committing to long tournaments

Before launching N=1000+ tournaments, run a small N=24 (one game per permutation) probe and measure wall-clock per game per worker. Extrapolate. Only commit if the projected total is acceptable. Don't trust extrapolation from 12-game runs that happened to hit short-game seeds.

Trivial — just a `--probe` flag that prints "estimated total wall-clock: X hours" and exits.

### 4b.8. Live progress plot during training

`train.py` writes a `progress.png` after every epoch (same plot as `scratch_plot_v2_d25_progress.py`, but baked in). Any session can see how training is going visually without parsing the log.

**Implementation:** ~30 lines. Maintain a list of per-epoch dicts; after each epoch, regen the plot. Saves to `<out_dir>/progress.png`.

---

## 5. Out of scope (for v2 engine)

- Dev cards (Knight, Year of Plenty, Monopoly, Road Building, VP)
- Largest Army (depends on dev cards)
- Trading between players (Tier-2 / Tier-3)
- 5-6 player variant
- Custom board layouts beyond standard 19-hex
