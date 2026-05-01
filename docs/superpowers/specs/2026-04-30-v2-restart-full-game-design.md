# v2 Restart — full Catan with bit-packed state

**Date:** 2026-04-30 (end of day-2)
**Status:** design pivot. Captured immediately after observing that no GNN we've trained beats LookaheadMcts at any sims/depth, and watching three "best" d25 self-play games to diagnose why.

---

## What we learned (the negative result)

We spent two days building the full pipeline (engine → MCTS → recorder → cache → GNN → tournament) on a Tier-1 (subset) Catan engine. Every metric came out the same way:

- **No GNN ever beat LookaheadMcts**, across 12-game, 36-game, and 9-game tournaments at sims=50/depth=25 *and* sims=9/depth=9.
- **GNN policies are statistically indistinguishable from each other** (v2_value vs v2_pw4: mean VP gap of 0.17, n=36 — within noise). Loss-weighting did nothing.
- **Random play is statistically competitive** at sims≤9 (44% LookaheadMcts vs 22% Random at d=9, s=9).
- **41% of d25 training games timed out** at the 30k-step engine cap → 41% of value targets are zeroed out. Massive data-quality hole we missed earlier.

**The pipeline works. The strategy signal it's learning is too thin to be useful.**

## Why — three structural problems with the v1 engine

Watching the three "best" finished d25 self-play games (seeds 41400315, 41400279, 41400295) made the diagnosis concrete:

### 1. Wood/Brick monopoly creates road-spam blockade

A player who lands on wood + brick hexes can build roads indefinitely (cost: 1 wood + 1 brick). They block opponents from expanding into open vertices, but **they don't actually win** because settlements/cities also need sheep/wheat/ore. The result: long stalled mid-games where the wood/brick player blocks expansion but can't cash in.

### 2. No mechanism to trade for missing resources

If you don't sit on a sheep tile, you literally cannot make a settlement (cost: 1W/1B/1S/1G). In real Catan, the player-to-player trade phase is the universal release valve — "I'll give you 2 wheat for 1 sheep." Our engine has **none** of that:
- No 4:1 maritime trade with the bank
- No 3:1 / 2:1 ports
- No player-to-player trades

So a resource-starved player just sits there forever, hoping the dice change.

### 3. Robber steal mechanism may not be transferring cards

We need to verify: when a player rolls a 7 → moves the robber → "steals" — does the stolen card actually arrive in the thief's hand and leave the victim's? If the engine logs the steal but doesn't move the card, that's a silent rules bug that could explain a lot of the imbalance. **First task on restart: write a unit test for the steal transfer.**

### 4. Missing rule features amplify items 1–3

Real Catan has natural pressure-release mechanisms that v1 lacks:
- **Longest road** (+2 VP for ≥5 connected roads, transferable): rewards the road-spammer instead of stalling them.
- **Largest army** (+2 VP for 3+ used Knights): rewards aggressive robber play.
- **Development cards**: knight (re-position robber), road building (place 2 free roads), monopoly (steal all of a resource), year of plenty (gain 2 from bank), VP cards.
- **Building inventory caps** (5 settlements, 4 cities, 15 roads): forces the road-spammer to cash in or stop.

**Without all of these, the game we trained on isn't really Catan.** It's a strategically degenerate subset where stalling is rewarded.

---

## The pivot: full Catan, bit-packed

### Why we can afford full Catan now

Original concern was state-space explosion: with dev cards, trades, longest road, etc., the per-state representation gets large, and we'd be storing millions of these in caches (already at 3.4 GB for 200 games). Multiplying state size by 3-5× hits the 31 GB WSL cap fast.

**Counter-argument from today's findings:**
- Most state values are tiny. Resource counts: 0–19 (5 bits each). VP: 0–10 (4 bits). Building counts: 0–15 (4 bits). Robber position: 0–18 (5 bits).
- The current observation uses **f32 floats** for everything (4 bytes per value). A bit-packed representation could be **8–32× smaller** without losing any information.
- Cache size grows with positions, not state-element count. **Bigger states, fewer training examples needed** because each example carries more strategic content (real trade signals, real dev-card play).

### Bit-packing scheme (sketch)

For each player's state in the v2 engine:

```
struct PlayerState {
    resources: [u5; 5],         // 25 bits — wood/brick/sheep/wheat/ore (0..19 each)
    dev_cards_played: [u3; 5],  // 15 bits — knight/yop/mono/rb/vp (0..7 each)
    dev_cards_held:   [u3; 5],  // 15 bits
    settlements_built: u3,       // 3 bits — 0..5
    cities_built:      u2,       // 2 bits — 0..4
    roads_built:       u4,       // 4 bits — 0..15
    longest_road_len:  u4,       // 4 bits — 0..15
    largest_army_size: u3,       // 3 bits — 0..7 typical
    has_longest_road:  u1,       // 1 bit
    has_largest_army:  u1,       // 1 bit
    vp_total:          u4,       // 4 bits — 0..10
}
// Total: ~77 bits per player → pack into 10 bytes
```

Compared to today's per-player state (5×f32 hand + 4×u8 totals + flags ≈ 32+ bytes), that's **~3× smaller** even for v1's smaller feature set. Adding dev cards, longest road, largest army actually *fits* in less space than the current f32 layout.

For board state (settlements/cities/roads/robber): same story — vertex owner is u3 (4 players + empty), edge owner is u3, robber hex is u5. Static board features (resource type, dice number, port type) are constants per seed and don't need to be in the per-state tensor at all.

### Where this goes

A v2 engine with:
1. **Full rules** — trades (4:1 bank, ports, player-to-player), longest road, largest army, all 5 dev card types, building inventory caps, 10-VP win condition with all VP sources.
2. **Bit-packed observation** — 50–80 bytes/state instead of today's ~3 KB/state. **40× cache compression.**
3. **Same Rust + PyO3 architecture** — but the observation builder unpacks bits → tensors only at inference time, not in the cache.

This is a substantial rewrite, but the v1 codebase has answered every "can the architecture work?" question. We know the GNN training pipeline closes end-to-end; we know MCTS works under chance; we know the recorder/replay/dataset chain is correct. **What we haven't shown is that the GNN learns anything useful, and we now believe that's because the game itself is too thin.**

---

## Improvement list (full restart scope)

Numbered for easy reference. Some are necessary for v2; others are "while we're at it."

### A. Engine — rules completeness

1. **Verify and fix robber-steal card transfer** — write a regression test that asserts `victim.hand[r] -= 1, thief.hand[r] += 1` after a steal, with the resource sampled from victim's actual cards.
2. **Building inventory caps** (5 settlements, 4 cities, 15 roads) in `legal_actions()`.
3. **Longest road tracking** — per-player `longest_road_length`, holder bit, +2 VP, transferable, recomputed when a road is built or broken by an opponent's settlement.
4. **Largest army tracking** — per-player `knights_played` count, holder bit, +2 VP, transferable.
5. **Maritime trades** — 4:1 with bank from any port-less position; 3:1 generic port; 2:1 specific-resource ports. Add as new actions; encode trade-give-resource × trade-get-resource = 25 actions (or fewer if we exclude same-for-same).
6. **Player-to-player trades** — proposal/accept/reject phase. This is the most complex new addition; will need careful action encoding (proposal action emits an offer; opponents have an accept/reject action). May want to defer to v3 if v2 scope grows too big.
7. **Development cards** — purchase action (1S/1G/1O), 5 types: knight (move robber + steal), road building (2 free roads), monopoly (claim all of one resource from all opponents), year of plenty (2 free resources from bank), VP card (silent +1 VP, revealed at win). Each has its own action encoding.
8. **Win condition expansion** — VP sources: settlements, cities, longest road, largest army, dev-card VPs. Game ends when total ≥ 10 on the active player's turn.

### A-bis. Engine — rules simplifications (deliberate v2 scope cuts)

These are choices to **make the v2 engine faster while still being playable**. We can revisit any of them later if the simplification turns out to hurt model quality.

8a. **Robber-7 discard: instant random discard, not turn-by-turn.** Today's engine has a multi-step Discard phase where each player owing a discard picks one card at a time, in turn order. For v2: when a 7 is rolled, **automatically discard `floor(hand/2)` random cards from each player owing one**, in a single step. No discard phase, no per-card decisions. **Rationale:** in real Catan the discarder picks which cards to dump, but at our depth=25 lookahead nobody is meaningfully optimizing this — the lookahead's greedy heuristic was just dumping cards in fixed priority order anyway. A random discard preserves the "lose half your cards" pressure without the per-card overhead. Saves ~5-15 moves per 7-roll game (and we average 3-5 sevens per game).

   **Impact on action space:** removes the `Discard(r)` actions (199-203) entirely, freeing 5 action slots and eliminating an entire game phase. Engine becomes simpler.

   **Future revisit:** if v3 wants strategic discard decisions, add it back as an explicit phase.

### B. Engine — state representation

9. **Bit-packed observation tensor** — design and ship the bit-layout above; verify size reduction empirically (target 50–80 bytes/state vs ~3 KB today).
10. **Static board cache** — board topology (vertex/edge/hex adjacency, resource types, number tokens, port types) doesn't change per game. Store ONCE per seed, not per state. Big cache size win.
11. **Per-state delta encoding (optional)** — many actions change one or two fields of the state; could store states as deltas from a base. Defer unless §9+§10 don't get us under target.

### B-bis. Engine — fast legality and pruned action space

The v1 engine recomputes `legal_actions()` from scratch on every call — linear scan of all 206 action IDs, each running its own rule check. At sims=400 × ~150 moves/game = 60,000 calls/game, this is **the dominant CPU cost in MCTS leaf exploration**. Adding dev cards + trades + ports would push action space past 250 and break this.

The fix is structural: **store legality as part of state, update incrementally, expose as a bitmap.**

11a. **`state.legal_mask: [u64; N]` updated incrementally on every state transition.** Each action's legality is a single bit; total mask is ~512 bits = 64 bytes. After a state mutation, only update the bits whose legality could have changed — typically a few dozen, not all 256.

11b. **Decompose action legality into affordability × position flags, AND at lookup.** For build actions, "is this legal?" factors into:
   - Can the current player afford the cost? (1 bit per build type)
   - Is this position structurally valid? (1 bit per vertex/edge for that build type)

   ```rust
   struct LegalCache {
       can_afford_settlement: bool,      // 1 bit, recomputed when hand changes
       can_afford_city: bool,
       can_afford_road: bool,
       can_afford_dev_card: bool,

       // Per-position flags, recomputed only when board topology changes:
       settlement_valid_pos: u64,        // 54 bits, distance rule + own roads
       city_valid_pos: u64,              // 54 bits, my settlement at v
       road_valid_pos: [u64; 2],         // 72 bits, connected to my network
   }

   fn legal_mask(&self) -> [u64; N] {
       let mut m = [0u64; N];
       if self.can_afford_settlement { m |= settlement_valid_pos; }
       if self.can_afford_city { m |= city_valid_pos << 54; }
       // ...
       m
   }
   ```

   **Power of this:** when the player buys a dev card (hand changes), we flip 4 affordability bits; the position flags don't move at all. When a settlement is built (board changes), only the affected position bits update; affordability is a single re-check.

11c. **Property test: `incremental_legal_mask == brute_force_legal_mask` after every state mutation.** The incremental updates are easy to forget; the test catches missed bits. Run on a random sequence of 1000+ actions.

11d. **PyO3 surface: `engine.legal_mask() -> np.ndarray[bool, N]`.** Replace `engine.legal_actions()` with the bitmap directly — Python uses `np.flatnonzero(mask)` to get the action ID list. The GNN's policy mask becomes a memcpy from this bitmap.

11e. **Action space layout: group by mutation-blast-radius.** New layout for v2 (with all rules):
   ```
   0–53:    BuildSettlement(v)            (54)
   54–107:  BuildCity(v)                  (54)
   108–179: BuildRoad(e)                  (72)
   180–198: MoveRobber(h)                 (19)
   199–203: BuyDevCard, EndTurn, Roll     (~5)
   204–223: PlayKnight(h), PlayRoadBuild, (20)
            PlayMonopoly(r), PlayYoP
   224–248: TradeBank(give×get)           (25)   ← maritime + ports
   249–...: TradeOffer(give[5], get[5])   (?)    ← player trade, deferred to phase 2
   ```

   Grouping by mutation type makes incremental updates cleaner: "hand changed → recheck affordability for build group" rather than "hand changed → scan all 256 actions."

11f. **Estimated speedup: 30–100× on `legal_mask()` lookup, ~30–40% reduction in MCTS leaf-eval wall-clock at sims=400.** Bigger payoff at higher sim counts. Single biggest engine perf win in v2.

11g. **Stable bit positions for future-proofing.** Reserve action ranges with gaps so adding new actions (e.g. v3 player trades) doesn't require renumbering existing bits. The cost is a few unused slots; the benefit is forward compatibility for trained models loaded with newer engine versions.

### C. Engine — APIs we already wishlisted (carry over from §2 of `2026-04-30-engine-v2-wishlist.md`)

12. **`engine.bank() -> [u8; 5]`** — expose bank resources directly. Replay tooling needs this.
13. **`engine.all_hands() -> [[u8; 5]; 4]`** — absolute-seat hands without perspective rotation.
14. **`engine.observation_for(viewer)`** — explicit-viewer perspective rotation. Unlocks 4× training data per game.
15. **Missing PyO3 stats fields** — surface `resources_gained`, `resources_lost_to_robber`, etc. that the Rust struct already tracks.

### D. Balanced map generation

16. **ABC method (Tier 1 from `engine-v2-wishlist` §4a)** — randomize hex resources, place tokens A→R in fixed order along the canonical spiral. Guarantees no-adjacent-reds by construction. ~30 lines of Rust. Match what Colonist.io and the official rulebook recommend.
17. **Rejection-sampling generator (Tier 2, deferred)** — only if §16's "ABC permutes only resources" lacks the diversity we want for training-data variety.

### E. Training pipeline (carry over from §4b)

18. **Per-epoch checkpoints + best tracking** — done in v1. Carry forward.
19. **Resume-from-checkpoint** — done. Carry forward.
20. **Per-game val variance reporting** — done. Carry forward.
21. **Live progress.png** — done. Carry forward.
22. **Two-GNN tournament mode (e8)** — done. Carry forward.
23. **Round-robin tournament loop (4b.6)** — outer round, inner permutation, so kills produce balanced data. **Add this BEFORE running another big tournament.**
24. **Throughput probe (4b.7)** — small N=24 probe before launching N=1000+ runs, project total wall-clock, only commit if acceptable.
25. **Filter timed-out games before caching** — `CatanReplayDataset` should drop rows where `winner == -1`. Today these become value=[0,0,0,0] training targets, which teach the model to predict null outcomes. Big quality lift, ~5 lines of code.

### F. Pipeline architecture for "doing this for real"

26. **AlphaZero iteration loop (open question §1 from v1 journal)** — train v1 → use v1 in self-play → train v2 on v1's data → train v3 on v2's data, etc. Single-shot supervised learning from a fixed teacher (lookahead-VP) plateaus at the teacher's strength. The whole AlphaZero idea is iterative self-play.
27. **Batched GNN evaluator (deferred Pipeline Limitation A)** — collect MCTS leaves across sims, batch the forward pass. ~50× speedup on GPU. Required to make the GNN evaluator run at sims=400+ in reasonable wall-clock.
28. **Multi-perspective training data** — once §14 lands, each game produces 4× the rows by recording all 4 players' perspectives, not just one. Unlocked.

### G. Data quality

29. **Investigate and fix the 41% timeout rate** — even before bit-packing or rule expansion, the v1 engine produces 41% timeouts at the 30k-step cap on d=25 self-play. Either the cap is wrong, the lookahead heuristic is degenerate at depth=25, or there's a real rule deadlock (item §1 above). Diagnostic: replay a random timeout game step-by-step.
30. **Diversify training opponents** — current data is all `LookaheadMcts vs LookaheadMcts vs LookaheadMcts vs LookaheadMcts`. The trained GNN has no exposure to weaker or differently-styled opponents. Consider mixing in Random + GreedyBaseline + LookaheadMcts at varied (depth, sims) to widen the strategic distribution.

### H. Replay tooling

31. **Port the `CardTracker` plug-and-play interface** — once §13 lands, swap `PythonCardTracker` for `EngineCardTracker` (one-line import change in viewer).
32. **Show longest road, largest army, dev cards in replay viewer** — once those exist in the engine. Currently the viewer just shows raw VP and buildings, which won't reflect new VP sources without an update.
33. **Show trade history in replay** — when player-to-player trades happen, show "P0 gave P2: 2 wheat for 1 sheep" in the narration column.

### J. Other engine performance wins (beyond legality)

37. **Eliminate `state.clone()` on the search hot path.** OpenSpiel's MCTSBot clones state on every node expansion via `state.clone()`; with chance steps the engine often clones tens of thousands of times per leaf. Investigate whether clone can be replaced with **explicit unmake-move** (push action to undo stack, pop to revert). Common technique in chess engines (Stockfish, etc.). Could 5–10× MCTS throughput.

38. **Avoid Python ↔ Rust boundary churn.** The journal already noted `tuple(action_history)` was O(N) per cache key call (fixed via `id(state)`). Same pattern likely lurks elsewhere — the GNN evaluator currently does one PyO3 call per leaf eval, plus separate calls for `observation()`, `legal_actions()`, etc. Consolidate into a single `engine.observe_and_evaluate()` PyO3 call returning a struct with everything the evaluator needs.

39. **Move `LookaheadVpEvaluator`'s greedy rollout entirely into Rust.** Currently `evaluate()` calls `engine.lookahead_vp_value()` from Python, which is a Rust function — but the evaluator class itself is Python. Eliminate the wrapper class; expose `engine.lookahead_value(depth, seed)` directly. Saves a Python frame per leaf.

40. **Rust-native chance sampling in MCTS.** OpenSpiel samples chance outcomes in Python (`state.chance_outcomes()` → list → weighted sample). Each sample is a list build + a random.choice. Could be a single Rust call: `engine.sample_chance(rng_seed) -> u32`.

### K. GNN architecture improvements

41. **Action-conditioned policy head.** Today the policy head outputs a length-N logit vector and we mask. With dev card actions and trade actions added, many positions will have only 1–3 legal actions out of 250+, and the policy head wastes most of its capacity. Investigate a **two-stage policy:** (1) head A outputs a coarse "what action category" (build / trade / play dev / end turn), (2) head B conditions on category and outputs the specific action. Could substantially improve sample efficiency.

42. **Dynamic mask within the GNN's loss.** Today the model computes 250 logits and we mask after softmax. Compute logits **only for legal actions** (using indexed gather). At ~5 legal actions per position avg, this is a 50× reduction in policy-head compute. Especially valuable once trades + dev cards inflate action space.

43. **Edge-features for trade actions.** Dev card / trade actions don't fit the current hex/vertex/edge graph — they're "global" moves. Either add a "global action token" node type to the heterogeneous graph, or encode trade actions via an auxiliary feature vector concatenated to the GNN body output. Defer until phase 2 (when trades are in scope).

44. **Auxiliary loss heads for stronger learning signal.** Beyond value + policy, add cheap auxiliary targets that are well-defined per state:
   - **Predict opponent's hand totals** (regression target, 3 floats per state) — uses already-recorded info; trains the body to attend to opponent state.
   - **Predict who'll get longest road eventually** (binary classification, 4 logits) — captures strategic positioning beyond immediate VP.
   - **Predict 5-step value lookahead** vs the current 25-step rollout — gives a denser, lower-variance signal to learn from.

   Each aux head is ~5 lines; the regularization effect on the body is often substantial. AlphaStar / OpenAI Five used many of these.

### L. Search algorithm improvements

45. **Replace OpenSpiel's MCTSBot with a Rust-native MCTS.** OpenSpiel is general-purpose and ports a Python implementation; for a single game we control fully, a Rust MCTS could be 5–20× faster. Especially if combined with §37 (unmake-move), §41 (action-conditioned), §42 (mask-aware policy), and §27 (batched evaluator).

46. **Dirichlet exploration noise at the root** (AlphaZero standard). Currently MCTS has c=sqrt(2) deterministically; AlphaZero injects Dirichlet noise into root priors so self-play games explore more openings. Without this, self-play data diversity is too narrow. Single most important "AlphaZero training trick" we're missing.

47. **Temperature-scheduled root selection.** Today we pick action by argmax of root visit counts. AlphaZero uses temperature τ=1 for the first ~30 moves of self-play (sample proportional to visits) then switches to argmax. **More opening diversity in training data without changing the model.**

48. **Progressive widening for chance nodes.** Today MCTS treats chance nodes as having 11 children (dice 2-12). Most plays only exercise 4-5 of those (the high-probability ones); the low-prob ones get ~1 sim each in a 50-sim budget. **Progressive widening expands chance children only as visits accumulate** — better sim budget allocation.

49. **PUCT instead of UCB1.** AlphaZero uses PUCT (Predictor + UCT) which incorporates the policy network's prior into the exploration term: `PUCT(child) = mean_value + c · prior(child) · sqrt(parent_visits) / (1 + child_visits)`. UCB1 ignores the policy. Switching to PUCT lets a strong policy steer search more aggressively. Big sample efficiency win once the policy is learning.

### M. Self-play data generation

50. **Self-play with diverse opponents in the same game.** Today's self-play has all 4 seats playing the same player kind. This creates a narrow strategic distribution. Mix opponents per seat: e.g. seat 0 = LookaheadMcts(d25,s100), seat 1 = Random, seat 2 = GreedyBaseline, seat 3 = LookaheadMcts(d10,s50). Each game produces training data from 4 different strategic regimes. Likely as valuable as 2-4× more games.

51. **Auto-curriculum: train on positions where the current model is most wrong.** After training, evaluate val_top1 per game; the games where the model is most surprised (lowest top1) are the ones that contain the strongest learning signal. Re-weight the next training run to oversample those games. AlphaZero used this implicitly via tree search; we'd do it explicitly.

52. **Online dataset augmentation: random board rotations.** The Catan board has 6-fold rotational symmetry around the center hex. Today's training set sees each game with one canonical orientation; we can augment to 6× by rotating the board (and remapping vertex/edge IDs) when materializing positions from action_history. **6× data, free.** Pairs naturally with §52's perspective rotation: 6 board-rotations × 4 player-perspectives = 24× effective data.

### N. Tooling and observability

53. **Live GNN-vs-LookaheadMcts benchmark inside training.** Every N epochs, run a 10-game tournament against LookaheadMcts at sims=50 and report winrate alongside loss. Today bench-2 is just policy-KL/value-MAE on static positions; nothing connects training metrics to actual play strength until you stop training and run e8. **Add the in-loop tournament so we know if epoch 7 is genuinely playing better than epoch 4 in real time.**

54. **Loss decomposition by game phase.** Track val_loss separately for setup-phase positions, early-main-phase, mid-main-phase, late-main-phase. Today we know "loss is X"; after this we'd know "model is great at setup but bad at endgame" or vice versa. Cheap to add (~30 lines), highly diagnostic.

55. **Action-distribution histograms.** When v1 was diagnosed as "always picks the same settlement spot," we saw it manually. Should be automatic: log the policy entropy at each game phase (setup vs main vs endgame). Low entropy → policy collapse → flag.

56. **Dataset health dashboard before training.** Every cache build should report:
   - % timed-out games (today's 41% caught us off guard)
   - Distribution of game lengths (very long games may be degenerate)
   - Distribution of margin-of-victory (lots of margin=1 games means the teacher is barely winning)
   - VP-source distribution (does the teacher ever build longest road / large army? in v1, never)
   - Resource-collection histograms per resource (does the teacher accumulate ore? if not, no city training signal)

   **Catches data-quality holes before we burn 7 hours training on bad data.**

### O. Determinism and reproducibility

57. **Single source of seed truth.** Currently we have `seed_base` (per-experiment), `seed` (per-game), `eval_seed` (per-leaf-eval), random.Random instances seeded from various things. Hard to reproduce exact behavior. Move all randomness through one `engine.rng_seed()` source so a saved seed exactly reproduces a played game.

58. **Action-history hash as state ID.** Today's `id(state)` cache key works for one Python session but isn't durable. A canonical hash of `(seed, action_history)` would let caches share state IDs across sessions and machines.

59. **Deterministic chance-action ordering.** When MCTS samples chance children, today we sample stochastically with weighted probability. For reproducibility in unit tests, support a `--deterministic-chance` mode that always picks outcomes in fixed order.

### P. Code organization

60. **Split `engine.rs` into per-phase files.** It's already 600+ lines and growing. With trades + dev cards added it'll be 1500+. Move main-phase, setup-phase, dev-card-phase, trade-phase into separate modules. Clear separation of concerns.

61. **Canonical Action enum, not action IDs.** Today we encode/decode `u32 ↔ Action` via `actions.rs::encode/decode`. With dev cards and trades there's growing risk of off-by-one bugs. Internal Rust API should use the typed `Action` enum exclusively; `u32` is only for PyO3 boundary.

62. **Extract a `RuleSet` trait.** Today's rules are baked into `apply()`. With v2 wanting selective rule subsets (e.g. "tier-1 lite" for fast unit tests, "tier-2 full" for production), parameterize rules behind a trait. Lets us test the v1 game alongside v2 for regression. Defer if it's premature abstraction.

### I. Process / engineering

34. **Branch hygiene** — v2 should land on a fresh branch off main, not gnn-v0. The two are diverging structurally and will eventually need to be merged once v2 stabilizes; cleaner if they don't share commits.
35. **Schema v3 recorder** — v2 engine state is fundamentally different. Bump SCHEMA_VERSION = 3, and have CatanReplayDataset filter to schema_version >= 3 going forward. (V1 and v2 parquets become useless training data — that's OK, the game changed.)
36. **Test coverage on rule additions** — every new rule (trades, dev cards, longest road, largest army) needs a unit test. The v1 engine had reasonable test coverage; v2 should be stronger.

---

## Suggested phasing

Don't do everything at once. Three phases:

### Phase 1: Engine v2 minimum-viable (1–2 weeks of focused work)

**Rules + perf foundation that makes everything else faster:**

- §A1 (steal verification fix), §A2 (inventory caps), §A3 (longest road), §A8 (full win condition with longest road)
- §A5 (maritime trades only — defer player-to-player)
- §A-bis 8a (instant random discard — kill the discard phase, big simplification)
- §B9 (bit-packed obs), §B10 (static board cache)
- §B-bis 11a-11g (fast legality bitmap — **the biggest perf win in v2**)
- §C12-13 (bank, all_hands APIs)
- §D16 (ABC map gen)
- §G29 (timeout investigation), §G25 (filter timeouts before caching)
- §J37 (eliminate state.clone on hot path), §J39 (Rust-native lookahead evaluator)
- §N56 (dataset health dashboard — catches the kind of "41% timeouts" hole we missed in v1)

**Why this set:** these unlock the most strategy at the lowest implementation risk, plus the perf foundation (fast legality + bit-pack + clone elimination) that makes phases 2 and 3 actually fast. Trades-to-bank gives the resource-starved-player escape valve. Longest road gives the road-spammer a payoff. Inventory caps prevent infinite road blockades. Bit-packed obs + fast legality lets us train on 10× more games per same wall-clock.

**Skip in Phase 1:** dev cards, largest army, player-to-player trades. They add a lot of complexity. Phase 1 is "is the game playable, fast?" not "is the game complete?"

### Phase 2: Strategy completeness (1–2 weeks)

**The remaining real-Catan rules:**

- §A4 (largest army), §A7 (dev cards), §A6 (player-to-player trades)
- §K43 (edge-features for trade actions in the GNN)

These are highly intercoupled (knight uses a dev card and contributes to largest army; trade is a standalone phase but shares VPCard tracking with dev cards). Once §A6+A7 land, the action space stabilizes for v2 and the model architecture (§K43) needs a small extension.

### Phase 3: Training at scale

**The AlphaZero loop becomes feasible:**

- §F26 (AlphaZero iteration loop), §F27 (batched GNN evaluator), §F28 (multi-perspective data)
- §G30 (opponent diversity in self-play), §M50-52 (diverse opponents per game, auto-curriculum, board rotation augmentation)
- §L46-49 (Dirichlet noise, temperature schedule, progressive widening, PUCT)
- §K41-42, §K44 (action-conditioned policy head, dynamic mask in loss, auxiliary loss heads)
- §N53-55 (live in-training tournaments, loss decomposition by phase, action-distribution histograms)
- §J38, §J40 (consolidated PyO3 boundary, Rust-native chance sampling), §L45 (Rust-native MCTS)

**Why all of these here:** Phase 3 is where we stop tweaking individual ideas and try to do the AlphaZero thing properly. Self-play loop, batched inference, search algorithm improvements, augmentation, observability. The earlier phases are about "make the game right and the engine fast"; Phase 3 is about "make the training process strong enough to actually beat the teacher."

### Phase 4: Polish (any time)

- §J38 (PyO3 boundary cleanup), §K41 (action-conditioned policy)
- §O57-59 (determinism polish — single seed source, durable state hash, deterministic chance)
- §P60-62 (split engine.rs, Action-enum-internally, RuleSet trait)

These are quality-of-life improvements that don't unlock new capability but make the codebase pleasanter to work with. Pull in opportunistically when working in the affected files anyway.

---

## What stays the same from v1

- The GNN architecture itself (hetero PyG model, hex/vertex/edge node types, value + policy heads). The body learns features regardless of state encoding; we'll just need to re-derive `state_to_pyg` for the bit-packed observation.
- The recorder schema philosophy — record action_history + outcome, materialize tensors on demand.
- The `CardTracker` plug-and-play interface (just changes which backend it uses).
- All training-pipeline improvements (per-epoch saves, resume, best tracking, per-game val variance, live progress, two-GNN tournament).
- The replay viewer (with v2 engine extension for longest road / largest army / dev cards).

## What we throw away

- All v1/v2 trained checkpoints (`gnn_v1_d15_keep`, `gnn_v2_d25_keep`, `gnn_v2_d25_pw4_keep`). They were trained on a strategically-different game. **Useful as reference data only.**
- All v1 cache files (`cache_d15_subset.pt`, `cache_v2_d25_w0w1.pt`). Same reason. **Delete after Phase 1 is stable** to free disk.
- The 41% timeout-prone d25 / d35 parquets. Actually no — the action_histories there might be useful as **negative examples** ("here's what a degenerate stall looks like"). Hold on to them.
