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

### B. Engine — state representation

9. **Bit-packed observation tensor** — design and ship the bit-layout above; verify size reduction empirically (target 50–80 bytes/state vs ~3 KB today).
10. **Static board cache** — board topology (vertex/edge/hex adjacency, resource types, number tokens, port types) doesn't change per game. Store ONCE per seed, not per state. Big cache size win.
11. **Per-state delta encoding (optional)** — many actions change one or two fields of the state; could store states as deltas from a base. Defer unless §9+§10 don't get us under target.

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

### I. Process / engineering

34. **Branch hygiene** — v2 should land on a fresh branch off main, not gnn-v0. The two are diverging structurally and will eventually need to be merged once v2 stabilizes; cleaner if they don't share commits.
35. **Schema v3 recorder** — v2 engine state is fundamentally different. Bump SCHEMA_VERSION = 3, and have CatanReplayDataset filter to schema_version >= 3 going forward. (V1 and v2 parquets become useless training data — that's OK, the game changed.)
36. **Test coverage on rule additions** — every new rule (trades, dev cards, longest road, largest army) needs a unit test. The v1 engine had reasonable test coverage; v2 should be stronger.

---

## Suggested phasing

Don't do everything at once. Three phases:

### Phase 1: Engine v2 minimum-viable (1–2 weeks of focused work)

- §A1 (steal verification fix), §A2 (inventory caps), §A3 (longest road), §A8 (full win condition with longest road), §A5 (maritime trades only — defer player-to-player), §B9 (bit-packed obs), §B10 (static board cache), §C12-13 (bank, all_hands), §D16 (ABC map gen), §G29 (timeout investigation), §G25 (filter timeouts before caching).

**Why this set:** these unlock the most strategy at the lowest implementation risk. Trades-to-bank gives the resource-starved-player escape valve. Longest road gives the road-spammer a payoff. Inventory caps prevent infinite road blockades. Bit-packed obs lets us train on 4× more games per same RAM. ABC map gen makes the boards fair.

**Skip in Phase 1:** dev cards, largest army, player-to-player trades. They add a lot of complexity. Phase 1 is "is the game playable?" not "is the game complete?"

### Phase 2: Strategy completeness (1–2 weeks)

- §A4 (largest army), §A6 (player trades), §A7 (dev cards). All three together because they're highly intercoupled (knight uses a dev card and contributes to largest army; trade is a standalone phase but shares VPCard tracking with dev cards).

### Phase 3: Training at scale

- §F26 (AlphaZero loop), §F27 (batched evaluator), §F28 (multi-perspective data), §G30 (opponent diversity). With a complete game and 40× cache compression, the question becomes "can iterative self-play actually improve the model" — the original AlphaZero question.

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
