# Deep behavioral analysis — Gate 2, RL iter-1, RL iter-2

**Date:** 2026-05-31
**Method:** `analyses/tournament_deep_analysis.py` replays every game's
`action_history` through the engine (vp=10, bonuses) and tallies per-role
build/bonus/trade/closeout/port/resource metrics, stratified by win/loss.
Same methodology as the 2026-05-27 full-Catan deep analysis, generalized to
both parquet layouts. **0 replay failures across all 320 games.**

Three tournaments analyzed:
1. **Gate 2** (`gate2_e10e_async`, 80 games) — the async MCTS validation.
2. **RL iter-1 arena** (`rl_iter1_eval`, 120 games) — first RL net vs Cell6.
3. **RL iter-2 arena** (`rl_iter2_arena`, 120 games) — second RL net vs Cell6.

Metrics and observation offsets cited in the script header (observation.rs).

---

## Part A — How we measure in-game progress (the metric set)

Each game is replayed action-by-action. Per role we extract, post-setup:

| Metric | What it measures | Why it matters in full Catan |
|---|---|---|
| **roads/g, settle/g, cities/g** | main-phase builds | direct VP (settle=1, city=2) + LR enabler |
| **dev/g** | BuyDevCard actions | the engine for knights (→LA) + VP cards |
| **knights/g** | PlayKnight actions | ≥3 = Largest Army (+2 VP) |
| **mean_LR** | longest-road length | ≥5 contiguous = Longest Road (+2 VP) |
| **%LR_held / %LA_held** | bonus possession at game end | each bonus = 2 VP; 4 VP is ~40% of a 10-VP win |
| **mean_VP** | victory points at game end | the closeout proxy — stalling shows here |
| **bank/g, propose/g** | trade activity | resource-conversion intensity |
| **median game length, 1st-city turn** | closeout speed | fast clean win vs slog |
| **% settlements on ports** | 2:1/3:1 trade access | cheaper resource conversion |
| **resource specialization** | hex types adjacent to settlements | ore/wheat = city + dev economy |

A 10-VP win is typically: cities (each 2) + settlements (each 1) + the two
bonuses (LR+LA = 4) + occasional VP card. So **dev→knights→LA** and
**roads→LR** are the two load-bearing 2-VP engines; mean_VP at game end is the
diagnostic for whether a policy can actually *close*.

---

## Part B — Gate 2: why value-fixed GNN+MCTS wins (53.8%)

Winrate: **GnnMcts_Cell6 53.8%**, LookV3 33.8%, PureGnn_Cell6 6.2%, PureGnn_Cell1 6.2%.

The decisive comparison is **GnnMcts_Cell6 vs PureGnn_Cell6 — the SAME network,
the only difference is whether MCTS sits on top.** Search transforms it:

| metric (overall) | PureGnn_Cell6 | GnnMcts_Cell6 | LookV3 |
|---|---:|---:|---:|
| roads/g | 4.81 | **5.88** | 8.25 |
| settle/g | 1.14 | **1.40** | 1.70 |
| cities/g | 0.79 | **1.39** | 1.27 |
| dev/g | 5.81 | **6.62** | 2.11 |
| knights/g | 2.56 | **3.20** | 0.47 |
| **mean_VP (end)** | 5.53 | **7.69** | 7.21 |
| %LR_held | 17.5% | **30.0%** | 47.5% |
| %LA_held | 23.8% | **40.0%** | 0.0% |

**Search makes the net build more of everything and — critically — close
games.** PureGnn ends at mean 5.53 VP (it *stalls* around the old v3 5-VP
threshold); GnnMcts ends at 7.69 VP. MCTS looks ahead and steers toward the
VP-completing moves the raw policy leaves on the table.

**The bonus economy is where GnnMcts beats LookV3.** LookV3 is a pure
Longest-Road machine (47.5% LR, mean 0.48 knights — it essentially never plays
knights, 0% LA). GnnMcts is the only **dual-bonus** player: 30% LR **and** 40%
LA. It inherits Cell6's dev-card economy (3.20 knights/g → crosses the 3-knight
LA threshold) and adds MCTS-guided road/city building. In its 43 wins: **62.8%
held LA, 48.8% held LR** — it wins via both 2-VP bonuses, where LookV3 wins via
LR alone (81.5% LR, 0% LA) and tops out.

**Closeout:** GnnMcts reaches its first city at EndTurn 50.6 (vs PureGnn's 75.6)
and wins at median 430 moves — faster *and* cleaner. **Resource
specialization:** GnnMcts settles for wood+wheat (20.8% / 22.0%) and avoids
desert (3.8% vs PureGnn's 8.1%) — search picks better vertices.

**Conclusion:** with correct value backup, MCTS converts Cell6's policy into a
materially stronger player — more builds, both bonuses, faster closeout,
higher final VP. This is the mechanism behind 6.2% → 53.8%.

---

## Part C — RL iter-1 vs iter-2: a behavioral trajectory

Winrate vs parent Cell6: iter-1 **15.0%** (gap −40.8pp), iter-2 **22.5%**
(gap −24.2pp). The build data shows *why* both lose and *how* iter-2 improved.

### iter-1: a passive, under-building net

| overall | RL_iter1 | Cell6 |
|---|---:|---:|
| roads/g | **1.49** | 6.03 |
| settle/g | **0.54** | 1.59 |
| cities/g | 0.54 | 1.28 |
| dev/g | **3.49** | 7.27 |
| knights/g | **1.05** | 2.80 |
| mean_VP (end) | **4.21** | 8.17 |
| %LR_held | 6.7% | 50.8% |
| %LA_held | 13.3% | 32.5% |

RL_iter1 **barely acts** — a third of Cell6's roads, a fifth of its bank trades
(1.93 vs 20.92), half the dev cards. It ends at mean **4.21 VP** — it stalls
below even the v3 5-VP line. Training on only 66 self-play games collapsed the
policy toward inactivity; it can't build the bonus engines (6.7% LR, 13.3% LA).

### iter-2: more active, closer — but still stalls

| overall | RL_iter2 | Cell6 |
|---|---:|---:|
| roads/g | **2.22** (↑ from 1.49) | 4.87 |
| settle/g | 0.64 | 1.23 |
| cities/g | 0.45 | 1.20 |
| dev/g | **4.03** (↑) | 6.88 |
| knights/g | **1.27** (↑) | 2.88 |
| mean_VP (end) | **4.53** (↑ from 4.21) | 7.32 |
| %LR_held | **15.0%** (↑ from 6.7%) | 32.5% |
| %LA_held | **16.7%** (↑ from 13.3%) | 32.5% |

**Every activity metric rose** with 3× the data: roads 1.49→2.22, dev 3.49→4.03,
knights 1.05→1.27, mean_VP 4.21→4.53, LR-held 6.7→15.0%, LA-held 13.3→16.7%.
Note Cell6's *own* numbers dropped too (mean_VP 8.17→7.32, roads 6.03→4.87) —
iter-2 is a tougher opponent that contests more games, pulling everyone's
per-game build counts down.

**The persistent failure is closeout.** Both RL nets end at mean ~4.2–4.5 VP —
they **freeze around 4-5 VP** and rarely reach 10. (Compare GnnMcts's 7.69 and
Cell6's 7.3–8.2.) In the games RL_iter2 *does* win (27 of them), it looks
normal — 6.37 roads, 2.22 settles, 4.22 knights, 51.9% LR, 74.1% LA, mean 10.19
VP. So it *can* close; it just does so far less often than the parent. The
trajectory (4.21→4.53 mean VP, gap −40.8→−24.2pp with 3× data) is the
canonical AlphaZero "student approaching teacher" curve — it needs more
data/iterations to push mean-VP past the closeout cliff.

iter-2 closeout oddity: `mean_1st_city_endturn` = 325 (vs iter-1's 82) — across
ALL games it reaches first city very late, consistent with a net that spends a
long time at low VP before (sometimes) committing to cities.

---

## Part D — The value-perspective bug, explained precisely

This is the bug whose fix produced Gate 2's 6%→53.8% result.

### The setup: an ego-relative value head

The GNN value head outputs a length-4 vector. The training target
(`catan_gnn/dataset.py:118-126`) is **ego-relative**:

```python
# value[offset] = +1 if (current_player + offset) % 4 == winner, else -1
```

So `value[0]` is the value for **whoever is to move at this state**, `value[1]`
for the next seat, etc. This pairs with the engine's observation
(`catan_engine/src/lib.rs:149`), which is **ego-centric** — it always renders
the board from `viewer = current_player`. Net input is mover-relative, so net
output is mover-relative. Self-consistent.

### The bug: MCTS indexed it as absolute-seat

MCTS backs up a leaf's value to every node on the path. Each node belongs to a
specific player (`node.to_play`, an absolute seat 0–3). The old code (and
OpenSpiel's MCTSBot, and the old `gnn_evaluator.py`) did:

```python
node.value_sum += value_vec[node.to_play]      # WRONG
```

It indexed the **ego-relative** vector with an **absolute** seat number. That's
only correct for the one node whose `to_play` happens to equal the leaf's mover.
**For every other node on the path, it read the wrong player's value** — often a
*different* player's, since `value[k]` means "the player k steps after the leaf
mover," not "seat k."

Concretely: a leaf evaluated when seat 2 is to move returns
`value = [v_for_seat2, v_for_seat3, v_for_seat0, v_for_seat1]`. An ancestor node
owned by seat 0 wants its own value, `v_for_seat0` — which lives at index **2**,
not index 0. The buggy code read `value[0]` (= seat 2's value) and added it to
seat 0's Q. **Poisoned Q-values → MCTS steered by garbage → it played worse than
no search at all.** That is exactly the 2026-05-29 finding: buggy GnnMcts won
1.2%, *below* PureGnn.

### The fix: rotate ego→absolute at the leaf

`async_mcts.py` `_expand_and_evaluate` now rotates the value vector into
absolute-seat order before returning it, so backup can index by `node.to_play`
uniformly:

```python
leaf_mover = state.current_player()
value_abs = np.empty(4)
for offset in range(4):
    value_abs[(leaf_mover + offset) % 4] = value[offset]
return value_abs
```

Now `value_abs[seat]` is genuinely seat's value for every seat. Terminal leaves
need no rotation — `state.returns()` is already absolute-seat-indexed — so only
the GNN-value branch rotates. Proven by `test_value_rotated_to_absolute_seat`
(asserts `value_abs[seat] == ego_value[(seat - leaf_mover) % 4]` for all seats).

**Result of the fix:** GnnMcts 1.2% → 53.8%; search went from *hurting* to being
the strongest player measured. The "GNN+MCTS is worse than the policy" conclusion
from the prior session was entirely this indexing artifact.

**Still-open:** the OLD `catan_mcts/gnn_evaluator.py` retains the bug (it returns
the raw ego-relative vector and OpenSpiel indexes it absolutely). Any future use
of the sync evaluator for GnnMcts must apply the same rotation, or use the async
stack (now the reference MCTS).

---

## Summary table — the three tournaments

| | Gate 2 | iter-1 arena | iter-2 arena |
|---|---|---|---|
| Winner | **GnnMcts_Cell6 53.8%** | Cell6 55.8% | Cell6 46.7% |
| Key finding | search (value-fixed) makes Cell6 dominant; beats LookV3 | RL net passive, stalls at 4.2 VP | RL net more active, stalls at 4.5 VP, gap halved |
| Mechanism | dual-bonus (30% LR + 40% LA), fast closeout, mean 7.69 VP | under-builds everything; 6.7% LR / 13.3% LA | every metric up vs iter-1; closeout still the wall |

## Cited
- Script: `analyses/tournament_deep_analysis.py`
- Prior methodology: `2026-05-27-fullcatan-deep-behavioral-analysis.md`
- Bug + fix: `project_gnn_value_perspective_bug_2026_05_30` (memory),
  `2026-05-31-gate2-clean-rerun.md`
- RL journals: `2026-05-31-rl-loop-iter1.md`, `2026-05-31-rl-loop-iter2.md`
