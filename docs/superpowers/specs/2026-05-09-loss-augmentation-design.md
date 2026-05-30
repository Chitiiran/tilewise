# Loss Augmentation Design — addressing GNN behavior gaps observed in 1200-game tournament

**Date:** 2026-05-09
**Status:** Design exploration. NOT a commitment to implement yet.
**Tournament evidence base:** `runs/v3/tournaments/grid_pass100k_tournament/h32_l2_dual_best_1200/2026-05-09T03-43-e10b_dual_gnn/`

## Background — what the cited evidence shows

From `scratch_midgame_actions.py` reading 40 parquets of the e10b_dual_gnn
tournament (1200 games), action-rate per role per 100 turns:

| Action | PureGnnA | PureGnnB | LookaheadV3 | Random |
|---|---|---|---|---|
| BuildCity (upgrade) | 6.03 | **3.11** | 6.32 | **0.46** |
| BuildRoad (post-setup) | **11.62** | **12.44** | 23.05 | 22.93 |
| BuildSettlement | 3.57 | 3.82 | 5.29 | 0.52 |
| BuyDevCard | 4.42 | **1.15** | **9.33** | 5.89 |
| ProposeTrade (% of all actions) | **36.4%** | **37.2%** | 31.9% | 32.5% |

From `scratch_opening_analysis.py` (cited mean opening pip count by role):

| Role | Mean opening pips | Median |
|---|---|---|
| PureGnnA | 18.19 | 19 |
| PureGnnB | 18.16 | 18 |
| LookaheadV3 | 16.32 | 17 |
| Random | 12.03 | 12 |

Conditional win rate (when this role had STRICT highest opening pips):

| Role | n_top | wins | conv |
|---|---|---|---|
| PureGnnA | 356 | 39 | **11.0%** |
| PureGnnB | 371 | 10 | 2.7% |
| LookaheadV3 | 263 | 228 | **86.7%** |
| Random | 38 | 1 | 2.6% |

## Three cited gaps the GNNs exhibit

1. **Under-builds roads** (PureGnnA: 11.62 / PureGnnB: 12.44 vs Lookahead 23.05 / Random 22.93 per 100 turns).
2. **PureGnnB rarely upgrades cities** (3.11 vs Lookahead 6.32 per 100 turns). PureGnnA closed this gap (6.03), suggesting the 100k corpus already taught it. The pass-3 corpus did not.
3. **Over-uses ProposeTrade** (~36% of all actions) compared to Lookahead (~32%). Hypothesis from user: trades aren't being accepted, so this is wasted action budget.

The result of these gaps is best summarized by the conditional win-rate
table: **PureGnnA picks better openings on average than Lookahead but
converts only 11% of "best opening" games into wins, vs Lookahead's 87%**.
The opening is fine. The mid-game conversion is the failure.

---

## Three loss-augmentation candidates

### Candidate 1 — Pip-weighted settlement KL (auxiliary policy prior)

**Goal:** During training, push the policy probability mass toward higher-pip
settlement vertices when settlement actions are legal.

**Mechanism:**
```
For each training sample:
  legal_settle_mask = legal_mask[0:54]  # cited playback.py:260
  if any(legal_settle_mask):
      pip[v] = sum(PIP_BY_DICE[hex_dice[h]] for h in HEX_TO_VERTICES_REVERSE[v])
                                                # cited adjacency.HEX_TO_VERTICES
      target = softmax(pip[legal_settle_mask])
      pred   = softmax(logits[legal_settle_mask])
      aux_loss = KL(pred || target)
      loss += λ_settle * aux_loss
```

**λ_settle suggestion:** 0.05 (small — MCTS visits remain dominant signal).

**Cited evidence justifying this:**
- The GNN already picks high-pip openings on average (18.19 vs 16.32 for Lookahead, cited above), but only converts 11% to wins. So **this aux loss may not move the needle on conversion.** It would mainly tighten variance.

**Risks:**
1. Pip count is a heuristic. It ignores **resource diversity, port access, blocking, robber threat**. A pip-only prior pulls the model away from MCTS in cases where MCTS correctly weighed those other factors.
2. Mid-game settlements (post-setup) are rare and tactical. Applying pip pressure to them could cause the model to over-extend to high-pip vertices that have other liabilities.
3. **The conditional win-rate data suggests opening placement isn't the bottleneck.** This loss may not help.

**Mitigation:** apply only during setup phase (first 16 actions) by gating
on `phase == setup` from `scalars`. Cited `playback.py:307: SCALAR_PHASE = 13`.

---

### Candidate 2 — City-upgrade policy boost

**Goal:** When BuildCity is legal somewhere, blend the MCTS visit-count target
with an indicator that favors the city action.

**Mechanism (target-side, not separate loss):**
```
new_target = (1 - α) * mcts_visits_normalized + α * city_indicator
where city_indicator[a] = 1 / count(legal city actions) if a is BuildCity, else 0
                          0 if no city action is legal
```

**α suggestion:** 0.10 (10% nudge toward cities).

**Cited evidence justifying this:**
- PureGnnB upgrades cities at 3.11/100 turns vs Lookahead's 6.32/100 — exactly half the rate.
- Cities are 2 VP and 2× resource yield (cited Catan rules + observation.rs).
- The 100k corpus already lifted PureGnnA to 6.03/100 (essentially matching Lookahead), confirming this is *learnable from data alone*.

**Risks:**
1. **Hand-coding "this action is good" is brittle.** A city upgrade in late game when you're already at 4 VP and need only roads (longest road) is a wasted move. The blanket boost punishes this nuance.
2. Could create distribution shift: the model learns to favor city upgrades whenever legal, even when MCTS would have correctly skipped due to resource conservation needs.
3. The 100k corpus already mostly fixed this for PureGnnA. So the gain may already be available via more training, not a code change.

**Mitigation:** condition the boost on having ≥3 ore + ≥2 wheat (the city
cost — cited Catan rules). If we don't have the resources, the boost is
zero. This requires reading `scalars[SCALAR_HAND..SCALAR_HAND+5]` per sample.

---

### Candidate 3 — Edge-pip road KL (the hardest one — see "consequences" section below)

**Goal:** Push road-action probability mass toward edges that unlock
high-pip settlement vertices in 1-2 hops.

**Mechanism:**
```
For each edge e:
  candidate_vertices = [v adjacent to e via VERTEX_EDGE_INDEX, cited adjacency.py]
  reachable_in_1 = filter(candidate_vertices, settlement_legal=True)
  reachable_in_2 = vertices adjacent to (vertex adjacent to e) via paths of length 2
                   that are settlement-legal AND not blocked by Catan distance rule
  edge_score[e] = max(pip[v] for v in reachable_in_1 ∪ reachable_in_2)
target_road = softmax(edge_score[legal_road_actions])
pred_road   = softmax(logits[legal_road_actions])
aux_loss = KL(pred_road || target_road)
loss += λ_road * aux_loss
```

**λ_road suggestion:** 0.05.

**This is where the user's question kicks in →** see next section.

---

## Deep-dive: consequences of the road policy (THE USER'S CONCERN)

### The user's exact framing
> "we have not considered the consequences deeply if we add policy for [the
> road], we need to consider what happens after one road is built. what
> gets priority then. hopefully building settlement is picked as correct."

### Why this concern is real

The training data is **per-state, not per-trajectory**. Each training sample
is a (state, action) pair where action is the MCTS-chosen move. If we add
λ_road * KL(road_pred || edge_pip_prior) to the loss, the model gets a
gradient signal for road-favoring **whenever a road is the legal target**.

Consider this trajectory after we add the road prior:
```
turn t:   state has 2 wood + 2 brick. policy puts mass on BuildRoad(e=42)
            (e=42 unlocks vertex v=27 which has pip=10).
            Action chosen: BuildRoad(e=42). Engine applies it.
turn t+1: state now has the road but lost wood+brick. To build the
            settlement at v=27, the model needs to wait for resources.
            What does the model want NOW?
              - Right answer: collect resources (EndTurn), then BuildSettlement(v=27)
              - Wrong answer: build ANOTHER road (because the prior keeps pushing
                roads — every road has SOME pip score)
```

**Specifically, the auxiliary loss applied at turn t+1 will compute
edge_score for whatever roads are legal NOW. Those roads will have
different pip-targets than v=27. So the model's prior will push it to
build a *different* road, not to wait for v=27.**

This is the bug pattern the user named:
> "what gets priority then. hopefully building settlement is picked as
> correct."

### Three ways to address this consequence

**Option A — Trust MCTS to dominate.** With λ_road=0.05 and MCTS visit-count
weight=0.95, the MCTS signal still dominates. If the MCTS at turn t+1
correctly chose BuildSettlement(v=27), the policy target for that sample
is that settlement (visit=1 at v=27, road visits=0). The road KL only
applies to the road-subset of the action space, but its gradient is small
relative to the settlement target gradient. **Likely fine in practice.**

**Option B — Mask out the road KL when settlement is legal.** Add a gate:
```
if any(legal_settlement_actions):
    skip the road KL for this sample
```
This says: when the model could build a settlement, don't reward it for
building roads instead. Cleaner but requires the dataset to track this.

**Option C — Multi-step conditioning (the right answer, but expensive).**
Rather than a per-state heuristic, compute the road's value from the
**eventual settlement that gets built on this trajectory**. This requires
walking forward in the recorded trajectory at dataset-build time:
```
For each road action at state t in a game:
  Look forward in the trajectory (recorded actions) for the same player.
  If they later build a settlement at vertex v, AND v is reachable via
  this road, AND no intervening road would have made v reachable already:
      then this road's "true value" = pip[v] (the settlement that this road
      enabled).
  Else: the road was wasted, true value = 0.
target_per_road_action = computed from above, not from per-state pip heuristic
```

**This is the trajectory-aware version.** It's expensive (re-walks the
game per sample) but gives an unbiased signal: the road's value comes from
the settlement it enabled in the actual rollout, not from a snapshot heuristic.

### Recommended path

Start with **Option A** (small λ, trust MCTS dominance). Measure whether
the road-rate metric (currently 11.62/100 turns) moves toward Lookahead's
23.05. If it doesn't move, escalate to Option B. Don't go to Option C
without a clear sign that the heuristic is causing a chain-of-roads
pathology in playbacks.

---

## Trade investigation — user's concern

### The user's exact framing
> "right now it seems like no trade is accomplished because other players
> dont get to play during one player's trade. so maybe for training we
> limit player to player trade proposal and only allow maritime trade.
> document all this and we can then think from external pov to crack more
> lateral thinking"

### What the cited data shows

From the 1200-game analysis (`scratch_midgame_actions.py`):
- PureGnnA: **ProposeTrade = 36.4% of all post-setup actions** (19,184 / 52,759 total)
- PureGnnB: 37.2%
- LookaheadV3: 31.9%
- Random: 32.5%

**Both GNNs propose trades MORE than Lookahead or even Random.** This is
huge. ProposeTrade has 20 distinct action IDs (260..279, cited
playback.py:292-299), so the action space heavily incentivizes trading
proposals just by sheer count.

### What we don't yet have evidence for
- Whether any of these proposed trades **succeed** (are accepted by another player)
- Whether the GNN's trade proposals are *redundant* (proposing the same trade pair multiple turns in a row)
- Whether trade success rate differs between GNN and Lookahead

### The engine's trade-acceptance rule (needs verification)
From the action layout cited in playback.py:299:
```
ProposeTrade(give↔get) — 5 give × 4 get = 20 actions
```
There is **no `AcceptTrade` action ID**. This means the engine must have
an automatic acceptance rule. Two plausible rules:
1. **Auto-accept if another player benefits more** (heuristic-based)
2. **Engine resolves immediately after the propose action** (no other player's
   turn to accept/reject)

We need to read `catan_engine/src/` to confirm. This is critical: if the
engine auto-rejects (or auto-accepts) deterministically, then ProposeTrade
is functionally a one-sided action and the model's high rate may be
"rational from MCTS's view" — every trade succeeds with the engine and
the GNN learns to spam them.

### The user's proposed mitigation: ban player-to-player trade in training
> "for training we limit player to player trade proposal and only allow
> maritime trade"

This means: at dataset-build time, mask out ProposeTrade (260..279) from
the legal_action_mask, leaving only TradeBank (206..225, cited
playback.py:271-278) as the trade option.

**Why this could work:**
- TradeBank is unambiguous: bank gives 4-for-1 (or 3-for-1 with port, 2-for-1
  with specific resource port). Cited Catan rules.
- TradeBank's value is computable: ROI = (wanted resource - 4×given resource).
- Forcing the model to use only TradeBank during training removes the
  ProposeTrade ambiguity entirely.

**Why this could backfire:**
- If the engine's ProposeTrade auto-accepts, the GNN was getting **free
  resources** via trading. Banning ProposeTrade would force it to use
  TradeBank (4-for-1, very inefficient) and tank its win rate.
- The opponent (Lookahead) still uses ProposeTrade in real evaluation. If
  we train the GNN without ProposeTrade exposure, it has no policy mass
  for that action at evaluation time → it loses tempo whenever a
  ProposeTrade would have been the right move.

**The right experiment:** measure ProposeTrade success rate from existing
data first. If success rate is near zero, the proposals are wasted action
budget and the user is right; banning them helps. If success rate is
non-trivial, banning them tanks performance.

### Open questions on trades
1. What is the engine's ProposeTrade resolution rule? (read `catan_engine/src/state.rs`)
2. What fraction of GNN-proposed trades resulted in a state change vs no-op? (parse moves.parquet but it's empty in this tournament; need to use action_history before/after deltas)
3. Does Lookahead have a higher trade-success rate? If yes, the GNN's failure isn't "trades don't work" but "GNN proposes worse trades."

---

## Lateral-thinking prompts (per user's request)

The user explicitly asked for "external POV to crack more lateral
thinking." Some angles worth considering:

### A. Maybe the loss is fine — the **dataset** is wrong  ❌ FALSIFIED 2026-05-09

**Original hypothesis (now refuted):** that setup placements aren't in the
training data because the recorded-player MCTS filter would skip them.

**Evidence that refutes this** (`scratch_check_setup_samples.py` reading
`runs/v3/data_gen/2026-05-05T05-50-e9_v3_data_gen_100k_w12/.../worker0/moves.v3-final.parquet`):

```
=== move_index=0 action distribution (n=7917) ===
  Settle: 7917  (100.0%)

=== mi=0 sample legal_mask cardinality ===
  median legal: 54.0   ← all 54 vertices legal = empty board = setup move 1

=== move_index 0..3 categories ===
  mi=0: 100% Settle (n=7917)   ← recorded_player's 1st setup settlement
  mi=1: 100% Road   (n=7917)   ← recorded_player's 1st setup road
  mi=2: 100% Settle (n=7917)   ← recorded_player's 2nd setup settlement
  mi=3: 100% Road   (n=7917)   ← recorded_player's 2nd setup road
  mi=4+: mostly ProposeTrade/EndTurn/etc. (mid-game)
```

**Cited mechanism** (`e9_v3_data_gen.py:66`): `recorded_player = seed % 4`.
The recorded seat plays through setup with MCTS like everyone else, and
its 4 setup decisions are recorded with full visit counts. **Setup is in
the dataset, structured as the first 4 move_index entries per game.**

**Cited counts:** 31,668 setup samples per worker (12.4% of 255,370 rows).
Across the 100k cache (3.22M positions), setup samples are roughly 12-15%
of all training data — well-represented.

### A' (REVISED) — The model has the road/settle data and is failing to learn it

The interesting question shifts from "where's the data?" to "why doesn't
the model learn from the data it has?"

**Cited mismatch** between training data composition and tournament behavior:

| | Training data (worker0 sample) | PureGnnA tournament behavior |
|---|---|---|
| Settle rate | 7.61% of all rows | 1.35% of post-setup actions |
| Road rate | 16.10% of all rows | 4.39% of post-setup actions |

(Training-data figures cited from `scratch_check_setup_samples.py` whole-worker
stats; tournament figures from `scratch_midgame_actions.py` per-role rate
table. These aren't directly comparable — training counts ALL recorded
actions including setup; tournament counts only post-setup. But the
scale of the gap, ~3-4× lower in tournament, persists even when restricting
training to mi≥4: post-setup road actions in worker0 sample = 41,102
roads − 7,917×2 setup roads = 25,268 roads / (255,370 − 7,917×4 setup) =
**11.2%** post-setup, vs PureGnnA's **4.4%** in tournament. ~2.5× gap.)

**Hypotheses (now hypotheses, not facts):**

1. **Loss imbalance across action ID density.** Cross-entropy on a 280-dim
   policy spreads gradient across all action types equally. With 72 edge
   IDs (BuildRoad) and 20 ProposeTrade IDs, but ProposeTrade typically
   has more legal options at any state than BuildRoad does, the
   normalized softmax gives ProposeTrade actions more total mass per
   sample. The model learns this distribution.

2. **MCTS visit-count distribution itself is biased.** Lookahead at depth=10
   with 200 sims may genuinely visit ProposeTrade subtrees more than
   BuildRoad subtrees (because trade outcomes are deterministic given
   resources, while road outcomes branch into many future moves). The
   GNN learns Lookahead's biased exploration, not the optimal action.

3. **Road action IDs are spatially diffuse, settle/city are concentrated.**
   Roads occupy 72 distinct edge IDs; the model has to learn which of 72
   edges to favor at each state. Settlements occupy 54 vertex IDs but
   most are illegal at any given state (distance rule), so the effective
   choice space per sample is small (~5-15 legal vertices). Roads have
   more "competing" outputs per sample, diluting the gradient.

**This argues that loss augmentation IS the right intervention** — but
not via "add settlement data." It should reweight or restructure the
existing policy supervision so that the high-information-density actions
(roads, cities) get more gradient than the low-density actions
(ProposeTrade variants).

**Concrete proposal added to the candidate list:**

### Candidate 7 (NEW) — Action-class-balanced policy loss

Reweight the masked CE so each action *class* contributes equally to the
loss, regardless of how many IDs it occupies in the action space:

```
class_weight[a] = 1.0 / count(legal IDs in action_class_of(a))
target_normalized = target * class_weight (then renormalize per sample)
loss = masked_CE(logits, target_normalized, mask)
```

This says: a ProposeTrade gradient is divided by ~20 (number of trade
variants), a BuildRoad gradient is divided by ~5-10 (number of legal
roads), a BuildCity gradient is divided by ~1-3 (number of legal cities).
The model gets equally strong signal per *class decision* rather than per
*action ID*.

**Risk:** this is a target-side rescaling. If the MCTS truly knows that
ProposeTrade is the right move at this state, downweighting its gradient
just slows learning, doesn't change the final answer. But MCTS visit
counts are noisy enough at 200 sims that systematic rebalancing across
the action space could plausibly help.

**Cost:** ~30 LOC in `dataset.py::__getitem__` or in `train.py`'s loss
computation. No model surgery.

### B. Maybe the model is right and tournament configuration is wrong
The 1200-game tournament uses `lookahead_depth=10, base_sims_v3=200`
(cited `worker0/config.json`). Lookahead's 200 sims with depth-10 is
a HUGE search budget compared to PureGnn (zero search). What if PureGnn's
86% loss rate is just "no search vs heavy search" and **adding ANY MCTS
to PureGnn (i.e., GnnMcts at 100 sims) closes most of the gap**?

The pass-3 tournament showed GnnMcts winning 518/1080 games (cited
`scratch_pick_matches.py` original output, before the rotation bug was
fixed) — but that was the wrong convention. Need to recompute under the
correct convention. **If GnnMcts wins meaningfully more than PureGnn does,
then the loss isn't the problem — search is.**

### C. Maybe early-stopping the training is the highest-impact fix
The current training shows val_loss climbing from epoch 2 onward (cited
`grid_pass100k_diagonal_2026-05-09T07-49.log` epochs 1-4). Best is at
epoch 3 for h32_l2 (val_top1=0.184) and ep2-3 for h128_l4. This means we
already had the best model after **3 epochs**. The remaining 17 epochs of
each cell are net-negative.

If overfitting is bottlenecking the model, **regularization** (dropout,
weight decay, label smoothing on the policy target) may be a higher-value
change than auxiliary losses. The 100k cache is already fixed — we can't
add data. But we can stop the model from memorizing it.

### D. Resource-conditioned action masks
The legal_action_mask currently encodes engine-level legality (you have
the resources). But it doesn't encode "this would be a strict economic
loss." Example: building a road with 1-wood-1-brick when the only
unlockable vertex is desert-adjacent is a strict waste. We could add an
**economic-legality mask** that's stricter than the engine's legality:
```
if BuildRoad(e) AND no high-pip vertex unlocked AND no longest-road threat:
    suppress in policy_target
```

This is similar to Candidate 3 but as a **target-side modification** (at
dataset construction) rather than a loss. Cleaner because no λ to tune.

### E. The VP-economy of v3 is asymmetric and the loss treats actions symmetrically (NEW 2026-05-09)

**Cited fact:** v3 runs with `bonuses=false` (cited
`worker0/config.json::bonuses=false` for the 1200-game tournament; cited
`engine.rs:43` "bonuses_enabled=false disables the +2 VP awards"). This
disables longest-road and largest-army bonuses. In v3, **roads have zero
direct VP value** — they only matter as **enablers** of future settlements.

**Cited VP-grant points** (from `rules.rs`):
- `rules.rs:97`  Setup1Place settlement → vp += 1
- `rules.rs:121` Setup2Place settlement → vp += 1
- `rules.rs:181` Main-phase BuildSettlement → vp += 1
- `rules.rs:210` BuildCity → vp += 1 (comment: "settlement was 1VP, city is 2VP, net +1")
- `rules.rs:266` PlayVpCard → vp += 1
- `rules.rs:765,790,793,...` longest-road / largest-army holder transitions
  — **these branches are dead in v3** because they're gated on `bonuses_enabled`.

**VP-yielding actions in v3:** {BuildSettlement, BuildCity, PlayVpCard}. That's it.

**Path-to-5-VP minimum:** start with 2 VP from setup; need 3 more. Most
efficient is 3 city upgrades (cost 9 ore + 6 wheat). No roads required.

**The cited behavior gap reframed:**
- LookaheadV3: 6.32 cities/100 turns (cited from 1200-game analysis) →
  high VP-generation rate → 86.2% win rate.
- PureGnnB: 3.11 cities/100 turns → exactly half the VP-generation rate
  → 3.2% win rate.
- The model's "city under-builds" isn't a settlement-placement issue or a
  pip-count issue. It's that **the loss never explicitly told the model
  that BuildCity is worth 1 VP and ProposeTrade is worth 0 VP.**

The terminal value signal (±1 at game end) is too diffuse. MCTS visit
counts are a teacher's compression of value into action probability, but
they ignore the structure: "this action class produces VP, that one
doesn't."

**This argues for two new candidates:**

### Candidate 8 (NEW) — Action-class VP prior (target-side, ~30 LOC)

Reweight the policy target with a per-action-class VP-yield score:

```python
CLASS_VP_VALUE = {
    "BuildCity":       1.0,   # +1 VP direct (cited rules.rs:210)
    "BuildSettlement": 1.0,   # +1 VP direct (cited rules.rs:181)
    "PlayVpCard":      1.0,   # +1 VP direct (cited rules.rs:266)
    "BuyDevCard":      1.0/14, # ~1/14 chance of VP card per Catan rules
                                # (cited KNIGHT_DECK_TOTAL=14 in observation.rs:64
                                # plus 5 progress + 5 VP cards typical)
    "BuildRoad":       0.0,   # zero direct (handled separately by Cand. 9)
    "PlayKnight":      0.0,   # zero direct (largest-army disabled in v3)
    "TradeBank":       0.0,
    "ProposeTrade":    0.0,
    "MoveRobber":      0.0,
    "EndTurn":         0.0,
    "RollDice":        0.0,
    "Discard":         0.0,
    "PlayMonopoly":    0.0,
    "PlayRoadBuilding":0.0,
    "PlayYearOfPlenty":0.0,
}
```

Build a per-sample target prior over the action space:

```python
vp_score = np.zeros(ACTION_SPACE_SIZE)
for a in range(ACTION_SPACE_SIZE):
    vp_score[a] = CLASS_VP_VALUE[categorize(a)]
vp_score *= legal_mask                       # zero out illegal
if vp_score.sum() > 0:
    vp_target = vp_score / vp_score.sum()    # softmax-equivalent
else:
    vp_target = uniform(legal)               # all-zero day; no signal
```

Then add to the loss:
```python
loss += λ_vp * KL(softmax(logits) over legal || vp_target)
```

**λ_vp suggestion:** 0.10 (a real nudge, but MCTS visits remain dominant).

**What this teaches the model:** when a VP-generating action is legal
(BuildCity/BuildSettlement/PlayVpCard), prefer it over zero-VP actions
(trades, EndTurn). Directly addresses the cited city-upgrade gap.

**Risks:**
1. Forces the model toward "build city whenever legal" — but sometimes
   the right move is to save resources for a more impactful turn (e.g.,
   build 2 cities back-to-back next turn). This prior degrades to a
   greedy heuristic.
2. Doesn't help roads at all (they get 0 weight). Need Candidate 9 for that.
3. BuyDevCard's 1/14 weight is a guess; needs to be cited from the
   actual deck composition in `state.rs` (knight 14 / road-building 2 /
   monopoly 2 / year-of-plenty 2 / VP 5 = 25 total in v2; v3 may differ).
   **Investigation TODO before implementation.**

**Mitigation for risk 1:** condition the prior on resource availability.
Only push toward BuildCity if you actually have 3 ore + 2 wheat (cited
`scalars[SCALAR_HAND..SCALAR_HAND+5]`). If the resources aren't there,
the BuildCity action isn't legal anyway, so the legal_mask handles it.

### Candidate 9 (NEW) — Trajectory-conditioned road VP attribution (dataset-side, ~150 LOC)

For each road action by the recorded_player in a recorded game, walk
forward in the trajectory and credit the road with the VP of the next
settlement it enables. The cited problem this solves:

> Roads have zero direct VP. Their true value is the VP of the
> settlement they unlock that wouldn't have been unlockable otherwise.

**Algorithm (dataset-construction time):**

```python
def annotate_road_vp_credit(game_seed, action_history, recorded_player):
    """For each recorded road action, compute VP credit = was this road
    on the path to a future settlement?
    
    Cited engine model: a settlement is legal at vertex v iff:
      - v is empty (cited rules.rs:78)
      - no neighbor vertex has a settlement/city (distance rule, rules.rs:82)
      - in main phase: v is reachable from the player's roads (cited Catan rules)
    
    So a road at edge e provides "reachability" to the 1-2 vertices
    adjacent to e via VERTEX_EDGE_INDEX (cited adjacency.py).
    """
    eng = Engine(game_seed)
    road_credits = {}  # action_history_index -> credit (0.0 or 1.0)
    
    # Replay the game tracking which roads belong to recorded_player.
    # Track the set of vertices reachable_from_roads[recorded_player] over time.
    # When a new BuildSettlement(v) happens for recorded_player at step t', 
    # find the LATEST road r built by recorded_player at step t < t' such that
    # the road r was a NECESSARY component of v's reachability path.
    # Credit: road_credits[t] += 1.0
    
    return road_credits
```

This requires graph reachability tracking per-step. Manageable but ~150
LOC of careful implementation.

**Use the credits as a target-side modification:**

```python
# In dataset.py CatanReplayDataset.__getitem__:
if action_taken in BUILD_ROAD_RANGE:
    credit = road_credits.get(action_history_index, 0.0)
    # Blend with MCTS visit-count target:
    target[action_taken] = (1 - α_road) * mcts_visits[action_taken] / s + α_road * credit
```

**α_road suggestion:** 0.20 (significant; roads need this signal more
than other classes since they have no direct VP).

**What this teaches the model:** "Build the road that historically led
to a settlement, not roads that turned out to be wasted." This is the
trajectory-truth version of Candidate 3c (which used a static heuristic).

**Risks:**
1. Recorded games are one trajectory. A road that was wasted in this
   game might have been correct given the dice rolls that occurred.
   We're attributing credit based on hindsight rather than expected
   value at decision time.
2. Computational cost. Walking forward in each game's history per
   training sample is expensive. Mitigation: precompute road_credits
   ONCE per game during cache build, store alongside mcts_visit_counts
   in the parquet. **Requires cache rebuild** — significant infrastructure
   change. Could be deferred until next training cycle.
3. Credit assignment ambiguity. If a player builds 5 roads then builds a
   settlement, which roads get the credit? Options: the necessary
   road(s), all roads on the path, the most recent road. Need to define.
4. Doesn't generalize to longest-road strategy if v3 ever re-enables
   bonuses (cited current state: bonuses=false). For pure 5-VP no-bonus,
   roads exist only as settlement-enablers, so trajectory-attribution
   is conceptually clean.

### Combined effect of Candidates 8 + 9

- **Candidate 8** tells the model: "directly favor VP-generating actions
  when legal."
- **Candidate 9** tells the model: "for roads, the value is the VP of
  the settlement they enable in your eventual play."
- **Together:** the model learns to value all action types in proportion
  to their **eventual VP attribution**, not their action-space density.

This is the structural fix the user identified: **connect VP scoring to
the action space.**

---

## Decision matrix (still pending user input)

| Augmentation | Effort (LOC) | Risk | Likely impact | Order |
|---|---|---|---|---|
| 1. Pip-weighted settlement KL | ~50 | Medium | **Low** (data shows GNN already picks high-pip openings) | 4 |
| 2. City-upgrade target boost | ~30 | Medium | **Medium** (PureGnnB has clear gap; PureGnnA may not need this) | 3 |
| 3a. Edge-pip road KL (Option A) | ~80 | High (chain-of-roads risk) | **Medium** (addresses biggest gap, 11.6 vs 23) | 2 |
| 3c. Edge-pip road with trajectory truth | ~150 | Lower (true target) | **Higher** (unbiased signal) | 2' |
| 4. Ban ProposeTrade in training mask | ~15 | High (tanks if trades work) | **Unknown** until trade-success measured | depends on investigation #2 below |
| 5. ~~Setup placements added to dataset~~ | ~~100~~ | ~~Low~~ | **N/A — falsified 2026-05-09; setup IS in the dataset** | ❌ removed |
| 6. Regularization (dropout/wd/label smoothing) | ~20 | Low | **Medium-High** (addresses overfitting directly) | could be highest |
| 7. Action-class-balanced policy loss (NEW) | ~30 | Medium | **High** (directly addresses the 16% road in data → 4.4% road in play gap) | 2 |
| 8. Action-class VP prior (NEW from item E) | ~30 | Medium | **Highest for cities** (directly addresses 3.11 vs 6.32 city gap) | **1** |
| 9. Trajectory-conditioned road VP credit (NEW from item E) | ~150 + cache rebuild | Medium-High | **Highest for roads** (structural fix; teaches model that road value = enabled settlement VP) | 3 (after 8 lands) |

---

## Investigation backlog (cited TODOs before any code change)

1. **Read `catan_engine/src/state.rs`** to find ProposeTrade resolution rule.
2. **Measure ProposeTrade success rate** by walking action_history of the 1200 games and detecting state deltas after each ProposeTrade.
3. **Recount tournament wins under correct rotation convention** for the pass3 and pass3_lastepoch tournaments to validate or refute the GnnMcts claim.
4. **Re-examine overfitting** — confirm best_top1_epoch for each of the 9 grid_full20 cells. If all are at epoch 1-3, the regularization fix (option C in lateral thinking) becomes the priority over loss augmentation.
5. ~~**Read existing dataset.py to confirm setup placements are excluded** (dataset.py:96 cited above).~~ ✅ **DONE 2026-05-09** — `scratch_check_setup_samples.py` verified that setup placements ARE in the dataset (mi=0..3 are 100% setup actions, 7917 per worker). See revised lateral-thinking item A' above.
6. **Measure per-class loss contribution.** For one trained checkpoint, compute the masked-CE loss broken out by action class (Settle / City / Road / Trade / etc.). If road-class loss is much larger than ProposeTrade-class loss at convergence, that confirms hypothesis 1 in item A' (loss imbalance) and supports Candidate 7 as priority 1.
7. **Compare visit-count distribution by action class** in the e9 training data. Are MCTS visit counts on roads systematically lower than on trades? If yes, hypothesis 2 in item A' is supported (MCTS itself is biased) and Candidate 7 alone won't fix it — we'd need to retrain self-play with a different evaluator.

---

## Summary — what user has explicitly stated

- The settlement upgrade rate is bad (cited 3.11/100 vs 6.32/100 for PureGnnB) → wants reward for upgrades.
- Random wins because it takes action; inaction costs (cited 39.9% RollDice+EndTurn ratio for PureGnnB).
- Pip-average settlement should reward more (Candidate 1).
- Settle-upgrade-to-city should reward more (Candidate 2).
- Road placement is tricky; needs 2-step settlement reachability (Candidate 3).
- Concerned that adding road policy will mess up "what comes next" (chain-of-roads vs settle pickup).
- Suspects trades fail because of engine mechanics; proposes banning ProposeTrade in training, allowing only TradeBank.
- Wants this whole thinking documented so we can "think from external POV to crack more lateral thinking."

This document is that record.
