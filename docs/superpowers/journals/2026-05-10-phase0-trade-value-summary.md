# Phase 0 — Trade-value investigation results

**Date:** 2026-05-10
**Plan:** `C:\Users\chiti\.claude\plans\let-s-read-for-context-humming-storm.md`
**Script:** `mcts_study/scratch_trade_value.py` (new)
**Tests:** `mcts_study/tests/test_scratch_trade_value.py` (new, 5 tests, all passing)
**Raw output:** `runs/v3/archive/trade_value_analysis_2026-05-10/summary.json` (gitignored, regenerable)
**Corpus:** `runs/v3/data_gen/2026-05-05T05-50-e9_v3_data_gen_100k_w12/2026-05-05T09-50-e9_v3_data_gen/worker0..11/`

## Goal

Decide whether to ban ProposeTrade (action IDs 260..279) in training, per Cand 4 of the loss-augmentation roadmap. The doc's TODO #1 + #2 — "what does the engine do with ProposeTrade, and how often do those trades actually pay off?" — gated this decision.

## Engine resolution rule (TODO #1)

Cited `catan_engine/src/rules.rs:347-374`: engine iterates opponents from `(current_player + 1) % 4` in seat order; first opponent with ≥1 of `get` accepts a deterministic 1-for-1 swap. If none have it, the action is a silent no-op. There is no AcceptTrade / RejectTrade action.

## Methodology

For each game in the 100k corpus, replay the engine through `action_history` (handling chance-bit `0x80000000` outcomes via `apply_chance_outcome`). For each ProposeTrade encountered:

- **accepted:** snapshot proposer's hand via `engine.all_hands()[proposer]` before and after the step; flag accepted iff hand changed.
- **value_adding:** flag iff (accepted) AND the proposer subsequently executes one of {BuildSettlement, BuildCity, BuildRoad, BuyDevCard} before their next EndTurn (action 204).

Per user: roads count toward value-adding even though they have 0 direct VP in v3.

12-way `multiprocessing.Pool` over the 12 worker shards. 95,000 games scanned in 43.8 s wall-clock.

## Headline results

| Metric | Value |
|---|---|
| Games scanned | **95,000** |
| ProposeTrade actions total | **4,576,912** (~48/game) |
| Accepted | **4,576,912 (100.0%)** |
| Value-adding | **1,362,725 (29.8%)** |

### Per-give-resource breakdown

| Resource | n | Accept % | Value % |
|---|---:|---:|---:|
| wood | 1,077,804 | 100.0% | 29.3% |
| brick | 810,095 | 100.0% | 28.9% |
| sheep | 1,037,424 | 100.0% | 30.5% |
| wheat | 885,229 | 100.0% | 30.2% |
| ore | 766,360 | 100.0% | 29.9% |

Tight ±0.8pp band; no resource is materially better or worse than the others.

## Two surprising findings

### 1. 100% acceptance rate

Every single ProposeTrade in 4.58M actions across 95k self-play games was accepted. The engine's deterministic seat-order acceptance rule is virtually always satisfied — with 4 players each holding ~3-7 cards in mid-game, "no opponent has any of resource X" effectively never happens at the scale this corpus probes. **ProposeTrade is not a silently-failing action; it is a real 1-for-1 resource swap every time the proposer chooses it.**

This refutes one of the doc's hypotheses ("maybe trades aren't being accepted, so this is wasted action budget"). Trades are accepted; the question becomes whether they're being *used productively*.

### 2. ~30% value-adding (boundary call)

Of the 4.58M accepted trades, only 1.36M (29.8%) were followed by the proposer building a settlement / city / road or buying a dev card before their next EndTurn. The other 70% are "accepted but no immediate build" — the resource swap happened, but the proposer EndTurned without converting it to a VP-relevant action that turn.

Two interpretations are both consistent with this number:

- **(A) Many trades are wasted action budget.** ~70% of trades don't pay off the same turn. If the model is spamming trades for ambient resource shaping rather than imminent build moves, training on this would teach noise.
- **(B) Trades pay off in future turns.** A trade for ore at turn 12 could enable a city at turn 14 — outside the same-turn window. The 29.8% is a strict lower bound on "trades that ever pay off."

The script's design captures only the strict same-turn window (per the plan). Cross-turn correlations would require a richer analysis.

## Decision matrix (from plan)

| Value-adding rate | Action |
|---|---|
| ≥60% | Drop Cand 4 — trades are productive |
| 30-60% | Drop Cand 4 — keep trades (user tie-break: default no-ban) |
| <30% | Keep Cand 4 — ban ProposeTrade in training |

**Observed: 29.8%** — fractionally below the 30% threshold by 0.2pp. Mathematically on the "ban" side; substantively right on the boundary.

## Decision (user, 2026-05-10)

> "leave the trade as is. no change needed we will let the new VP based backpropagation handle this"

**Cand 4 DROPPED from the roadmap.** Trades remain in the training mask unchanged. The plan's Cand 10 (1-step VP-comparison rule) is expected to handle over-trading implicitly: when the model picks ProposeTrade and a VP-grant action would have yielded strictly higher VP one step later, the comparison rule swaps the supervised target away from the trade toward the VP action. Trades that *do* pay off (the 29.8% same-turn or some additional cross-turn fraction) survive because the VP comparison would tie or favor the trade.

Rationale captured: 29.8% is too close to the boundary to be confidently directional, and the 100% acceptance rate proves trades are real economic activity rather than silent failures. A blanket ban risked tanking the model's resource throughput. Letting Cand 10 do the work surgically is the principled call.

## Test coverage

5 unit tests in `mcts_study/tests/test_scratch_trade_value.py`:

1. `test_classifier_returns_empty_for_no_trades_history` — setup-only prefix yields no trade records.
2. `test_classifier_count_matches_history` — record count equals raw ProposeTrade count in action_history.
3. `test_classifier_records_have_required_fields` — schema check.
4. `test_classifier_value_adding_implies_accepted` — invariant check.
5. `test_classifier_first_trade_accepted_flag_matches_independent_replay` — cross-checks accepted flag against independent engine replay.

Bug-injection sanity test verified: dropping every other trade from the classifier output causes test 2 to fail with `26 == 51` AssertionError (pre-restore).

All 5 tests pass post-implementation in 2.94s.

## Reproduction

```bash
# WSL Ubuntu, venv at ~/catan_mcts_venvs/mcts-study/
source ~/catan_mcts_venvs/mcts-study/bin/activate
cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study

# Smoke (100 games, single shard, ~1s)
python scratch_trade_value.py --shards 0 --max-games-per-shard 100 --pool 1 --out /tmp/smoke

# Full 95k games, 12-way pool (~44 s wall clock)
python scratch_trade_value.py --pool 12 --out runs/v3/trade_value_analysis_2026-05-10
```
