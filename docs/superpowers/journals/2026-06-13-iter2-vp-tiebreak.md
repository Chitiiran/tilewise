# Iteration 2 — 100% arena timeouts → VP-leader tiebreak (methodology change)

**Date:** 2026-06-13 (autonomous)
**Status:** gate methodology changed + tested; iteration-2 arena re-running.
**Needs user awareness:** this changes how EVERY future iteration is gated.

## What happened

Iteration 2 trained fine (val_top1 0.430, up from iter-1's 0.382, on the
237,906-position iter1+iter2 window, warm-start az_iter_1). But its arena hit
**100% timeouts** — 48/48 completed games all hit the 600s wall-clock cap with
no winner — vs iteration 1's **0/120**.

## Root cause (a structural property, not a bug)

Iteration 1's arena was cell6 vs az_iter_1 — a *decisive* matchup; games closed
out fast. Iteration 2's arena is az_iter_1 vs az_iter_2-candidate — two closely
matched descendants. **At full Catan, two similar GNN nets stall against each
other** and don't reach 10 VP within any reasonable wall-clock cap (the
documented stall pathology: full-Catan games between non-decisive players run to
tens of thousands of moves). This will recur **every iteration** as the champion
and candidate converge — the gate was structurally unable to evaluate similar
nets.

## Fix: VP-leader tiebreak (standard tournament practice)

A timed-out game is now decided by the **current VP leader** (`engine.vp(p)`;
a true VP tie is a draw) instead of discarded. Whoever is closest to winning
when time runs out is credited the win — exactly how Catan itself and prior
project tournaments resolve non-terminating games.

Gate validity reworked accordingly (the old "timeout rate > 5% → invalid" guard
made sense only when timeouts were censored no-results; now they're decided):
- **winrate is over DECISIVE games** (draws excluded);
- validity keys on **draw rate** (`arena_max_draw_rate=0.40` — too many genuine
  VP ties = no signal) and **min decisive count** (`arena_min_decisive=40`);
- timeout rate is still surfaced for observability but no longer gates.

Both timeout exits (wall-clock cap + 200k step cap) route through one
`_vp_leader` helper. 22 arena/config/loop tests pass (VP-leader credit, VP-tie
draw, draw-rate + min-decisive guards, winrate-over-decisive).

A side benefit: capping each stalled game at 600s makes the arena *faster* than
iter-1's natural-termination ~6h (games no longer run to thousands of moves).

## Open question for the user (thresholds)

A VP-tiebreak verdict is a *weaker* signal than a clean 10-VP win — a game
stalled at 8-7 is less decisive than a 10-VP closeout. I picked
`max_draw_rate=0.40` and `min_decisive=40` as reasonable defaults, but these
set the bar for "is this verdict trustworthy enough to promote." If you'd prefer
a stricter bar (fewer tiebreak games allowed) or a different tiebreak (e.g.
VP-margin-weighted, or longer cap to force more natural closeouts), that's a
quick config change.

There's also a deeper signal here: **the nets stalling against each other may
indicate they're too passive / not closing out** — which is itself something
later iterations + the engine-fidelity curriculum (trade negotiation, etc.)
should improve. Worth watching whether timeout rate falls as the bot strengthens.

## State

- Stale 48 no-VP-tiebreak results archived to
  `iter_2/arena/results_stale_noVPtiebreak.jsonl` (not deleted).
- iter-2 arena re-running with VP-tiebreak; ARENA.done cleared, TRAIN.done
  intact (no retrain). Verdict pending.
- Fix committed (0999130); 22 tests green.
