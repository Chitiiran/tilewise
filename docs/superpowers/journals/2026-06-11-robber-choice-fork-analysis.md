# Robber-as-a-choice — the two interpretations (analysis, not yet implemented)

**Date:** 2026-06-11 (while iteration 1 trains)
**Context:** AZ-loop spec §4 lists F1 = "robber steal-victim becomes a decision."
The approved 2026-06-01 steal-victim spec and the user's stated intent point at
*different* changes. Capturing the fork before implementing so the curriculum
stage is the right one.

## The user's literal intent (planning session, verbatim)

> "currently the engine doesn't have robber as a choice, because the game state
> will be too big. but as we go forwards we can find better ways to make the
> state smaller and introduce them slowly. what you are describing are the bot's
> strategy, what i'm asking for is slowly improving the engine to resemble real
> life games so that the training allows strategies to be learnt by self-playing."

Key phrase: **"so that training allows strategies to be learnt by self-playing."**
The robber TARGET must become a **learned decision** the policy head can optimize
— not just a smarter auto-rule.

## Two interpretations

### Interpretation A — the 2026-06-01 spec (option 3, "chance node stays")
- `chance_outcomes()` Steal branch defaults to the **most-cards** victim instead
  of `from_options[0]`.
- Steal stays a **chance node**; `ACTION_SPACE_SIZE` stays **280**; trained bots
  need **no retraining**.
- Effect on self-play: a strict realism nudge (greedy steal target). The bot
  does **NOT learn** whom to rob — it's a fixed heuristic baked into the engine.
- This is what the existing spec delivers. It was scoped for the WEB APP (human
  picks victim via modal); the self-play-relevant part is just the 4a default.

### Interpretation B — what the user actually asked for (the rejected option 1)
- Robber targeting (which hex + which victim) becomes **decision actions** the
  policy head ranges over, so self-play **learns** robber strategy.
- This **expands the action space** (e.g. MoveRobber already has 19 hex actions
  at ids 180–198 — those ARE already learned; the missing learned choice is the
  VICTIM). Adding `StealFrom(victim)` decision actions grows 280 → ~283 and
  forces the planned single action-space expansion + retraining.
- The user explicitly accepted the state-size cost ("the game state will be too
  big ... introduce them slowly").

## Important nuance discovered while reading the engine

`MoveRobber(hex)` is ALREADY a learned decision (actions 180–198, engine.rs).
So the bot already chooses WHERE to put the robber. The ONLY non-learned part is
the **victim** when a hex has 2+ eligible victims — and that is exactly the
chance-node steal. So "robber as a choice" reduces precisely to: **make the
steal-victim a decision action** (interpretation B) vs **a smarter chance
default** (interpretation A).

## Recommendation for the curriculum

1. **Now (iteration 1 baseline):** change nothing — let the loop establish the
   Elo-vs-games slope on the current rules. Engine edits need `maturin develop`
   which disrupts the running venv; do not touch mid-training.
2. **Cheap intermediate (retraining-free):** ship spec option 3's 4a default
   (most-cards victim) — strict realism gain, no action-space change, the loop
   absorbs it on the next self-play stage. Good "F0.5".
3. **The real F1 (interpretation B):** at the planned single action-space
   expansion event, add `StealFrom(victim)` as decision actions so self-play
   LEARNS robber targeting. This is the user's actual ask and the first true
   fidelity-grows-the-strategy stage. Sequence it once the baseline Elo exists
   and the B1 throughput verdict is in (more games/iter makes the bigger action
   space learnable).

## Decision status

This is a design fork that changes the action space and forces retraining — it
is **not** an autonomous call. Surface to the user with this analysis when
iteration 1 completes; let them confirm B (and its timing) before implementing.
The A-style 4a default can proceed autonomously as the intermediate step since
it's retraining-free and matches an already-approved spec.
