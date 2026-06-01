# Robber Steal — Victim Choice — Design

**Date:** 2026-06-01
**Worktree:** `interactive-play` (branch `worktree-interactive-play`)
**Status:** design — implementation gated on user approval

## 1. Background & problem

In the interactive play-vs-bots web app, when the human moves the robber onto a
hex with **two or more** eligible opponents (opponents with a building on that
hex and a non-empty hand), they are never asked **whom to rob**. The engine
auto-picks the victim.

Root cause (engine, "Tier 1" simplification):

- After `MoveRobber`, the engine enters `GamePhase::Steal { from_options }`,
  which `is_chance_pending()` reports as a **chance node**
  (`catan_engine/src/engine.rs:147-150`).
- `chance_outcomes()` enumerates the cards of **`from_options[0]` only** — the
  first eligible victim in seat order (`engine.rs:179-191`).
- `apply_chance_outcome()` **hard-asserts** `victim == from_options[0]`
  (`engine.rs:209-219`) and panics otherwise.

Because the victim is forced to the first option and the engine rejects any
other, **no Python-layer intercept can override the victim** — unlike the
auto-resolved player trades (which a Python intercept *can* handle). Fixing this
requires an engine change. See memory `project-engine-trades-auto-resolve` for
the related trade simplification and this steal note.

Both consumers resolve chance nodes the same generic way — call
`state.chance_outcomes()`, sample by probability, `apply_action(value)`:
`mcts_study/catan_mcts/experiments/common.py:33-78` (self-play) and
`mcts_study/catan_mcts/web/game_session.py:179-206` (web). So any change to the
chance **distribution** is automatically honored by self-play and bot turns.

## 2. Goal & non-goals

**Goal:** when the **human** moves the robber onto a hex with 2+ eligible
victims, let them choose whom to rob (the stolen card stays random per Catan
rules). Bots steal from the eligible victim holding the **most cards**.

**Non-goals / constraints:**

- **Do NOT change the action space.** `ACTION_SPACE_SIZE` stays `280`
  (`catan_engine/src/actions.rs:49`). Steal is a chance outcome, never a
  decision action, so trained GNN bots (policy head over the 280 actions) are
  **unaffected** and need no retraining.
- No change to `MoveRobber`, the observation tensor, or the legal-action set.
- Single-victim and zero-victim steals behave exactly as today (unambiguous).

## 3. Approach (chosen of 3)

1. **Add a `StealFrom(player)` decision action** — expands the 280-action
   space, breaking every trained bot's policy head + forcing retraining.
   **Rejected.**
2. **Pure uniform chance distribution over all victims×cards** — would make
   self-play steal a *random* victim automatically, but cannot express "bots
   steal from the most-cards victim," and gives the human no control (a generic
   sampler picks). **Rejected.**
3. **✅ Relax the victim assertion + carry the chosen victim in the chance
   value.** Keep Steal a chance node, keep 280 actions. The engine's default
   chance distribution targets the **most-cards** victim (so generic samplers —
   self-play and bot turns — steal greedily from them). `apply_chance_outcome`
   accepts **any** victim in `from_options`, so the **human path** in the web
   session can intercept the Steal node and apply the human-chosen victim.
   **Chosen.**

## 4. Engine changes (`catan_engine/src/`)

All keep `ACTION_SPACE_SIZE = 280`.

### 4a. `chance_outcomes()` Steal branch — default to most-cards victim
`engine.rs:179-191` currently enumerates `from_options[0]`'s cards. Change the
victim selection to the eligible victim in `from_options` holding the **most
total cards** (ties broken by seat order — i.e. the first in `from_options`
among those tied for the max). The value encoding is unchanged:
`victim * 256 + card_index`, with each of that victim's cards an equally-likely
outcome. Generic samplers therefore steal from the most-cards victim — the
required bot behavior — with no sampler change.

### 4b. `apply_chance_outcome()` Steal branch — accept any eligible victim
`engine.rs:209-219`: replace `assert_eq!(victim, from_options[0], …)` with a
**membership + validity** check:
- `assert!(from_options.contains(&victim), …)`
- victim's hand non-empty and `card_index` within the victim's flat card index
  range (the existing `apply_steal` already asserts the latter at
  `rules.rs:958-985`).

This is the unblock: the engine now accepts a steal from any eligible victim,
which the human path uses.

### 4c. PyO3 accessor `steal_options()`
Add to `catan_engine/src/lib.rs` a method
`steal_options(&self) -> Vec<(u8, u8)>` returning, **only at a Steal chance
node**, `(victim_seat, total_card_count)` for each seat in `from_options`
(in `from_options` order). At any non-Steal state it returns an empty vec. This
lets the orchestration layer present the victim modal and build the chance value
without reconstructing `from_options` from the board.

### 4d. Rebuild
After the Rust edits, run `maturin develop --release` from the worktree's
`catan_engine` (per memory `feedback-rebuild-pyo3-after-engine-changes` — pytest
is a false negative without the rebuild). WSL build, then the venv sees the new
`_engine`.

### Impact on existing self-play
The only behavioral change to current self-play is that a multi-victim steal now
targets the most-cards victim instead of the first — a strict realism
improvement. Previously-recorded data is unaffected; newly generated data
carries this minor, principled bias.

## 5. Python orchestration (`mcts_study/catan_mcts/web/game_session.py`)

Mirrors the existing trade-intercept (`_pending_trade` / `respond_to_trade`).

### 5a. Intercept the human Steal chance node in `advance()`
At a chance node, before sampling: if it is a **Steal** node **and**
`current_player == human_seat` (the robber-mover is still the current player at
the Steal node) **and** `len(steal_options()) >= 2`, then **pause** — store
`_pending_steal = [(seat, cards), …]` and return `state_json()` with status
`steal_choice`. Otherwise sample-and-apply as today (which, post-4a, steals from
the most-cards victim for bot/auto paths; a single victim is unambiguous).

### 5b. `respond_to_steal(victim_seat)`
- Validate `victim_seat` is one of `_pending_steal`'s seats; else
  `raise ValueError("invalid steal victim")`.
- Choose `card_index` **uniformly at random from the victim's hand** using the
  session RNG (`self._rng`) — the stolen card is random per Catan; the human
  only picks the victim. `card_index` is the flat index into the victim's hand
  as `chance_outcomes`/`apply_steal` expect.
- Build `value = victim_seat * 256 + card_index` and
  `self._state.apply_action(value)` (the adapter routes a chance value through
  `apply_chance_outcome`). Clear `_pending_steal`. Then drive on.
- Async variant `respond_to_steal_async(victim_seat)` returns immediately and
  calls `advance_async()` (mirrors `respond_to_trade_async`); the sync
  `respond_to_steal` is kept for tests.

### 5c. `state_json()` additions
- New status `steal_choice` (from `_status()` when `_pending_steal` is set,
  ordered like `trade_offer`).
- When status is `steal_choice`, include
  `"steal_choice": {"options": [{"seat": s, "name": seat_names[s], "cards": n}, …]}`.

### 5d. Card-index mapping
`apply_steal` (`rules.rs:958-985`) treats `card_index` as a flat index across the
victim's hand in resource order (0..total_cards). The session computes
`card_index = rng.randrange(total_cards)` for the chosen victim, where
`total_cards` comes from `steal_options()` (or `all_hands()[victim].sum()`).

## 6. Server API (`mcts_study/catan_mcts/web/server.py`)

One new endpoint, mirroring `trade-response`:

| Method & path | Body | Action | Errors |
|---|---|---|---|
| `POST /api/games/{id}/steal-response` | `{victim: int}` | `respond_to_steal_async(victim)` | `ValueError → 409` |

No other server change — `state_json` already carries the `steal_choice` block.

## 7. Frontend (`mcts_study/catan_mcts/web/static/play.js` + `style.css`)

A steal-victim modal, reusing the trade-modal styling and the 5-second timer
pattern already built for trades.

- In `applyStateNoStream`, when `status === 'steal_choice'`, show a modal:
  *"Choose who to rob:"* with one button per option, labeled
  `"{seat_name} — {cards} cards"` and color-coded by seat
  (`.seat-{i}`). Clicking posts `{victim: seat}` to `/steal-response` and applies
  the returned state.
- **5-second auto-pick timer** (matches the trade modal): a visible countdown +
  progress bar; at 0, auto-pick the **most-cards** victim (the option with the
  largest `cards`; ties → first listed). A manual click cancels the timer. The
  timer is cleared on any non-`steal_choice` state and before opening a fresh
  modal (reuse the `clearTradeTimer` discipline; add a parallel
  `clearStealTimer`).
- `renderStatus`: add `steal_choice → "Choose who to rob"` (use the
  attention-grabbing `status-alert` styling, like the robber prompt).
- SSE/async: `steal_choice` is a yield point (treated like `trade_offer` /
  `your_turn`), not `bot_thinking`. `respondSteal(victim)` mirrors
  `respondTrade`: clear timer, remove modal, POST, apply.

## 8. Error handling

- Invalid victim (not in options / stale) → server `409` → client re-fetches
  `/state` and re-renders (same pattern as `postAction`'s 409 path).
- A steal node that is **not** a human 2+victim case never pauses — it is
  sampled, so there is no modal and no hang.
- Engine: `apply_chance_outcome` asserts victim membership + card range; an
  out-of-range value is a programming error (the session only ever builds valid
  values), surfaced as a 500 with a traceback in the log (acceptable — not a
  user-reachable path).

## 9. Testing

**Rust** (`catan_engine/tests/`, a new `robber_steal_victim.rs`):
- Robber onto a hex with 2 eligible victims of **different** hand sizes →
  `chance_outcomes()` enumerates the **most-cards** victim's cards (default).
- `apply_chance_outcome` with the **non-default** (smaller-hand) victim's value
  **succeeds** (no panic) and moves a card from that victim.
- The old "always first victim" is no longer forced (a non-first victim is
  accepted).
- Zero-victim steal → phase returns to `Main` (unchanged).

**Python** (`tests/test_game_session.py`):
- Drive (or craft) a 2-victim human Steal → `advance()` returns `steal_choice`
  with both options and does **not** auto-resolve.
- `respond_to_steal(chosen)` removes exactly one card from the **chosen**
  victim and none from the other.
- A 1-victim or **bot** robber-move does **not** pause (auto-resolves).
- Invalid victim → `ValueError`.

**API** (`tests/test_web_api.py`):
- `POST /steal-response` happy path returns a live state; bad victim → `409`.

**Frontend** (headless browser, the existing Playwright harness):
- Inject a `steal_choice` state → assert the modal lists the options with names +
  card counts; clicking a victim POSTs `{victim}`; the 5s timer auto-picks the
  most-cards option if untouched.

## 10. Files touched

- `catan_engine/src/engine.rs` (chance_outcomes + apply_chance_outcome Steal
  branches), `catan_engine/src/lib.rs` (`steal_options` PyO3 method);
  `catan_engine/tests/robber_steal_victim.rs` (new). Rebuild via
  `maturin develop --release`.
- `mcts_study/catan_mcts/web/game_session.py` (intercept + `respond_to_steal`
  [+async] + `state_json` + `_status`).
- `mcts_study/catan_mcts/web/server.py` (`/steal-response`).
- `mcts_study/catan_mcts/web/static/play.js` + `style.css` (modal + timer +
  status).
- Tests in `catan_engine/tests/`, `mcts_study/tests/test_game_session.py`,
  `test_web_api.py`.

## 11. Implementation plan handoff

Tests-first per project convention. Engine first (Rust tests + rebuild), then
orchestration, then server, then frontend. Next step: invoke the `writing-plans`
skill to produce the TDD-ordered task list.
