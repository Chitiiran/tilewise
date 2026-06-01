# Interactive "Play vs Bots" Mode — Design

**Date:** 2026-05-31
**Worktree:** `interactive-play` (branch `worktree-interactive-play`)
**Status:** design — implementation gated on user approval

## 1. Background

Today the project has an **offline replay viewer** — `mcts_study/catan_mcts/playback.py`
emits a single self-contained `index.html` that replays a *pre-recorded* game
from parquet: a static board PNG + per-step SVG overlays + a player panel
(hands, dev cards, VP, longest road / largest army, bank). It has no server;
you double-click the HTML and scrub through a finished game.

This spec adds a fundamentally different, **live interactive mode**: the user
*plays* a real Catan game as one of the four seats, choosing actions in real
time, against bots they select per seat. The user must also respond to trade
requests from bots. Because a live game must run the Rust engine and the bots
on every move (and must pause for human input), this cannot be a static HTML
file — it needs a small local web server. We host both modes in one website
with nav tabs ("Play" and "Replay"), reusing the replay's board-rendering and
state-serialization code.

The eventual goal is to publish this as a live website, so the architecture is
**local-first but deploy-ready**: a clean REST + SSE API, a static frontend,
configurable paths, and one game per server-side session — so a later deploy is
mostly hosting/config, not a rewrite. Multi-user, auth, and concurrent-game
scaling are explicitly out of scope for this iteration.

## 2. The hard constraint: engine auto-resolves trades

The Rust engine (`catan_engine/src/rules.rs:347-374`, verified by
`catan_engine/tests/player_trades.rs`) resolves a `ProposeTrade { give, get }`
action (action ids 260-279) **instantly on apply**: it scans opponents in seat
order (`current_player + 1, +2, +3`); the **first** opponent holding ≥1 of
`get` auto-accepts a 1-for-1 swap; if none holds it, nothing moves.
`MAX_TRADES_PER_TURN = 1`.

There is **no pending-trade game phase, no response action, and no chance node**
for trade acceptance. The engine therefore cannot natively pause for a human to
accept or reject a bot's trade offer.

**Consequence (decided):** the human trade-response UX lives entirely in the
**Python orchestration layer**, not the engine. No Rust changes, no engine
rebuild. The session layer intercepts a bot's `ProposeTrade` *before* applying
it, determines whether the engine would auto-match the human, and if so pauses
for the human's Accept/Reject. See §4.

## 3. Architecture & components

A thin local web server (FastAPI) holds the live game in a per-game session; a
single-page vanilla-JS frontend talks to it over REST + SSE. No server-side HTML
templating of game state, no page reloads mid-game.

New package layout:

```
mcts_study/catan_mcts/web/
├── server.py          FastAPI app: serves the frontend + REST/SSE API. No game logic.
├── game_session.py    GameSession: owns one live CatanState + bots + the trade-intercept
│                      driving loop. The heart of the feature.
├── bot_registry.py    Discovers bot types + GNN .pt checkpoints; builds a bot from a spec.
├── board_layout.py    Board geometry + static board PNG (EXTRACTED from playback.py, shared).
├── serializers.py     Engine state → client JSON (EXTRACTED from playback.py's _replay_to_states).
├── action_decode.py   Decode raw action ints → {id, label, kind, target} for the UI.
└── static/
    ├── index.html     Nav tabs: [Play] [Replay] + shared shell.
    ├── play.js        Setup (lobby) + live game UI (board, action panel, trade modal, log).
    ├── replay.js      Replay browser: list + serve existing replays, optionally generate new.
    └── style.css      Shared styles, lifted from playback.py's CSS + grid_dashboard.html idiom.
```

**Code-reuse refactor (a targeted improvement to code we're working in).**
`playback.py` is a single ~1050-line file mixing four concerns: board geometry,
static-PNG rendering, per-state serialization, and HTML/JS emission. We extract
the first three into `board_layout.py` and `serializers.py` so the offline
replay and the live server share **one source of truth** for board coordinates,
the board PNG, and the per-state dict shape. `playback.py` is then rewritten to
import from these modules and keep only its HTML-emission role. This is scoped
to what the live server needs — not a gratuitous rewrite — and is guarded by a
golden test (§8) so the replay output does not regress.

**Component responsibilities:**

- **`GameSession`** — one object per game. Owns the `CatanState`, the three bot
  instances, and which seat is human. Public surface:
  - `state_json()` → full client state object (§6)
  - `apply_human_action(action_id)` → validate ∈ legal, apply, then `advance()`
  - `respond_to_trade(accept: bool)` → resolve a pending trade, then `advance()`
  - `advance()` → run chance + bot moves until the game reaches the human's
    turn, a trade needs the human's response, or terminal. Where trade-intercept
    lives (§4).
- **`bot_registry`** — `list_available()` returns bot types + discovered `.pt`
  checkpoints under a configurable dir; `build(spec, game, seed)` constructs a
  bot instance. Validates a GNN checkpoint loads at build time.
- **`server`** — thin HTTP/SSE layer; routes map 1:1 to session methods. Holds
  the `{game_id: GameSession}` registry. No game logic.

## 4. Game loop & trade interception

`GameSession.advance()` mirrors `mcts_study/catan_mcts/experiments/common.py`'s
`play_one_game`, but **cooperatively yields** instead of running to terminal:

```
advance():
  loop:
    if state.is_terminal():            return {status: "game_over", returns}
    if state.is_chance_node():
        outcome = sample(state.chance_outcomes(), self._rng)   # server-owned RNG
        state.apply_action(outcome);   continue
    cp = state.current_player()
    if cp == human_seat:               return {status: "your_turn", legal_actions}
    # --- bot seat ---
    action = bot[cp].step(state)
    if _is_propose_trade(action) and _engine_would_match_human(state, action):
        self._pending_trade = (cp, action)
        return {status: "trade_offer", from_seat: cp, you_give, you_get}
    state.apply_action(action);        continue
```

**`_engine_would_match_human(state, action)`** replicates the engine's
seat-order scan in Python (read `all_hands()`): for the bot's
`ProposeTrade{give,get}`, find the first opponent in order `cp+1,+2,+3` holding
≥1 of `get`. Return `True` iff that first match is the human seat. (If a bot
ahead of the human in seat order also holds `get`, the engine would pick *that*
bot, so we do **not** prompt — staying faithful to actual engine behavior.)

**`respond_to_trade(accept)`:**
- **Accept** → apply the pending `ProposeTrade` action normally. The human is
  the first matched holder, so the engine performs the 1-for-1 swap. Then
  `advance()`.
- **Reject** → do **not** apply the trade (it would auto-swap with the human).
  Mask the rejected `ProposeTrade` action id and **re-query the bot once** for a
  new action on a masked view of legal actions; if the bot has no other useful
  move, apply `EndTurn` (action 204). Then `advance()`. The human's hand is
  unchanged. Clear `_pending_trade`.

**Other forced human responses** (discard on a 7 → actions 199-203; move robber
→ 180-198; steal target) are ordinary `your_turn` legal actions under full
manual control — no special-casing.

**Concurrency.** One game per session, single-threaded per session. Bot moves
(MCTS / GNN-MCTS) can take seconds; `advance()` runs in a worker thread so the
server stays responsive. SSE pushes a `bot_thinking` status immediately and the
final state when `advance()` returns. A configurable per-move timeout
(generous default) guards against an infinite hang.

## 5. Rules & bot selection

- **Rules** — full Catan by default (`vp_target=10`, `bonuses=True`). The setup
  screen exposes `vp_target` (10/5) and `bonuses` (on/off) so the v3 short
  variant (`vp_target=5`, `bonuses=False`) is also playable. Rankings of bot
  strength invert across rulesets (project memory) — informational only; we do
  not constrain which bot plays which ruleset.
- **Seating** — four seats P0-P3, fixed turn order. The human picks their seat;
  each other seat gets an independent bot spec.
- **Bot roster** — Random, Greedy baseline, LookaheadMctsV3, PureGnn, GNN-MCTS.
  For GNN types the user picks **any `.pt` from the training library**: the
  registry scans a configurable checkpoints dir and exposes the list; the UI
  shows a second dropdown of discovered checkpoints. (Checkpoints currently live
  on the WSL filesystem; the dir is a config value, never hardcoded, for
  deployability.)

## 6. API & client state

**REST + SSE (FastAPI, JSON):**

| Method & path | Purpose | Returns |
|---|---|---|
| `GET /api/bots` | Bot types + discovered checkpoints | `{types, checkpoints:[{name,path}]}` |
| `POST /api/games` | Create a game from a setup spec | `{game_id, state}` |
| `GET /api/games/{id}/state` | Current full state (reconnect/poll) | `state` |
| `POST /api/games/{id}/action` | Apply human action `{action:int}` | `state` after advancing |
| `POST /api/games/{id}/trade-response` | Answer pending trade `{accept:bool}` | `state` after advancing |
| `GET /api/games/{id}/events` | **SSE** stream: pushes state / `bot_thinking` | event stream |

We use **SSE** (one-directional server→client) for live updates, consistent
with the dashboard's SSE migration (project memory).

**Setup spec** (`POST /api/games` body):
```json
{
  "human_seat": 0,
  "seats": {
    "1": {"type": "PureGnn", "checkpoint": "cell6_ep10.pt"},
    "2": {"type": "Random"},
    "3": {"type": "LookaheadMctsV3"}
  },
  "rules": {"vp_target": 10, "bonuses": true},
  "seed": 12345
}
```
`seed` is optional (random if omitted). Seats other than `human_seat` must each
carry a spec.

**Client state object** (returned by every state-changing call; pushed over SSE).
The `state` sub-object is **byte-for-byte the same shape** as `playback.py`'s
per-step dict, so the frontend reuses the replay's `renderState()` rendering:

```json
{
  "status": "your_turn" | "bot_thinking" | "trade_offer" | "game_over" | "error",
  "current_player": 0,
  "human_seat": 0,
  "phase": "Main",
  "narration": "P1 BuildSettlement(v=12)",
  "state": { "s": [], "c": [], "r": [], "rh": -1, "vp": [],
             "hands": [], "bank": [], "dev_held": [], "ports": [],
             "lr_len": [], "knights": [], "built": [],
             "lr_holder": -1, "la_holder": -1, "vp_played": [] },
  "legal_actions": [ {"id": 0, "label": "BuildSettlement(v=0)",
                      "kind": "build_settlement", "target": 0} ],
  "trade_offer": { "from_seat": 1, "you_give": [2, 1], "you_get": [0, 1] },
  "seat_names": ["You", "P1 PureGnn", "P2 Random", "P3 LookV3"],
  "returns": [-1, 1, -1, -1]
}
```

- `legal_actions` present only when `status == your_turn`.
- `trade_offer` present only when `status == trade_offer`. From the human's
  perspective: `you_give = [get_idx, 1]` (the bot's requested resource, which
  the human hands over) and `you_get = [give_idx, 1]` (the bot's offered
  resource, which the human receives). Both are `[resource_idx, qty]` and qty is
  always 1 (engine does 1-for-1 only).
- `returns` present only when `status == game_over`.
- `board` layout + base64 PNG sent once at game start (separate field on the
  create response), not on every update.

**Action enrichment** (`action_decode.py`): the engine returns raw action ints.
The server decodes each legal action into `{id, label, kind, target}` using the
existing `_action_desc()` plus a `kind` (build_settlement / build_city /
build_road / roll / trade_bank / propose_trade / play_dev / move_robber /
discard / end_turn) and a `target` (vertex / edge / hex id, or null for
non-spatial actions). This drives the clickable board: spatial targets light up
and are clicked; non-spatial actions are buttons.

## 7. Frontend / UI

Single-page vanilla JS (no build step — matches existing `playback.py` /
`grid_dashboard.html`). Visual style matches the project: `system-ui`, white
panels with `#ddd` borders, the four seat colors
(`#cc3333 / #3366cc / #33aa55 / #cc8833`), `#ffd633` amber for current/active,
the existing resource/dev-card emoji set. Two nav tabs:

**Tab 1 — Play.**

- **Setup screen (lobby, colonist.io-style):** a 4-seat table; a radio picks
  which seat is "You"; each other seat has a bot-type dropdown, and selecting a
  GNN type reveals a second dropdown of discovered `.pt` checkpoints. A rules row
  (VP target, bonuses, optional seed). **Start Game** → `POST /api/games` →
  game screen.
- **Game screen:**
  - **Left — board:** the static PNG + live SVG overlay, identical rendering to
    `playback.py`, but **interactive** — on your turn, legal spatial targets
    (settlement vertices, road edges, robber hexes) highlight and are clickable.
  - **Right — player panel:** the replay's seat strip + VP/hand/dev/bank table,
    same component.
  - **Action panel (below board):** buttons for non-spatial legal actions
    (🎲 Roll, Buy Dev Card, Bank Trade ▸, Propose Trade ▸, Play
    Knight/Mono/YOP/RoadBldg, End Turn). Greyed when not your turn.
  - **Trade modal:** on a `trade_offer`, a centered modal — *"P1 (PureGnn)
    offers: you give 🐑×1, you get 🌲×1"* with **Accept** / **Reject**. Blocks
    until answered.
  - **Live log:** a scrolling narration feed (reusing `formatNarration()`).
  - **Status line:** "Your turn" / "P2 (Random) thinking…" / "You win 🎉".

**Trade UI depth (decided): match the engine exactly.** Propose Trade is a
submenu to pick give-resource + get-resource (1-for-1), mapping straight to the
legal `ProposeTrade` action ids; Bank Trade picks give+get at the player's best
available ratio. No fake multi-resource trade builder — the engine only does
1-for-1 auto-matched swaps, so the UI reflects exactly that.

**Tab 2 — Replay.** Lists existing replay outputs (server scans for
`playback_seed_*/index.html` dirs) and links to them; optionally a form to
generate a new replay from a run_dir + seed by calling `playback.render()`.
Keeps the existing replay viewer reachable "in the same website" without
rewriting it.

**colonist.io reference (patterns, not assets):** lobby per-seat opponent
selection, click-the-board-to-build, the trade-offer modal, and the action
button bar. We use the project's own board rendering — no third-party art.

## 8. Error handling

- **Stale/invalid human action** → server validates `action ∈ legal_actions()`;
  if not, `409` + current state; client re-renders. No crash.
- **Bad/missing checkpoint** → validated at game creation (`POST /api/games`),
  not mid-game → `400` with a clear message on the setup screen.
- **Bot exception during `advance()`** → caught per-move; surfaced as
  `status: "error"` with a friendly "bot P2 errored" message; full traceback to
  the server log. The game does not hang.
- **Long bot moves** → `advance()` in a worker thread; immediate `bot_thinking`
  SSE push; configurable per-move timeout guards an infinite hang.
- **Reconnect / refresh** → `GET /api/games/{id}/state` returns full current
  state (session held server-side, keyed by `game_id`), so a refresh resumes.
- **Chance RNG** → server-owned, seeded from the game seed; reproducible, and a
  refresh never re-rolls.

## 9. Testing (TDD — tests first, per project workflow)

In `mcts_study/tests/`:

- **`test_serializers.py`** — the extracted serializer produces the **same dict
  shape** as the current `_replay_to_states` (golden test against current output;
  guards the refactor).
- **`test_bot_registry.py`** — lists bot types; discovers `.pt` checkpoints from
  a fixture dir; builds each non-GNN bot; rejects a bad checkpoint with a clear
  error.
- **`test_game_session.py`** (core):
  - A full game with the human seat driven by a "pick first legal action" stub
    plays to terminal without error.
  - **Trade intercept — targets human:** craft a state where a bot's
    `ProposeTrade` first-matches the human → `advance()` returns `trade_offer`
    and does **not** auto-apply. `Accept` performs the swap; `Reject` masks the
    trade + re-queries the bot and leaves the human's hand unchanged.
  - **Trade intercept — targets another bot:** a bot ahead of the human holds
    `get` → no pause; applied normally.
  - `apply_human_action` rejects an illegal action.
  - State JSON contains every field the replay renders (contract guard).
- **`test_web_api.py`** — FastAPI `TestClient`: create game → get state → apply
  action → trade-response → game-over, asserting status codes and the state
  contract; the SSE endpoint emits ≥1 event.
- **No-regression:** existing `mcts_study/tests/test_playback.py` still passes
  after the shared-module extraction.
- **Frontend smoke (optional/manual):** the `webapp-testing` (Playwright) skill —
  load setup, start a Random-only game, play a few clicks, assert the board
  renders and End Turn advances. Marked optional if Playwright isn't wired for
  WSL; the Python API tests are the hard gate.

## 10. Dependencies & runtime

- **New dependency:** FastAPI + an ASGI server (uvicorn). SSE via FastAPI's
  `StreamingResponse` (no extra dep). Add to the mcts_study env.
- **Runtime:** WSL Ubuntu (where torch, the GNN checkpoints, and the
  maturin-built engine already live); accessed from the Windows browser via
  `localhost`. All paths (checkpoints dir, replay-output dir) are config values —
  no hardcoded WSL paths — so a later deploy is hosting/config only.
- **No engine changes; no maturin rebuild.** Pure Python + frontend.

## 11. Out of scope (this iteration)

- Multi-user, authentication, concurrent-game scaling, hosting/deploy itself.
- Engine changes (trade-pending phase, multi-resource trades).
- Modifying the existing offline replay's *visual* behavior (only refactor its
  internals into shared modules; output unchanged).
- Game persistence/save-resume beyond an in-memory server-side session.

## 12. Implementation plan handoff

This design becomes: (1) a shared-module extraction from `playback.py` with a
golden regression test; (2) a new `web/` package (`bot_registry`,
`game_session`, `action_decode`, `serializers`, `server`); (3) a static
frontend (`index.html`, `play.js`, `replay.js`, `style.css`); (4) the test
suite in §9. Tests-first per project convention. Next step: invoke the
`writing-plans` skill to produce a TDD-ordered task list.
