# Interactive Play-vs-Bots

A local web app to play Catan as one seat against selectable bots, with
human-in-the-loop trade responses. Shares board rendering with the offline
replay viewer (`catan_mcts.playback`).

## Run (WSL, mcts-study venv active, from `mcts_study/`)

    pip install -e ".[web]"        # one-time: fastapi/uvicorn/httpx
    python -m catan_mcts.web \
        --checkpoints-dir /path/to/gnn/checkpoints \
        --replays-dir     /path/to/playback/outputs \
        --port 8000

Open http://localhost:8000 in your browser.

- **Play tab:** pick your seat, choose each opponent bot (GNN bots let you
  select any `.pt` from `--checkpoints-dir`), set rules (VP target 10/5,
  bonuses on/off, optional seed), Start. On your turn, legal build spots glow
  on the board (click to build) and non-spatial actions (roll, buy dev card,
  bank/propose trade, play dev cards, end turn) appear as buttons. When a bot
  offers you a trade, a modal asks Accept / Reject.
- **Replay tab:** lists `playback_seed_*/index.html` dirs under
  `--replays-dir`. Generate one with
  `python -m catan_mcts.playback <run_dir> <seed>` into that dir.

Both `--checkpoints-dir` and `--replays-dir` default to `.` if omitted (so the
server starts with no GNN bots / no replays, which is fine for a Random/Greedy
game). `--host` defaults to `127.0.0.1` (local only); `--port` to 8000.

## How it works

```
catan_mcts/web/
├── server.py        FastAPI app (REST + SSE). create_app(checkpoints_dir, replays_dir);
│                    paths are CLI args — no hardcoded paths, so deploy is config-only.
├── game_session.py  GameSession: one live game (CatanState + bots). Drives chance +
│                    bot turns; intercepts a bot ProposeTrade that would auto-match you.
├── bot_registry.py  Bot type listing + .pt checkpoint discovery + construction (lazy torch).
├── trade_logic.py   Engine-faithful prediction of who would accept a ProposeTrade.
├── action_decode.py Raw action ints -> {id, label, kind, target} for the clickable board.
├── board_layout.py  Board geometry + static board PNG (shared with playback.py).
├── serializers.py   Engine state -> client JSON (shared with playback.py).
└── static/          index.html (Play/Replay tabs), play.js, replay.js, style.css.
```

### The trade intercept (the key design point)

The Rust engine resolves a `ProposeTrade` **instantly** — it scans opponents in
seat order and the first one holding the requested resource auto-accepts a
1-for-1 swap. There is no pending-trade phase, so the engine can't pause for a
human to accept or reject. The human trade-response therefore lives in the
**Python orchestration layer**, not the engine:

- `GameSession.advance()` drives chance + bot turns. Before applying a bot's
  `ProposeTrade`, it predicts (via `trade_logic`) whether the engine would
  auto-match the human. If so, it pauses and returns a `trade_offer`.
- **Accept** applies the trade normally (you're the first match, so you get the
  swap). **Reject** never applies it — instead it re-queries the bot with that
  trade masked out (`_MaskedLegalView`, whose `clone()` preserves the mask so
  tree-search bots see it too) and falls back to `EndTurn`.

See `docs/superpowers/specs/2026-05-31-interactive-play-vs-bots-design.md` §2.

### Async bot driving + SSE

Bot turns run in a background daemon thread (`advance_async`, guarded by a
per-session `RLock`), so a slow GNN/MCTS bot doesn't block the HTTP request.
After your action the server returns immediately (status `bot_thinking` if bots
are still driving); the frontend opens the `/events` SSE stream and applies the
settled state when bots finish. The stream is status-transition-based and
capped (~60s); the client re-syncs and re-opens if a very slow turn exceeds the
cap.

## Bots

`Random`, `Greedy` (instant), `LookaheadMctsV3` (strong, slower), `PureGnn`
(needs a `.pt`, fast), `GnnMcts` (needs a `.pt`, slow). **Note:** `GnnMcts`
uses the older `GnnEvaluator` whose value head is known to be miscalibrated
(see project history) — it tends to play *weaker* than `PureGnn`, so prefer
`PureGnn` for a strong GNN opponent.

## Deploy note

Local-first but deploy-ready: clean REST/SSE API, configurable paths, one game
per server-side session. Out of scope this iteration: multi-user, auth,
concurrent-game scaling, and session TTL/cleanup (the in-memory `games` dict
grows for the process lifetime — fine for local single-user; add eviction +
`DELETE /api/games/{id}` before any shared deployment).

## Tests

    pytest tests/test_board_layout.py tests/test_serializers.py \
           tests/test_serializers_golden.py tests/test_playback.py \
           tests/test_bot_registry.py tests/test_action_decode.py \
           tests/test_trade_logic.py tests/test_game_session.py \
           tests/test_web_api.py

The frontend JS is verified by a headless-Chromium smoke (board + overlay +
player panel + trade modal render; a full game plays to terminal through the
same endpoints `play.js` uses) and `node --check`. The Python API tests are the
contract gate.
