# Interactive Play-vs-Bots — Progress Journal

**Branch:** `worktree-interactive-play` (worktree at `.claude/worktrees/interactive-play/`)
**As of:** 2026-06-01 — 47 commits ahead of `main`, working tree clean
(only untracked `_*.png` browser-verification screenshots remain, intentionally
uncommitted).

## What this is

A local FastAPI web app to play Catan as one seat against selectable bots, with
human-in-the-loop trade responses. Spec:
`docs/superpowers/specs/2026-05-31-interactive-play-vs-bots-design.md`. Plan:
`docs/superpowers/plans/2026-05-31-interactive-play-vs-bots.md`. Architecture
README: `mcts_study/catan_mcts/web/README.md`.

## Run it

WSL, mcts-study venv, from `mcts_study/`:

    python -m catan_mcts.web \
        --checkpoints-dir /home/chitii/catan_data/runs/v3 \
        --replays-dir     /home/chitii/catan_data/runs/v3/playback \
        --host 0.0.0.0 --port 8000

Open http://localhost:8000. The web server is a **standalone process** (not the
training venv's GPU work) and all bots run **CPU-only** — safe to run alongside
CUDA training.

Test command (any web work):

    wsl -e bash -lc 'source ~/catan_mcts_venvs/mcts-study/bin/activate && \
      cd /mnt/c/dojo/catan_bot/.claude/worktrees/interactive-play/mcts_study && \
      python -m pytest tests/test_*.py'

Git runs **Windows-side** (`git -C "C:\...\interactive-play"`) — the worktree
`.git` points at a Windows path so git fails inside WSL.
`node --check` validates the frontend JS (Node on Windows PATH).

## DONE (built, reviewed, tested, shipping)

Core (Phases 0–7 of the plan, all spec+quality reviewed, 48+ tests):
- Shared-module extraction from `playback.py` (`board_layout.py`, `serializers.py`),
  golden-test-guarded.
- `bot_registry` (type listing, `.pt` discovery, lazy-torch GNN construction;
  infers arch from checkpoint so any trained `.pt` loads).
- `action_decode`, `trade_logic`, `GameSession` (driving loop + Python-layer
  trade intercept + RLock + async advance + engine-fault capture).
- FastAPI server (REST + SSE, async non-blocking bot turns, error mapping,
  static serving, `python -m catan_mcts.web` launcher, no-store cache headers).
- Vanilla-JS frontend: lobby + game screen + replay tab.

UX iterations (this session, all frontend unless noted):
- No-scroll 2-column layout (board left; right = players → log → actions).
- Distinct settlement (house) vs city (house+tower) glyphs; last-move cyan glow.
- Win confetti + banner; named bots (Randy/Neura/etc., distinct personas).
- Checkpoint dropdown grouped by run dir via `<optgroup>` (417 `.pt`,
  dir-qualified labels — fixed "all named checkpoint_best.pt").
- Visible **bank strip** (lobby toggle); **clickable dev-card hand** (Knight /
  Road Building direct; Monopoly + Year of Plenty via resource icon pickers);
  **trade as grids only** (Bank Trade + Propose Trade 5×5 give→get, no buttons).
- **Obvious robber/discard prompt** (pulsing red banner + large robber-marked
  hexes) — was reading as "hung".
- **Year of Plenty** fixed: same-kind picks (Wheat+Wheat) now work; rewrote the
  picker stateless (DOM-attribute, no stale global) — no more hang. Regression
  test `test_same_kind_year_of_plenty_applies_cleanly`.
- **Longest Road / Largest Army badges** in the player panel (the +2 VP was
  always counted in `vp` but invisible).
- **Default opponents = PureGnn + Cell 6** (strongest fast bot for full Catan,
  ~54%, CPU-only). Checkpoint:
  `runs/v3/training/loss_aug/06_cand11_cand8_cand10_h128_l4/training_h128_l4/checkpoint_epoch10.pt`.
- **Per-player stat cards** (colonist-style): pieces remaining
  (settle/city/road = max−built), dev-card count, knights, longest-road length,
  opponent hand sizes, ports owned.

## KEY ARCHITECTURE NOTES

- **Engine auto-resolves trades** → human trade-response is a Python intercept
  in `GameSession.advance()` (pause on a bot ProposeTrade that would auto-match
  the human; reject re-queries the bot with the trade masked). Memory:
  `project-engine-trades-auto-resolve`.
- **Bots run CPU-only** (`bot_registry.build` defaults `device="cpu"`). GnnMcts
  is ~8× weaker than PureGnn (value-head bug) — avoid it. Memory:
  `project-e10e-gnnmcts-worse-than-puregnn`.
- **GNN arch inferred from checkpoint** (hidden_dim from `proj_hex.weight`,
  num_layers from max `body.convs.N` index) so h32/h64/h128 + 2-4 layers all
  load.

## PENDING — needs an engine rebuild (DEFERRED)

Two items need Rust changes + `maturin develop --release`, which is **held until
the user's `e10e_async` tournament finishes** (it uses the shared mcts-study
venv; rebuilding mid-run would corrupt it). Source edits are worktree-isolated;
only the compiled `_engine` leaks into the shared venv, hence the hold.

1. **Robber steal-victim choice** — fully specced + approved:
   `docs/superpowers/specs/2026-06-01-robber-steal-victim-choice-design.md`.
   Summary: the engine's Steal phase is a chance node that hardcodes
   `from_options[0]` and HARD-ASSERTS that victim (`engine.rs:213`), so a
   Python intercept can't override it (unlike trades). Fix keeps
   ACTION_SPACE_SIZE=280 (no bot retraining): default the Steal chance
   distribution to the **most-cards** victim (bots steal greedily), **relax the
   assert** to accept any eligible victim, add a `steal_options()` PyO3
   accessor. Then a Python `respond_to_steal` intercept + a UI victim modal with
   a **5s auto-pick-most-cards timer**. Tests-first (Rust + Python + API +
   frontend). Memory: `project-engine-trades-auto-resolve` (steal note added).

2. **Dev-deck-remaining counter** (colonist shows "25") — the engine tracks
   `dev_card_deck_remaining` (`state.rs:192`) but does NOT expose it via PyO3.
   Needs a small accessor + the dropdown render. NOT yet specced. Candidate to
   batch into the same engine rebuild as #1. (Open question the user left: add
   it to the engine batch or not.)

## NON-GOALS / answered questions

- **2-player Catan**: not possible without a 4→2 engine rewrite + bot
  retraining (4 players hardwired in the Rust core). Scoped, not built.
- **Turn timer / chat**: skipped — bots respond instantly; not meaningful vs
  bots.

## Colonist parity (final tally)

Have: bank, pieces-remaining, per-player stats (dev/knights/LR/hand-size),
ports, LR/LA holders, last-move glow (extra). Missing: dev-deck-remaining
(needs engine, see PENDING #2). Skipped by design: turn timer, chat.

## Resume checklist

1. Verify the tournament finished: `pgrep -f e10e_async` returns nothing.
2. (Optional) re-confirm no other sweep: `ps -ef | grep "catan_mcts run"`.
3. Invoke `writing-plans` on the steal-victim spec (and optionally add
   dev-deck-remaining), implement engine-first, then `maturin develop --release`
   from `catan_engine`, run Rust + Python tests.
4. Restart the web server to pick up the new `_engine` + any server changes.
5. Eventually: finish the branch (push + PR, or merge to main — merges need
   user approval per `catan_bot-git-policy`).
