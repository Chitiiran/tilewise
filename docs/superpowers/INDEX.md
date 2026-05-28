# Project documentation index (main branch)

Index of journals, plans, and specs on the `main` branch. The main branch is at 2026-05-02 (v3 design spec); active loss-augmentation work is on the `v3` branch — see [the v3 worktree's INDEX.md](../../.claude/worktrees/v3/docs/superpowers/INDEX.md) for that work.

## Phase 0 — v1 engine + MCTS study (Apr 2026)

The foundational layer: Rust engine + Python adapter + four-experiment MCTS study + GNN-ready self-play dataset.

**Specs**
- [`catan-engine-design.md`](specs/2026-04-27-catan-engine-design.md) — v1 engine architecture
- [`mcts-study-design.md`](specs/2026-04-27-mcts-study-design.md) — four-experiment study design

**Plans**
- [`catan-engine-v1.md`](plans/2026-04-27-catan-engine-v1.md)
- [`mcts-adapter-recorder-bots.md`](plans/2026-04-27-mcts-adapter-recorder-bots.md)
- [`mcts-experiments-and-writeup.md`](plans/2026-04-27-mcts-experiments-and-writeup.md)
- [`mcts-phase0-engine-chance-api.md`](plans/2026-04-27-mcts-phase0-engine-chance-api.md)

**Journals** — no formal journal at this stage; see `mcts_study/docs/learnings.md` and `mcts_study/docs/writeup.md`.

## Phase 1 — GNN v0 + v2 engine (late Apr → early May 2026)

**Specs**
- [`gnn-evaluator-design.md`](specs/2026-04-29-gnn-evaluator-design.md)
- [`engine-v2-wishlist.md`](specs/2026-04-30-engine-v2-wishlist.md)
- [`v2-restart-full-game-design.md`](specs/2026-04-30-v2-restart-full-game-design.md)
- [`phase4-strategy.md`](specs/2026-05-01-phase4-strategy.md)
- [`playback-ui-polish-design.md`](specs/2026-05-01-playback-ui-polish-design.md)

**Plans**
- [`gnn-v0-implementation.md`](plans/2026-04-29-gnn-v0-implementation.md)
- [`playback-ui-polish.md`](plans/2026-05-01-playback-ui-polish.md)

**Journals**
- [`gnn-v0-execution-journal.md`](journals/2026-04-29-gnn-v0-execution-journal.md)
- [`engine-v2-phase1-journal.md`](journals/2026-04-30-engine-v2-phase1-journal.md)
- [`engine-v2-phase2-journal.md`](journals/2026-05-01-engine-v2-phase2-journal.md)
- [`engine-v2-phase3-journal.md`](journals/2026-05-01-engine-v2-phase3-journal.md)

## Phase 2 — v3 design (Catan-Lite, 2026-05-02)

- [`v3-design.md`](specs/2026-05-02-v3-design.md) — the 5-VP, no-bonus simplified target

(All journals/plans/specs after 2026-05-02 are on the `v3` branch.)

## What's on the `v3` branch

The Phase 3 + Phase 4 work — loss-augmentation candidates (Cand 1, 2, 7, 8, 10, 11), Cand 11 development, Cell 5 v2 head-to-head, Cell 6 full-Catan inversion, and the 4-quadrant cumulative-best matrix — lives there with full journal history.

```
git fetch origin v3
git worktree add .claude/worktrees/v3 v3
```

Then see [`.claude/worktrees/v3/docs/superpowers/INDEX.md`](../../.claude/worktrees/v3/docs/superpowers/INDEX.md) and [`.claude/worktrees/v3/docs/superpowers/README.md`](../../.claude/worktrees/v3/docs/superpowers/README.md).

## Reference

- [`reference/geometry-tables.md`](reference/geometry-tables.md) — static board topology (HEX_TO_VERTICES, EDGE_TO_VERTICES)
- [`../../mcts_study/analyses/board_topology_derivation.py`](../../mcts_study/analyses/board_topology_derivation.py) — script that derives the geometry tables from cube coordinates
