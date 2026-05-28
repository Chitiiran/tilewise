# Project documentation index

Living index of all journals, plans, and specs in `docs/superpowers/`. Organized by project phase. The flat dirs (`journals/`, `plans/`, `specs/`) remain chronological — this index gives a topical view across them.

For quick navigation in the dojo IDE: click any link below.

---

## Phase 0 — v1 engine + MCTS study (Apr 2026)

The foundational layer: Rust engine + Python adapter + four-experiment MCTS study + GNN-ready self-play dataset.

**Specs**
- [`catan-engine-design.md`](specs/2026-04-27-catan-engine-design.md) — v1 engine architecture
- [`mcts-study-design.md`](specs/2026-04-27-mcts-study-design.md) — four-experiment study design

**Plans**
- [`catan-engine-v1.md`](plans/2026-04-27-catan-engine-v1.md) — engine build plan
- [`mcts-adapter-recorder-bots.md`](plans/2026-04-27-mcts-adapter-recorder-bots.md) — OpenSpiel adapter
- [`mcts-experiments-and-writeup.md`](plans/2026-04-27-mcts-experiments-and-writeup.md) — four experiments
- [`mcts-phase0-engine-chance-api.md`](plans/2026-04-27-mcts-phase0-engine-chance-api.md) — chance-node API

**Journals**
- (no formal Phase-0 journal; see `mcts_study/docs/learnings.md` and `mcts_study/docs/writeup.md`)

---

## Phase 1 — GNN v0 + v2 engine + Phase 4 strategy (late Apr → early May 2026)

Trained the first GNN policy/value network on MCTS-vs-random data; rebuilt the engine as v2 (full Catan rules); designed the Phase 4 roadmap after Phase 3 yielded an honest negative result.

**Specs**
- [`gnn-evaluator-design.md`](specs/2026-04-29-gnn-evaluator-design.md) — first GNN architecture
- [`engine-v2-wishlist.md`](specs/2026-04-30-engine-v2-wishlist.md) — what the engine v2 should do
- [`v2-restart-full-game-design.md`](specs/2026-04-30-v2-restart-full-game-design.md) — v2 full-rules design
- [`phase4-strategy.md`](specs/2026-05-01-phase4-strategy.md) — strategy after Phase 3 negative result
- [`playback-ui-polish-design.md`](specs/2026-05-01-playback-ui-polish-design.md) — playback viewer design

**Plans**
- [`gnn-v0-implementation.md`](plans/2026-04-29-gnn-v0-implementation.md) — GNN v0 plan
- [`playback-ui-polish.md`](plans/2026-05-01-playback-ui-polish.md) — UI polish plan

**Journals**
- [`gnn-v0-execution-journal.md`](journals/2026-04-29-gnn-v0-execution-journal.md)
- [`engine-v2-phase1-journal.md`](journals/2026-04-30-engine-v2-phase1-journal.md)
- [`engine-v2-phase2-journal.md`](journals/2026-05-01-engine-v2-phase2-journal.md)
- [`engine-v2-phase3-journal.md`](journals/2026-05-01-engine-v2-phase3-journal.md)

---

## Phase 2 — v3 design (Catan-Lite) + production data run (early May 2026)

Decided to simplify the win condition (5 VP, no bonuses) to train the GNN faster, then port back. Generated the 100k-game self-play corpus.

**Spec**
- [`v3-design.md`](specs/2026-05-02-v3-design.md) — Catan-Lite training target

**Journals**
- [`v3-phase1-journal.md`](journals/2026-05-02-v3-phase1-journal.md) — v3 launch
- [`v3-validation-journal.md`](journals/2026-05-02-v3-validation-journal.md) — 1k validation sweep
- [`v3-production-journal.md`](journals/2026-05-02-v3-production-journal.md) — 100k production run
- [`crash-and-cleanup-journal.md`](journals/2026-05-09-crash-and-cleanup-journal.md) — WSL OOM + recovery
- [`pass100k-tournament-results.md`](journals/2026-05-10-pass100k-tournament-results.md) — first 9-cell tournament
- [`phase0-trade-value-summary.md`](journals/2026-05-10-phase0-trade-value-summary.md) — trade-value analysis

---

## Phase 3 — Loss augmentation roadmap (mid May 2026)

After the 100k corpus didn't produce a competitive GNN, we tried adding auxiliary loss terms (Cand 1, 2, 7, 8, 10) on top of the supervised CE. Mostly negative results — but established the methodology and the val_top1-misleading rule.

**Spec**
- [`loss-augmentation-design.md`](specs/2026-05-09-loss-augmentation-design.md) — the candidate roadmap

**Journals**
- [`cell0-cell1-baseline-vs-cand8_10.md`](journals/2026-05-12-cell0-cell1-baseline-vs-cand8_10.md) — Cand 8 + Cand 10 stack
- [`cell2-cand7-stacked-regression.md`](journals/2026-05-25-cell2-cand7-stacked-regression.md) — Cand 7 rolled back

---

## Phase 4 — Cand 11 (road-pip prior) + cumulative-best discovery (late May 2026)

The breakthrough phase. Designed and implemented Cand 11 (pure-pip road prior); discovered Cand 11 alone wins v3 head-to-head; then discovered the full-Catan ranking inversion where the Cand 11 + Cand 8 + Cand 10 stack (Cell 6) dominates.

**Plan**
- [`road-pip-prior.md`](plans/2026-05-25-road-pip-prior.md) — Cand 11 implementation plan

**Spec**
- [`cand11-vectorization.md`](specs/2026-05-25-cand11-vectorization.md) — vectorization for GPU perf

**Journals (in arc order)**
- [`cell5-road-pip-prior.md`](journals/2026-05-25-cell5-road-pip-prior.md) — Cell 5 training (Cand 11 alone)
- [`cand11-perf-rca.md`](journals/2026-05-25-cand11-perf-rca.md) — v1 perf regression RCA + vectorization
- [`cand11-headtohead-tournament.md`](journals/2026-05-26-cand11-headtohead-tournament.md) — Cell 5 v2 = v3 cumulative best (16.83%)
- [`cell6-cand11-cand8-cand10-stack.md`](journals/2026-05-26-cell6-cand11-cand8-cand10-stack.md) — Cell 6 training (stack)
- [`4puregnn-no-lookahead-tournament.md`](journals/2026-05-27-4puregnn-no-lookahead-tournament.md) — v3 4-PureGnn
- [`full-catan-tournament-inversion.md`](journals/2026-05-27-full-catan-tournament-inversion.md) — full-Catan ranking inverts; Cell 6 wins
- [`fullcatan-deep-behavioral-analysis.md`](journals/2026-05-27-fullcatan-deep-behavioral-analysis.md) — mechanism: LR + LA bonus stacking
- [`fullcatan-with-lookv3-tournament.md`](journals/2026-05-28-fullcatan-with-lookv3-tournament.md) — final corner of the rule-conditional matrix

---

## Cross-reference — cumulative-best decisions over time

| Date | Claim | Superseded by |
|---|---|---|
| 2026-05-12 | Cell 1 (Cand 8 + Cand 10) is cumulative best (mid-tournament) | 2026-05-26 |
| 2026-05-25 | Cand 7 rejected (regression) | — |
| 2026-05-26 | **Cell 5 v2 (Cand 11 alone)** is cumulative best for v3 rules (head-to-head 16.83%) | — |
| 2026-05-27 | **Cell 6 (stack)** is cumulative best for full Catan (4-PureGnn 54.33%) | — |
| 2026-05-28 | Cell 6 confirmed cumulative best for full Catan + LookV3 (19.00%, 1.75× over Cell 1) | — |

Both Phase-4 cumulative-best claims still hold, but **scoped to a rule set**.

## Cross-reference — the four 1200-game tournaments

| Date | Tournament | Rule set | Lineup | Winner | Plot |
|---|---|---|---|---|---|
| 2026-05-26 | e10c head-to-head | v3 (vp=5, bonuses=off) | Cell0 + Cell1 + Cell5v2 + LookV3 | LookV3 67%, Cell 5 v2 16.83% | (cited journal) |
| 2026-05-27 | e10d 4-PureGnn (v3) | v3 | Cell0/1/5v2/6 (no LookV3) | Cell 5 v2 30.92% | (cited journal) |
| 2026-05-27 | e10d 4-PureGnn (full Catan) | full Catan (vp=10, bonuses=on) | Cell0/1/5v2/6 (no LookV3) | Cell 6 54.33% | `figures/winrate_by_rules.png` |
| 2026-05-28 | e10c full-Catan + LookV3 | full Catan | Cell6 + Cell5v2 + Cell1 + LookV3 | LookV3 70%, Cell 6 19% | `figures/rules_opponents_matrix.png` |

---

## How to use this index

- **New to the project**: read Phase 0 → 4 in order
- **Looking for a specific topic**: ctrl-F by keyword (cand11, loss-aug, full-catan, etc.)
- **Looking for a tournament result**: see Cross-reference table above
- **Looking for the current cumulative best**: Phase 4, last item

This index is regenerated by hand. Update when new journals/plans/specs land.
