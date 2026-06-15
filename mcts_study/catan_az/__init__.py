"""catan_az — AlphaZero iteration loop for the Catan GNN.

Spec: docs/superpowers/specs/2026-06-11-az-loop-design.md
Thin orchestration over proven pieces: self_play_async (self-play),
catan_gnn.train.train_main (training), AsyncMcts + BatchedGnnEvaluator
(arena). New code here is glue + bookkeeping, unit-testable without a GPU.
"""
