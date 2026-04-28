# tilewise

A lightweight Rust + Python Catan game engine designed for neural-network self-play training.

## What this is

`tilewise` is a headless, deterministic Catan engine built for one job: running millions of games per second so a neural network can learn to play through self-play (AlphaZero-style training).

The engine core is written in Rust and exposed to Python via PyO3 / maturin. The Python wrapper provides a `gym`-style RL environment.

## Status

- **v1 engine — done.** Tier 1 rules, fixed board, full PyO3 bindings, observations + stats, replay format, determinism + property tests. See [`docs/superpowers/specs/2026-04-27-catan-engine-design.md`](docs/superpowers/specs/2026-04-27-catan-engine-design.md).
- **Phase 0 (chance-node API) — done.** Engine exposes `is_chance_pending`, `chance_outcomes`, `apply_chance_outcome`, `clone`, `action_history`. Action space grew 205 → 206 (`Action::RollDice` is id 205). Required for honest tree search on stochastic games.
- **MCTS study (`mcts_study/`) — done.** OpenSpiel-driven MCTS, four-experiment scientific study, GNN-ready self-play parquet datasets. See [`mcts_study/README.md`](mcts_study/README.md) for the project, [`mcts_study/docs/writeup.md`](mcts_study/docs/writeup.md) for the writeup, and [`mcts_study/docs/learnings.md`](mcts_study/docs/learnings.md) for what we learned about MCTS as a tool.

## Roadmap

- **v1 — engine.** ✅ Tier 1 rules, fixed board, 4 players. Phase-0 chance-node API built on top.
- **Tier 2.** Randomized board, bank/port trades, development cards, Largest Army, Longest Road.
- **Tier 3.** Player-to-player trading.
- **MCTS study.** ✅ First project on top of v1. OpenSpiel `MCTSBot` driven through a `pyspiel.Game` adapter, four experiments (e1: win-rate vs random, e2: UCB c sweep, e3: rollout policy comparison, e4: MCTS-vs-greedy-vs-random tournament), parquet self-play dataset. See [`mcts_study/`](mcts_study/).
- **GNN policy/value network.** Next planned project. Trains on `mcts_study/runs/*/moves.parquet` (visit counts as policy targets, terminal returns as value targets).

## Design highlights

- **Headless + deterministic.** `(seed, action_sequence) → identical trajectory`. Replays are tiny (1-2 KB).
- **Graph-based board.** Hex / vertex / edge as a heterogeneous graph — natively GNN-friendly.
- **Always-on event log.** All stats are folds over a typed event stream; future analyses don't need engine changes.
- **TDD-heavy.** Unit tests per rule + property tests via `proptest` + determinism regression hashes.

## Building (once implementation starts)

Requires Rust (stable) and Python 3.10+.

```sh
pip install maturin
maturin develop          # builds Rust crate, installs Python module
pytest tests/            # Python integration tests
cargo test               # Rust unit + property tests
cargo bench              # throughput benchmark
```

## Performance (v1)

Single-threaded random-vs-random self-play:
- ~500 games/sec
- ~2.0 ms per game (median 2.07 ms fixed-seed, 1.99 ms varied-seed)
- Target was 10,000 games/sec (spec §10) — **not yet hit**; ~20x off.

Measured via `cargo bench` (criterion); CPU: AMD Ryzen 5 5600H. The "random"
policy here is `legal[steps % legal.len()]` — deterministic but varied per
step. Hot-path optimization (avoiding per-call `Vec` allocation in
`legal_actions`, skipping observation tensor construction during pure
self-play, etc.) is deferred — see plan task list for follow-ups.

## License

TBD.
