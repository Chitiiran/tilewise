# tilewise

A lightweight Rust + Python Catan game engine designed for neural-network self-play training.

## What this is

`tilewise` is a headless, deterministic Catan engine built for one job: running millions of games per second so a neural network can learn to play through self-play (AlphaZero-style training).

The engine core is written in Rust and exposed to Python via PyO3 / maturin. The Python wrapper provides a `gym`-style RL environment.

## Status

**Pre-implementation.** Design spec is complete; implementation has not started.

See [`docs/superpowers/specs/2026-04-27-catan-engine-design.md`](docs/superpowers/specs/2026-04-27-catan-engine-design.md) for the full v1 design.

## Roadmap

- **v1 — engine** (current focus): Tier 1 rules (base game minus trades and dev cards), fixed board, 4 players, ~10k games/sec target.
- **Tier 2:** randomized board, bank/port trades, development cards, Largest Army, Longest Road.
- **Tier 3:** player-to-player trading.
- **Future projects** (separate repos / dirs): GNN policy network, MCTS self-play loop, replay viewer.

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

## License

TBD.
