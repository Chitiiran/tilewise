# mcts_study

A scientific study of MCTS on Catan, using the v1 engine + OpenSpiel.

The project is the second deliverable on top of the catan_bot engine. The first was the engine itself (`catan_engine/` + `python/catan_bot/`). The next planned project is a GNN policy/value network trained on the self-play parquet this study generates.

## Why this exists

Goal: **learn how to use MCTS as a problem-solving tool**, not produce a maximum-strength bot. We treat the algorithm as a tool, the game as the test bench, and the deliverables are *understanding* — see [`docs/writeup.md`](docs/writeup.md) and [`docs/learnings.md`](docs/learnings.md).

## Quick start (Linux / WSL)

OpenSpiel does not install on Windows — its build pipeline is Unix-only. Use WSL Ubuntu (or any Linux distro) with Python 3.10+.

```bash
# 1. From a WSL shell, create a Linux-fs venv (don't put .venv on /mnt/c — too slow):
python3 -m venv ~/catan_mcts_venvs/mcts-study
source ~/catan_mcts_venvs/mcts-study/bin/activate
pip install --upgrade pip maturin

# 2. Build the catan_bot engine for Linux into the venv:
cd /path/to/catan_bot
maturin develop --release --manifest-path catan_engine/Cargo.toml

# 3. Install this project's deps:
cd mcts_study
pip install -e ".[dev]"

# 4. Run tests:
pytest                     # excludes slow tests
pytest -m slow             # opt-in to long-running e4 smoke

# 5. Run an experiment (small):
python -m catan_mcts run e1 --num-games 3 --sims-grid 5 25

# 6. Run the analysis notebook to plot results:
jupyter notebook notebooks/analysis.ipynb
```

## Layout

```
mcts_study/
├── catan_mcts/                  # the project package
│   ├── __init__.py              # exposes ACTION_SPACE_SIZE = 206
│   ├── adapter.py               # CatanGame / CatanState — registered pyspiel.Game subclass
│   ├── bots.py                  # GreedyBaselineBot + heuristic_rollout
│   ├── recorder.py              # SelfPlayRecorder writes parquet datasets
│   ├── cli.py                   # `python -m catan_mcts run <name>` dispatcher
│   ├── __main__.py              # module entrypoint
│   └── experiments/
│       ├── common.py            # play_one_game canonical driver + GameOutcome
│       ├── e1_winrate_vs_random.py
│       ├── e2_ucb_c_sweep.py
│       ├── e3_rollout_policy.py
│       └── e4_tournament.py
├── notebooks/
│   └── analysis.ipynb           # read-only over runs/, produces all plots
├── docs/
│   ├── writeup.md               # ~5-page study writeup
│   └── learnings.md             # 5 reusable MCTS learnings (+ 3 process notes)
├── tests/                       # pytest; 26+ tests at last count
└── runs/                        # gitignored; produced by `python -m catan_mcts run ...`
```

## Outputs

Each `python -m catan_mcts run <name>` invocation creates `runs/<utc-timestamp>-<name>/`:

| File | Schema |
|---|---|
| `moves.parquet` | One row per **MCTS-decided move**. Columns: `seed`, `move_index`, `current_player`, `legal_action_mask: list<bool>[206]`, `mcts_visit_counts: list<int32>[206]`, `action_taken`, `mcts_root_value`, `schema_version`. |
| `games.parquet` | One row per game. Columns: `seed`, `winner`, `final_vp: list<int32>[4]`, `length_in_moves`, `mcts_config_id`, `schema_version`. |
| `config.json` | Experiment parameters + `mcts_config_id` (UUID linking back to games.parquet). |

`SCHEMA_VERSION = 1`. Bumps require explicit `recorder.py` change + test update.

## Companion to the future GNN project

`runs/*/moves.parquet` is the AlphaZero-style training dataset — `(state, MCTS visit distribution, eventual outcome)` triples. Observation tensors are **not** stored; the GNN project's data loader replays each `(seed, action_history_up_to_move_index)` through the engine to materialize observations on demand. The engine is deterministic (catan_engine spec v1 §9, regression-tested).

This means this study's outputs are tiny (~3 MB per 1k games) but reproducible only as long as the engine's chance-node contract stays stable (catan_engine spec §3.7).

## Caveats

- **Compute scale is below the spec's intent.** The original spec called for sims grids up to 4000 and 200 games per cell. With Phase-0's honest chance points, real Tier-1 games are ~12k-15k steps long, which makes those defaults workstation-week workloads. This study runs at proof-of-pipeline scale (3-15 games per cell, sims ≤ 100). See `docs/writeup.md` §1 (Setup) for full numbers.
- **Random opponents are weak.** The numbers from this study should not be generalized to "MCTS at sims=N wins X% of Catan games." They're "MCTS at sims=N wins X% of Catan games against a uniform-random policy on a fixed Tier-1 board." Treat as qualitative.

## License

Same as the parent repo.
