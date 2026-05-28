# Main-repo analyses

Phase-0 and Phase-1 analysis scripts. The v3-branch worktree at `.claude/worktrees/v3/mcts_study/analyses/` has the Phase-3/4 loss-augmentation analyses; this directory holds the older foundational ones.

## Scripts

| Script | Purpose |
|---|---|
| `board_topology_derivation.py` | Derives the static `HEX_TO_VERTICES` and `EDGE_TO_VERTICES` tables for the standard 19-hex Catan board using cube-coordinate hex geometry. Output cross-checked against `catan_engine/src/board.rs`. |
| `e5_lookahead_winrate_aggregate.py` | Aggregates the e5 (lookahead-depth sweep) experiment results across worker shards. Reports winrate per `(depth, sims)` cell. |
| `e5_status.sh` | Shell wrapper showing per-worker progress on the e5 run. Reads `done.txt` / `skipped.csv` from each worker dir. |

## Running

These scripts hardcode `/mnt/c/dojo/catan_bot/...` paths. The e5_* scripts reference an old MCTS-study experiment directory that may no longer exist; treat them as historical reference. The board topology derivation has no path dependencies and can be re-run any time.
