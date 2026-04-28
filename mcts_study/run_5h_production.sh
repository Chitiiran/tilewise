#!/bin/bash
# 5-hour production sweep — runs e1, e2, e3, e4 sequentially. Each experiment
# flushes its parquet on completion, so partial results survive even if a
# later experiment runs long.
#
# Run from inside the WSL venv:
#   source ~/catan_mcts_venvs/mcts-study/bin/activate
#   bash mcts_study/run_5h_production.sh
#
# Outputs: mcts_study/runs/<timestamp>-<experiment>/{moves,games}.parquet + config.json
# Logs: mcts_study/runs/5h_production_log.txt

set -e
LOG=mcts_study/runs/5h_production_log.txt
mkdir -p mcts_study/runs
echo "=== 5h production run started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

t_start=$(date +%s)

# e1: sims grid [5, 25, 100] x 20 games = 60 games. Projected ~90 min.
echo "--- e1 starting $(date -u +%H:%M:%S)" | tee -a "$LOG"
python -m catan_mcts run e1 \
    --num-games 20 --sims-grid 5 25 100 \
    --out-root mcts_study/runs 2>&1 | tee -a "$LOG"

# e2: c grid [0.5, 1.0, 1.4, 2.0, 4.0] at sims=25, 10 games = 50 games. ~100 min.
echo "--- e2 starting $(date -u +%H:%M:%S)" | tee -a "$LOG"
python -m catan_mcts run e2 \
    --num-games 10 --c-grid 0.5 1.0 1.4 2.0 4.0 --sims 25 \
    --out-root mcts_study/runs 2>&1 | tee -a "$LOG"

# e3: random vs heuristic at sims grid [5, 25], 10 games = 40 games. ~50 min.
echo "--- e3 starting $(date -u +%H:%M:%S)" | tee -a "$LOG"
python -m catan_mcts run e3 \
    --num-games 10 --sims-grid 5 25 \
    --out-root mcts_study/runs 2>&1 | tee -a "$LOG"

# e4: mcts_sims=25, 5 games per seating x 4 rotations = 20 games. ~60-90 min.
echo "--- e4 starting $(date -u +%H:%M:%S)" | tee -a "$LOG"
python -m catan_mcts run e4 \
    --num-games-per-seating 5 --mcts-sims 25 \
    --out-root mcts_study/runs 2>&1 | tee -a "$LOG"

t_end=$(date +%s)
elapsed=$((t_end - t_start))
echo "=== 5h production run finished $(date -u +%Y-%m-%dT%H:%M:%SZ) elapsed ${elapsed}s ===" | tee -a "$LOG"
