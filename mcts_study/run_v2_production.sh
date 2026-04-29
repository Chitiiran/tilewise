#!/bin/bash
# v2 production sweep — runs e1/e2/e3/e4 sequentially with the v2 hardening:
#   - per-game wall-clock cap (default 300s) keeps any single game from
#     stalling the sweep; timeouts go to skipped.csv.
#   - per-cell parquet shards (e.g. moves.sims=5.parquet) land incrementally,
#     so partial results survive even if a later cell is killed.
#   - done.txt resumability: kill+restart picks up where left off (lose at
#     most one in-flight game per experiment).
#   - lower engine rollout cap (30k vs old 100k) bounds individual rollout cost.
#   - 4-worker multiprocessing pool — ~4x throughput on 8-thread boxes.
#
# Game counts MATCH the v1 sweep so v1-vs-v2 comparisons are apples-to-apples
# rather than confounded by sample size:
#   e1: 20 games per sims cell (3 cells = 60 games total)
#   e2: 10 games per c value  (5 c-values = 50 games total)
#   e3: 10 games per (policy, sims) cell (4 cells = 40 games total)
#   e4: 5 games per rotation (4 rotations = 20 games total)
#
# Run from inside the WSL venv at /mnt/c/dojo/catan_bot:
#   source ~/catan_mcts_venvs/mcts-study/bin/activate
#   bash mcts_study/run_v2_production.sh
#
# Expected wall-clock ~1-2 hours total with 4 workers (vs. v1's 12 hours).

set -e
LOG=mcts_study/runs/v2_production_log.txt
WORKERS=4
mkdir -p mcts_study/runs
echo "=== v2 production run started $(date -u +%Y-%m-%dT%H:%M:%SZ) (workers=$WORKERS) ===" | tee -a "$LOG"

t_start=$(date +%s)

# e1: sims grid [5, 25, 100] x 20 games = 60 games (matches v1).
echo "--- e1 starting $(date -u +%H:%M:%S)" | tee -a "$LOG"
python -m catan_mcts run e1 \
    --num-games 20 --sims-grid 5 25 100 \
    --max-seconds 300 --workers $WORKERS \
    --out-root mcts_study/runs 2>&1 | tee -a "$LOG"

# e2: c grid [0.5, 1.0, 1.4, 2.0, 4.0] at sims=25, 10 games = 50 games (matches v1).
echo "--- e2 starting $(date -u +%H:%M:%S)" | tee -a "$LOG"
python -m catan_mcts run e2 \
    --num-games 10 --c-grid 0.5 1.0 1.4 2.0 4.0 --sims 25 \
    --max-seconds 300 --workers $WORKERS \
    --out-root mcts_study/runs 2>&1 | tee -a "$LOG"

# e3: random vs heuristic at sims grid [5, 25], 10 games = 40 games (matches v1).
echo "--- e3 starting $(date -u +%H:%M:%S)" | tee -a "$LOG"
python -m catan_mcts run e3 \
    --num-games 10 --sims-grid 5 25 \
    --max-seconds 300 --workers $WORKERS \
    --out-root mcts_study/runs 2>&1 | tee -a "$LOG"

# e4: mcts_sims=25, 5 games per seating x 4 rotations = 20 games (matches v1).
echo "--- e4 starting $(date -u +%H:%M:%S)" | tee -a "$LOG"
python -m catan_mcts run e4 \
    --num-games-per-seating 5 --mcts-sims 25 \
    --max-seconds 300 --workers $WORKERS \
    --out-root mcts_study/runs 2>&1 | tee -a "$LOG"

t_end=$(date +%s)
elapsed=$((t_end - t_start))
echo "=== v2 production run finished $(date -u +%Y-%m-%dT%H:%M:%SZ) elapsed ${elapsed}s ===" | tee -a "$LOG"

# Quick post-run summary: count shards + skipped seeds across all four runs.
echo "" | tee -a "$LOG"
echo "--- Summary across all run dirs ---" | tee -a "$LOG"
for d in mcts_study/runs/*-e[1-4]_*; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    # Count shards across both top-level (serial mode) and worker*/ (parallel mode).
    shards=$(find "$d" -maxdepth 2 -name 'games.*.parquet' 2>/dev/null | wc -l)
    done_count=$(find "$d" -maxdepth 2 -name 'done.txt' -exec cat {} + 2>/dev/null | wc -l)
    skipped_count=$(find "$d" -maxdepth 2 -name 'skipped.csv' -exec tail -n +2 {} + 2>/dev/null | wc -l)
    echo "  $name: $shards shards, $done_count games done, $skipped_count skipped" | tee -a "$LOG"
done
