#!/bin/bash
# v2 production sweep — runs e1/e2/e3/e4 sequentially with the v2 hardening:
#   - per-game wall-clock cap (default 300s) keeps any single game from
#     stalling the sweep; timeouts go to skipped.csv.
#   - per-cell parquet shards (e.g. moves.sims=5.parquet) land incrementally,
#     so partial results survive even if a later cell is killed.
#   - done.txt resumability: kill+restart picks up where left off (lose at
#     most one in-flight game per experiment).
#   - lower engine rollout cap (30k vs old 100k) bounds individual rollout cost.
#
# Run from inside the WSL venv at /mnt/c/dojo/catan_bot:
#   source ~/catan_mcts_venvs/mcts-study/bin/activate
#   bash mcts_study/run_v2_production.sh
#
# Sized for ~3-4 hours single-process wall-clock (vs. the v1 sweep that took
# 8+ hours and stalled on game 18). For ~4x more speed, see the optional
# multiprocessing flags noted in the script.

set -e
LOG=mcts_study/runs/v2_production_log.txt
mkdir -p mcts_study/runs
echo "=== v2 production run started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

t_start=$(date +%s)

# e1: sims grid [5, 25, 100] x 15 games = 45 games. Each cell flushes a shard
# on completion. Per-game cap 300s means any pathological game gets bumped.
echo "--- e1 starting $(date -u +%H:%M:%S)" | tee -a "$LOG"
python -m catan_mcts run e1 \
    --num-games 15 --sims-grid 5 25 100 \
    --max-seconds 300 \
    --out-root mcts_study/runs 2>&1 | tee -a "$LOG"

# e2: c grid [0.5, 1.0, 1.4, 2.0, 4.0] at sims=25, 8 games = 40 games.
echo "--- e2 starting $(date -u +%H:%M:%S)" | tee -a "$LOG"
python -m catan_mcts run e2 \
    --num-games 8 --c-grid 0.5 1.0 1.4 2.0 4.0 --sims 25 \
    --max-seconds 300 \
    --out-root mcts_study/runs 2>&1 | tee -a "$LOG"

# e3: random vs heuristic at sims grid [5, 25], 8 games = 32 games.
echo "--- e3 starting $(date -u +%H:%M:%S)" | tee -a "$LOG"
python -m catan_mcts run e3 \
    --num-games 8 --sims-grid 5 25 \
    --max-seconds 300 \
    --out-root mcts_study/runs 2>&1 | tee -a "$LOG"

# e4: mcts_sims=25, 4 games per seating x 4 rotations = 16 games.
echo "--- e4 starting $(date -u +%H:%M:%S)" | tee -a "$LOG"
python -m catan_mcts run e4 \
    --num-games-per-seating 4 --mcts-sims 25 \
    --max-seconds 300 \
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
    shards=$(ls "$d"/games.*.parquet 2>/dev/null | wc -l)
    done_count=$(wc -l < "$d/done.txt" 2>/dev/null || echo 0)
    skipped_count=$(($(wc -l < "$d/skipped.csv" 2>/dev/null || echo 1) - 1))  # minus header
    [ "$skipped_count" -lt 0 ] && skipped_count=0
    echo "  $name: $shards shards, $done_count games done, $skipped_count skipped" | tee -a "$LOG"
done
