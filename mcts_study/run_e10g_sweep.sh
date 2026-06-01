#!/usr/bin/env bash
set -e
PY=/home/chitii/catan_mcts_venvs/mcts-study/bin/python
CK=/home/chitii/catan_data/runs/v3/rl_checkpoints/round0_Cell6.pt
cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study
for S in 8 16 32; do
  echo "===== SWEEP sims=$S start ====="
  "$PY" -m catan_mcts.experiments.e10g_cheapsearch_async \
    --checkpoint "$CK" --sims "$S" \
    --num-games-per-seating 30 --device cpu --n-concurrent 12 \
    --seed-base 20000000 \
    --out-root "/home/chitii/catan_data/runs/v3/e10g_sweep_sims$S"
  echo "===== SWEEP sims=$S done ====="
done
echo "ALL SWEEPS DONE"
