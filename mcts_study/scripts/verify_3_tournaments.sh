#!/bin/bash
# Reproducibility check: run 3 tournaments back-to-back for the same
# seed_base=19M:
#   1. ep5 of current pilot (re-test, to confirm 15/120 was not a
#      one-shot artifact)
#   2. ep10 of current pilot (re-test, to confirm 2/120)
#   3. grid_full20 checkpoint_best.pt (= best by val_top1 of pass-3
#      training, val_top1=0.183 at epoch 19; cited 4/120 in pass3)
#
# Each tournament: workers=10 cuda, num_games_per_seating=30 (120 total),
# sims=100, lookahead_depth=10, base_sims_v3=200. Standard Phase 1
# config per STANDARD_* constants in mid_training_tournament.py.

set -uo pipefail
cd "$(dirname "$0")/.."
source ~/catan_mcts_venvs/mcts-study/bin/activate

ROOT="runs/v3/loss_aug/00_baseline_h128_l4_pilot"

declare -A CHECKPOINTS
CHECKPOINTS["ep5_v2"]="${ROOT}/training_h128_l4/checkpoint_epoch05.pt"
CHECKPOINTS["ep10_v2"]="${ROOT}/training_h128_l4/checkpoint_epoch10.pt"
CHECKPOINTS["pass3_best"]="runs/v3/grid_full20/training_h128_l4/checkpoint_best.pt"

# Run order matters: smallest first (ep5 verifies the surprising number first).
RUN_ORDER=("ep5_v2" "ep10_v2" "pass3_best")

for label in "${RUN_ORDER[@]}"; do
  ckpt="${CHECKPOINTS[$label]}"
  out="${ROOT}/verify_${label}_seed19M"
  echo "============================================================"
  echo "[verify] $(date): running tournament for $label"
  echo "[verify] checkpoint: $ckpt"
  echo "[verify] out: $out"
  echo "============================================================"
  mkdir -p "$out"
  python -m catan_mcts.experiments.e10_v3_tournament \
    --checkpoint "$ckpt" \
    --hidden-dim 128 --num-layers 4 \
    --num-games-per-seating 30 \
    --sims 100 --lookahead-depth 10 --base-sims-v3 200 \
    --seed-base 19000000 \
    --workers 10 --device cuda \
    --out-root "$out"
  echo "[verify] $(date): $label tournament finished"
  # Brief settle before next
  sleep 15
done

echo "[verify] $(date): ALL 3 TOURNAMENTS DONE"
