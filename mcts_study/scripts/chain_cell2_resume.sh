#!/bin/bash
# Watchdog: wait for PID 22938 (Cell 2 ep1+tournament) to exit, then
# resume training from checkpoint_epoch01.pt running epochs 2-10 with
# mid-tournaments at epochs 5 and 10. lambda_settle=0.20 stays on.
#
# Same out-dir (03_cand1_only_h128_l4) so ep1 checkpoint and ep1's
# mid-tournament are preserved; new epoch checkpoints + mid-tournaments
# go into the same dir.

set -uo pipefail
cd "$(dirname "$0")/.."

WAIT_PID="${WAIT_PID:-22938}"

echo "[chain] $(date): waiting for Cell 2 PID $WAIT_PID to exit..."
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done

echo "[chain] $(date): PID $WAIT_PID exited; settling 15s for GPU/RAM release"
sleep 15

CKPT="runs/v3/loss_aug/03_cand1_only_h128_l4/training_h128_l4/checkpoint_epoch01.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "[chain] ERROR: $CKPT not found; cannot resume"
  exit 1
fi

echo "[chain] $(date): launching resume from $CKPT"

source ~/catan_mcts_venvs/mcts-study/bin/activate
exec python scripts/train_grid_inproc.py \
  --cache-path /home/chitii/catan_cache/cache_100k.pt \
  --out-root runs/v3/loss_aug/03_cand1_only_h128_l4 \
  --status-file runs/v3/dashboard/cell2.json \
  --epochs 10 \
  --early-stop-patience 0 \
  --batch-size 256 \
  --device auto \
  --rotate --rotate-mode random \
  --cells h128_l4 \
  --seed 0 \
  --resume-cell "h128_l4=$CKPT" \
  --mid-tournament-every 5 \
  --lambda-settle 0.20
