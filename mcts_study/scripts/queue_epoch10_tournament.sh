#!/bin/bash
# Wait for the current in-process mid-tournament's spawn workers to
# exit (so GPU has bandwidth), then launch a standalone tournament of
# checkpoint_epoch10.pt against the same seed_base=19M for direct
# comparison to the epoch 5 baseline (15/120).
#
# Note: training process (PID 12903) does NOT exit when its
# mid-tournament finishes — it continues to epoch 11. So we cannot
# wait on training's PID. Instead poll for spawn_main workers from
# the running mid-tournament to disappear, indicating the tournament
# wrapped up.

set -uo pipefail
cd "$(dirname "$0")/.."

source ~/catan_mcts_venvs/mcts-study/bin/activate

# Wait for the running mid-tournament's 10 spawn workers to all exit.
echo "[queue] waiting for current mid-tournament workers to exit..."
while [[ "$(ps -ef | grep 'spawn_main' | grep -v grep | wc -l)" -gt 0 ]]; do
  sleep 30
done

echo "[queue] mid-tournament finished; settling 10s for GPU"
sleep 10

OUT="runs/v3/loss_aug/00_baseline_h128_l4_pilot/standalone_tournament_epoch10_seed19M"
mkdir -p "$OUT"

echo "[queue] launching epoch 10 standalone tournament"
exec python -m catan_mcts.experiments.e10_v3_tournament \
  --checkpoint runs/v3/loss_aug/00_baseline_h128_l4_pilot/training_h128_l4/checkpoint_epoch10.pt \
  --hidden-dim 128 --num-layers 4 \
  --num-games-per-seating 30 \
  --sims 100 --lookahead-depth 10 --base-sims-v3 200 \
  --seed-base 19000000 \
  --workers 10 --device cuda \
  --out-root "$OUT"
