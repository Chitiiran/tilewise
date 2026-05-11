#!/bin/bash
# Launch Cell 0 baseline training (h128_l4, unmodified loss) with
# in-process mid-training tournaments every 5 epochs.
#
# Per Phase 1 standards (mid_training_tournament.py STANDARD_*):
#   - 120-game tournament every 5 epochs (epochs 5, 10, 15, 20, 25, 30)
#   - workers=10, device=cuda, seed_base=19M
#   - early-stop on >= 3-game PureGnn winrate drop vs previous tournament
#
# Training auto-resumes from --resume-cell after each mid-tournament
# (because the hook runs inline in the epoch loop; there is no manual
# kill/restart cycle). The 45-min cache load is paid ONCE.
#
# Resume usage:
#   ./launch_cell0_h128_l4.sh \
#     runs/v3/loss_aug/00_baseline_h128_l4_pilot/training_h128_l4/checkpoint_epochNN.pt
#
# Fresh-start usage:
#   ./launch_cell0_h128_l4.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CACHE_PATH="${CACHE_PATH:-/home/chitii/catan_cache/cache_100k.pt}"
OUT_ROOT="${OUT_ROOT:-runs/v3/loss_aug/00_baseline_h128_l4_pilot}"
STATUS_FILE="${STATUS_FILE:-runs/v3/dashboard/loss_aug_baseline_pilot.json}"
EPOCHS="${EPOCHS:-30}"
RESUME_CHECKPOINT="${1:-}"

mkdir -p "$OUT_ROOT" "$(dirname "$STATUS_FILE")"

ARGS=(
  --cache-path "$CACHE_PATH"
  --out-root "$OUT_ROOT"
  --status-file "$STATUS_FILE"
  --epochs "$EPOCHS"
  --early-stop-patience 0
  --batch-size 256
  --device auto
  --rotate --rotate-mode random
  --cells h128_l4
  --seed 0
  --mid-tournament-every 5
  # All other mid-tournament flags fall back to Phase 1 standards
  # encoded in mid_training_tournament.py (workers=10, device=cuda,
  # seed_base=19M, games_per_seating=30, sims=100, depth=10, base_sims=200).
)

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  echo "Resuming from $RESUME_CHECKPOINT"
  ARGS+=(--resume-cell "h128_l4=$RESUME_CHECKPOINT")
fi

source ~/catan_mcts_venvs/mcts-study/bin/activate
exec python scripts/train_grid_inproc.py "${ARGS[@]}"
