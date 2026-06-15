#!/usr/bin/env bash
# Cron-ready daily AZ trainer entry. Preflight -> daily driver (archive is
# per-cycle inside the driver). Run from WSL with the mcts-study venv active.
#
#   ./scripts/run_az_day.sh [LOOP_ROOT]
#   MAX_ITERS=3 ./scripts/run_az_day.sh           # cap iterations for the day
#   touch <LOOP_ROOT>/STOP                          # graceful pause (<=1 game lost)
set -euo pipefail
LOOP_ROOT="${1:-/home/chitii/catan_data/runs/v3/az_loop}"
cd "$(dirname "$0")/.."   # mcts_study/

mkdir -p "$LOOP_ROOT"
LOCK="$LOOP_ROOT/daily.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "[az-day] another run is active (PID $(cat "$LOCK")) — abort"; exit 1
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

# Reap stale self-play procs from a previous dead run (self-heal).
pkill -f 'experiments.self_play_async' 2>/dev/null || true

python -m catan_az.daily --loop-root "$LOOP_ROOT" --max-iters "${MAX_ITERS:-1000}"
