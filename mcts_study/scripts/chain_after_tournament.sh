#!/bin/bash
# Watchdog: wait for the standalone-tournament parent PID to exit, then
# launch the Cell 0 training loop with mid-tournaments. One-shot — does
# NOT re-arm. Intended use: launch this once after the standalone
# tournament that produces the epoch-5 baseline number.
#
# usage: chain_after_tournament.sh <tournament_pid> <resume_checkpoint>

set -uo pipefail

WAIT_PID="${1:?need tournament PID}"
RESUME_CKPT="${2:?need resume checkpoint path}"

echo "[chain] waiting for PID $WAIT_PID to exit before launching training"

# Use bash's `wait` if PID is a child of this shell; else poll.
# tournament PID 12536 is a child of the prior shell, not this one;
# fall through to polling.
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done

echo "[chain] PID $WAIT_PID has exited; launching training loop"
echo "[chain] resume from: $RESUME_CKPT"

# Brief settle so GPU/RAM free up cleanly
sleep 10

# Move into mcts_study root and exec the launcher in foreground
# (caller can re-direct stdout to a log file if they want detached output)
cd "$(dirname "$0")/.."
exec ./scripts/launch_cell0_h128_l4.sh "$RESUME_CKPT"
