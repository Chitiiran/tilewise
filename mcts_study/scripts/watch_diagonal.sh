#!/bin/bash
PID=624
LOG=/mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study/runs/v3/grid_pass100k_diagonal_2026-05-09T00-56.log
tail -n 0 -F "$LOG" 2>/dev/null &
TAIL_PID=$!
while kill -0 "$PID" 2>/dev/null; do
  sleep 30
done
echo "[monitor] python pid $PID exited"
kill "$TAIL_PID" 2>/dev/null
wait "$TAIL_PID" 2>/dev/null
exit 0
