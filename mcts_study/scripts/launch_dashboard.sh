#!/bin/bash
set -e
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study
LOG=/home/chitii/catan_data/runs/v3/az_loop/dashboard_8099.log
setsid nohup python3 scripts/run_arena_dashboard.py --port 8099 > "$LOG" 2>&1 < /dev/null &
echo "dashboard launched pid $!  log=$LOG"
