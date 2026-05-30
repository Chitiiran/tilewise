#!/bin/bash
# Launch the diagonal-pass training run in WSL with crash-prevention guards.
#
# Guards added 2026-05-09 after previous launch died ~95s after cache load:
#   1. WSL keepalive: a `sleep infinity` process holds the distro open even
#      if the spawning shell exits. Without this, WSL2 may auto-shutdown.
#   2. Memory observer: samples /proc every 5s while training runs.
#      Forensic evidence if the trainer dies again.
#   3. Set OOM-score-adjust to make the trainer LESS likely to be picked
#      by the kernel OOM killer than other processes.
#
# Pre-req: ~/.wslconfig has [wsl2] vmIdleTimeout=-1 (set 2026-05-09).
set -e
cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study

TS=$(date +%Y-%m-%dT%H-%M)
LOG=runs/v3/grid_pass100k_diagonal_${TS}.log
OBS_LOG=runs/v3/grid_pass100k_diagonal_${TS}.observ.log
echo "log=$LOG"
echo "obs_log=$OBS_LOG"

# --- Guard 1: WSL keepalive (detached) -----------------------------------
# Holds WSL alive across this shell's exit. PID written so we can kill
# later if needed (or just leave it; it costs ~50 KB RAM).
setsid bash -c 'exec sleep infinity' </dev/null >/dev/null 2>&1 &
KEEPALIVE_PID=$!
echo "$KEEPALIVE_PID" > /tmp/diagonal_keepalive_pid
echo "keepalive_pid=$KEEPALIVE_PID"

# --- Guard 2: launch the trainer (detached) ------------------------------
setsid /home/chitii/catan_mcts_venvs/mcts-study/bin/python scripts/train_grid_inproc.py \
  --cache-path /home/chitii/catan_cache/cache_100k.pt \
  --out-root runs/v3/grid_pass100k \
  --status-file runs/v3/dashboard/grid_pass100k.json \
  --epochs 20 --early-stop-patience 0 --batch-size 256 \
  --device auto --rotate --rotate-mode random \
  --cells h128_l4,h64_l3 \
  --resume-cell h64_l3=runs/v3/grid_pass100k/training_h64_l3/checkpoint_epoch10.pt \
  </dev/null >"$LOG" 2>&1 &
P=$!
echo "trainer_pid=$P"
echo "$P" > /tmp/diagonal_pid
echo "$LOG" > /tmp/diagonal_log

# Lower OOM kill priority — kernel will pick OTHER procs first.
echo -1000 > /proc/$P/oom_score_adj 2>/dev/null || \
  echo "[warn] couldn't set oom_score_adj for $P (need root); skipping"

# --- Guard 3: memory observer (detached) ---------------------------------
# Samples /proc/<P>/status + /proc/meminfo every 5s. Exits when P dies.
setsid /home/chitii/catan_mcts_venvs/mcts-study/bin/python scripts/observe_memory.py \
  --pid "$P" --log "$OBS_LOG" --interval 5 \
  </dev/null >/dev/null 2>&1 &
OBS_PID=$!
echo "$OBS_PID" > /tmp/diagonal_observer_pid
echo "observer_pid=$OBS_PID"

# Quick health check
sleep 3
ps -fp "$P" || echo "[error] trainer process died within 3s"
ps -fp "$OBS_PID" >/dev/null || echo "[warn] observer died within 3s"
echo "---"
ls -la "$LOG"
echo "---"
echo "watch:    tail -F $LOG"
echo "observe:  tail -F $OBS_LOG | jq -c '{ts,label,rss_gb,sys_available_gb}'"
echo "kill all: kill \$(cat /tmp/diagonal_pid) \$(cat /tmp/diagonal_observer_pid) \$(cat /tmp/diagonal_keepalive_pid)"
