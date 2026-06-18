#!/usr/bin/env bash
# Block until BOTH the self-play gate and arena gate finish, then print both
# results. Returns promptly once both logs show a pytest summary line.
done_sp() { grep -qE "[0-9]+ (passed|failed|error)" /tmp/gate12.log 2>/dev/null; }
done_ar() { grep -qE "[0-9]+ (passed|failed|error)" /tmp/arena_gate.log 2>/dev/null; }
for i in $(seq 1 240); do
  if done_sp && done_ar; then break; fi
  sleep 15
done
echo "===== SELF-PLAY GATE ====="
tail -8 /tmp/gate12.log
echo "===== ARENA GATE ====="
tail -8 /tmp/arena_gate.log
