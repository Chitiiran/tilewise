#!/usr/bin/env bash
for i in $(seq 1 150); do
  if grep -q "=== DONE\|error\[\|panicked\|FAILED" /tmp/phase5_check.log 2>/dev/null; then
    break
  fi
  sleep 5
done
echo "===== FINAL LOG ====="
cat /tmp/phase5_check.log
