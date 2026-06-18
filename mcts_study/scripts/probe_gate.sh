#!/usr/bin/env bash
PID=33394
echo "=== status ==="
grep -E "State|Threads|VmRSS" /proc/$PID/status 2>/dev/null
echo "=== main wchan ==="; cat /proc/$PID/wchan 2>/dev/null; echo
echo "=== per-thread (state, cpu sec) ==="
for t in /proc/$PID/task/*; do
  tid=$(basename "$t")
  awk '{print "tid", '"$tid"', "state", $3, "cpu_sec", ($14+$15)/100}' "$t/stat" 2>/dev/null
done
echo "=== aggregate tree CPU sample ==="
S1=$(awk '{s+=$14+$15} END{print s}' /proc/$PID/task/*/stat 2>/dev/null)
sleep 3
S2=$(awk '{s+=$14+$15} END{print s}' /proc/$PID/task/*/stat 2>/dev/null)
echo "delta ticks over 3s: $((S2 - S1))  (0 = idle/blocked, ~1200 = ~4 cores busy)"
