#!/bin/bash
RUNDIR=/mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/runs/2026-04-29T10-32-e5_lookahead_depth
for w in worker0 worker1 worker2 worker3; do
  d="$RUNDIR/$w"
  done_n=$(wc -l < "$d/done.txt" 2>/dev/null || echo 0)
  if [ -f "$d/skipped.csv" ]; then
    skip_n=$(( $(wc -l < "$d/skipped.csv") - 1 ))
    [ $skip_n -lt 0 ] && skip_n=0
  else
    skip_n=0
  fi
  shards=$(ls "$d"/games.*.parquet 2>/dev/null | wc -l)
  echo "$w: $done_n done, $skip_n skipped, $shards shards"
done
echo "---"
echo "All shards across workers:"
ls "$RUNDIR"/worker*/games.*.parquet 2>/dev/null | sed "s|$RUNDIR/||" | sort
