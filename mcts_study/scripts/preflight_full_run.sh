#!/usr/bin/env bash
LR=/home/chitii/catan_data/runs/v3/az_loop
echo "=== loop-root contents ==="
ls "$LR" 2>/dev/null | head -40
echo "=== STOP / PAUSE markers ==="
ls -la "$LR/STOP" "$LR/PAUSE" 2>/dev/null || echo "(none)"
echo "=== daily_state.json ==="
cat "$LR/daily_state.json" 2>/dev/null
echo
echo "=== journal tail ==="
tail -4 "$LR/journal.csv" 2>/dev/null
echo "=== ladder ==="
cat "$LR/ladder.json" 2>/dev/null
echo
echo "=== iter dirs ==="
ls -d "$LR"/iter_* 2>/dev/null | tail -5
echo "=== running az/self_play procs ==="
ps -eo pid,etimes,comm,args 2>/dev/null | grep -E "catan_az.daily|self_play" | grep -v grep | head
echo "=== disk free (loop fs) ==="
df -h "$LR" 2>/dev/null | tail -1
echo "=== config snapshot (engine/sims/concurrency) ==="
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate 2>/dev/null
python -c "from catan_az.config import AzConfig; c=AzConfig(); print('engine',c.engine,'sims',c.sims,'n_concurrent',c.n_concurrent,'max_batch',c.max_batch,'games_per_iter',c.games_per_iter,'arena_games',c.arena_games,'promote',c.promote_threshold,'max_iters_per_model',c.max_iters_per_model)" 2>&1 | tail -1
