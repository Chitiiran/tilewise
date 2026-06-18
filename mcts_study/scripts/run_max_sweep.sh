#!/usr/bin/env bash
# Wait for any running throughput_comp (the B=64 game bench) to free the GPU,
# then run the raw-forward batch-size sweep to find the MAXIMUM throughput /
# VRAM ceiling. Writes /tmp/max_sweep.log.
set -uo pipefail
LOG=/tmp/max_sweep.log
: > "$LOG"
exec >>"$LOG" 2>&1
echo "waiting for GPU (throughput_comp) to free..."
while pgrep -f throughput_comp >/dev/null 2>&1; do sleep 20; done
echo "GPU free; running max sweep."
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
SP=$(dirname "$TORCH_DIR")
NVLIBS=$(echo "$SP"/nvidia/*/lib | tr ' ' ':')
export LD_LIBRARY_PATH="$TORCH_DIR/lib:$NVLIBS:${LD_LIBRARY_PATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study
python scripts/bench_batch_sweep.py 2>&1 | grep -vE "Tracer|dim_size|UserWarning|warn"
echo "=== MAX SWEEP DONE ==="
