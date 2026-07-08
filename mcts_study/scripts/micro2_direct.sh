#!/usr/bin/env bash
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
SP=$(dirname "$TORCH_DIR")
NVLIBS=$(echo "$SP"/nvidia/*/lib | tr ' ' ':')
export LD_LIBRARY_PATH="$TORCH_DIR/lib:$NVLIBS:${LD_LIBRARY_PATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_PRELOAD="$TORCH_DIR/lib/libtorch_cuda.so"
cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study
rm -rf /tmp/micro2
timeout 300 python -m catan_mcts.experiments.self_play_rust \
  --out-root /tmp/micro2 \
  --checkpoint /home/chitii/catan_data/runs/v3/az_loop/checkpoints/az_iter_1.pt \
  --num-games 2 --n-sims 10 --hidden-dim 128 --num-layers 4 \
  --seed-base 88000000 --self-play 2>&1 | grep -vE 'Tracer|dim_size|warnings.warn|Deprecat' | tail -25
echo "exit=${PIPESTATUS[0]}"
echo "=== records ==="; ls /tmp/micro2/*/games*.parquet 2>/dev/null; cat /tmp/micro2/*/done.txt 2>/dev/null | wc -l
