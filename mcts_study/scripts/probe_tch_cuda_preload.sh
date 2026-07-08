#!/usr/bin/env bash
set -uo pipefail
source ~/.cargo/env
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
export LIBTORCH="$TORCH_DIR" LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="$TORCH_DIR/lib:${LD_LIBRARY_PATH:-}"
# Force-load the CUDA backend so libtorch registers it (tch otherwise sees CPU).
export LD_PRELOAD="$TORCH_DIR/lib/libtorch_cuda.so${LD_PRELOAD:+:$LD_PRELOAD}"
cd /tmp/tchprobe
CARGO_TARGET_DIR=/home/chitii/tchprobe_target cargo run --quiet 2>&1 | tail -5
