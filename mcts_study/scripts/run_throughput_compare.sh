#!/usr/bin/env bash
set -uo pipefail
source ~/.cargo/env
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
export LIBTORCH="$TORCH_DIR" LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="$TORCH_DIR/lib:${LD_LIBRARY_PATH:-}"
export CARGO_TARGET_DIR=/home/chitii/cmcts_target
# deterministic CUDA kernels (the replayable canonical mode)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/catan_mcts_rs
cargo test -p catan_mcts_rs --release --test throughput_compare -- --ignored --nocapture 2>&1 \
  | grep -E "device|B=1|batched|SPEEDUP|test result|error|panic"
