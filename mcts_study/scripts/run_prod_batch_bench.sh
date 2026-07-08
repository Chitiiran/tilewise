#!/usr/bin/env bash
# Production-config batched throughput: B_MAX=32, 32 concurrent games, sims=200,
# CUDA, deterministic. Skips the (very slow) B=1 baseline — already measured at
# 0.20 games/min. Re-exports the spike .ts at B_MAX=32 on CUDA first.
set -uo pipefail
source ~/.cargo/env
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
SP=$(dirname "$TORCH_DIR")
NVLIBS=$(echo "$SP"/nvidia/*/lib | tr ' ' ':')
export LIBTORCH="$TORCH_DIR" LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="$TORCH_DIR/lib:$NVLIBS:${LD_LIBRARY_PATH:-}"
export CARGO_TARGET_DIR=/home/chitii/cmcts_target
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_PRELOAD="$TORCH_DIR/lib/libtorch_cuda.so${LD_PRELOAD:+:$LD_PRELOAD}"

cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study
echo "re-exporting .ts at B_MAX=32 on CUDA..."
python scripts/reexport_spike_ts.py cuda 32 2>&1 | tail -1

cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/catan_mcts_rs
TP_BMAX=32 TP_GAMES=32 TP_SIMS=200 TP_SKIP_B1=1 \
  cargo test -p catan_mcts_rs --release --test throughput_compare -- --ignored --nocapture 2>&1 \
  | grep -E "device|config|B=1|batched|test result|error|panic"
