#!/usr/bin/env bash
# End-to-end game throughput at B_MAX=64 / 64 concurrent games (vs the 3.26
# games/min measured at B_MAX=32/32). Tells us if the raw-forward batch gain
# survives game desync end-to-end. CUDA, deterministic, sims=200, skip B=1.
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
echo "re-exporting .ts at B_MAX=64 on CUDA..."
python scripts/reexport_spike_ts.py cuda 64 2>&1 | tail -1

cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/catan_mcts_rs
TP_BMAX=64 TP_GAMES=64 TP_SIMS=200 TP_SKIP_B1=1 \
  cargo test -p catan_mcts_rs --release --test throughput_compare -- --ignored --nocapture 2>&1 \
  | grep -E "device|config|batched|test result|error|panic"
