#!/usr/bin/env bash
# Timed profile: split the GPU phase into marshal / forward_is / extract to size
# how much of "GPU time" is the irreducible serial forward vs parallelizable
# host work. Decides the max-throughput architecture. Args: GAMES SIMS BMAX.
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
python scripts/reexport_spike_ts.py cuda "${3:-32}" 2>&1 | tail -1
cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/catan_mcts_rs
TP_GAMES="${1:-64}" TP_SIMS="${2:-50}" TP_BMAX="${3:-32}" TP_TIMED=1 \
  cargo test -p catan_mcts_rs --release --test cpu_profile -- --ignored --nocapture 2>&1 \
  | grep -E "device|games=|total|GPU|CPU|marshal|forward|extract|leaves|error|panic"
