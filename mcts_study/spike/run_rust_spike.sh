#!/usr/bin/env bash
set -euo pipefail
source ~/.cargo/env
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate

TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
echo "LIBTORCH=$TORCH_DIR"

# tch-rs: use the pip torch wheel as libtorch. cxx11 ABI=1 (probed earlier).
# torch 2.11 is newer than tch 0.24's pinned 2.9 -> bypass the version assert.
export LIBTORCH="$TORCH_DIR"
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="$TORCH_DIR/lib:${LD_LIBRARY_PATH:-}"

cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study/spike/rust
# Keep build artifacts off the slow /mnt/c mount.
export CARGO_TARGET_DIR=/home/chitii/tch_spike_target
cargo run --release 2>&1 | tail -40
