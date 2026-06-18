#!/usr/bin/env bash
# Build the catan_mcts_rs PyO3 extension into the venv. Links tch -> needs the
# pip-wheel libtorch env. Run after any Rust change to the PyO3 surface.
set -euo pipefail
source ~/.cargo/env
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate

TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
export LIBTORCH="$TORCH_DIR"
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="$TORCH_DIR/lib:${LD_LIBRARY_PATH:-}"
export CARGO_TARGET_DIR=/home/chitii/cmcts_target

cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/catan_mcts_rs
maturin develop --release 2>&1 | tail -25
