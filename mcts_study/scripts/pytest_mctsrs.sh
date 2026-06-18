#!/usr/bin/env bash
# Run pytest with libtorch on LD_LIBRARY_PATH so the tch-linked catan_mcts_rs
# extension imports (patchelf rpath isn't set in this env). Args -> pytest.
set -euo pipefail
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
export LD_LIBRARY_PATH="$TORCH_DIR/lib:${LD_LIBRARY_PATH:-}"
cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study
python -m pytest "$@"
