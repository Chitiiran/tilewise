#!/usr/bin/env bash
set -uo pipefail
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
export LD_LIBRARY_PATH="$TORCH_DIR/lib:${LD_LIBRARY_PATH:-}"
cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study
SEEDS="${1:-2}"
SIMS="${2:-200}"
python scripts/validate_rust_path.py --seeds "$SEEDS" --n-sims "$SIMS" > /tmp/prodcheck.log 2>&1
echo "exit=$?"
grep -vE "TracerWarning|dim_size|warnings.warn|Deprecation" /tmp/prodcheck.log | tail -14
