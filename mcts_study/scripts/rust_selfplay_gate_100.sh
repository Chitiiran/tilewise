#!/usr/bin/env bash
# The full 100-seed self-play end-to-end gate (spec §5, NON-NEGOTIABLE).
# Runs the parametrized gate test over 100 seeds (override via SEEDS env).
# Must be GREEN before Rust self-play is trusted to feed the AZ loop.
set -uo pipefail
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
export LD_LIBRARY_PATH="$TORCH_DIR/lib:${LD_LIBRARY_PATH:-}"
export RUST_SELFPLAY_GATE_SEEDS="${RUST_SELFPLAY_GATE_SEEDS:-100}"
cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study
python -m pytest tests/test_rust_selfplay_gate.py -q 2>&1 | tail -20
