#!/usr/bin/env bash
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
SP=$(dirname "$TORCH_DIR")
NVLIBS=$(echo "$SP"/nvidia/*/lib | tr ' ' ':')
export LD_LIBRARY_PATH="$TORCH_DIR/lib:$NVLIBS"
export LD_PRELOAD="$TORCH_DIR/lib/libtorch_cuda.so"
python - <<'PY'
import catan_mcts_rs as m
print("run_selfplay     :", hasattr(m, "run_selfplay"))
print("run_arena        :", hasattr(m, "run_arena"))
print("run_arena_games  :", hasattr(m, "run_arena_games"))
import torch
print("cuda available   :", torch.cuda.is_available())
PY
