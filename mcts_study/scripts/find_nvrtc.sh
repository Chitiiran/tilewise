#!/usr/bin/env bash
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
SP=$(dirname "$TORCH_DIR")  # site-packages
echo "site-packages: $SP"
find "$SP/nvidia" -name 'libnvrtc*' 2>/dev/null
echo "--- cusparse/cublas dirs (often needed too) ---"
ls -d "$SP"/nvidia/*/lib 2>/dev/null
