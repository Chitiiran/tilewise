#!/usr/bin/env bash
# Run pytest with libtorch on LD_LIBRARY_PATH so the tch-linked catan_mcts_rs
# extension imports (patchelf rpath isn't set in this env). Args -> pytest.
set -euo pipefail
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
SP=$(dirname "$TORCH_DIR")
NVLIBS=$(echo "$SP"/nvidia/*/lib | tr ' ' ':')   # nvrtc/cublas/cudnn (cu13 wheel)
export LD_LIBRARY_PATH="$TORCH_DIR/lib:$NVLIBS:${LD_LIBRARY_PATH:-}"
# Deterministic CUDA kernels + force-load the CUDA backend so the Rust engine
# uses the GPU (tch links CPU libtorch; without this it silently runs on CPU).
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_PRELOAD="$TORCH_DIR/lib/libtorch_cuda.so${LD_PRELOAD:+:$LD_PRELOAD}"
cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study
python -m pytest "$@"
