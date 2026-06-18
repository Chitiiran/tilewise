#!/usr/bin/env bash
set -euo pipefail
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
echo "TORCH_DIR=$TORCH_DIR"
echo "--- lib contents (key libs) ---"
ls "$TORCH_DIR/lib" | grep -E 'libtorch|libc10|libgomp|nvrtc' || true
echo "--- cxx11 ABI ---"
python -c 'import torch; print("cxx11_abi:", torch._C._GLIBCXX_USE_CXX11_ABI)'
echo "--- torch version / cuda ---"
python -c 'import torch; print("torch:", torch.__version__, "cuda:", torch.version.cuda)'
echo "--- include dir present? ---"
ls "$TORCH_DIR/include" >/dev/null 2>&1 && echo "include/ exists" || echo "NO include/"
ls "$TORCH_DIR/share/cmake" >/dev/null 2>&1 && echo "share/cmake exists" || echo "NO share/cmake"
