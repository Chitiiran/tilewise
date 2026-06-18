#!/usr/bin/env bash
set -uo pipefail
source ~/.cargo/env
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
export LIBTORCH="$TORCH_DIR" LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="$TORCH_DIR/lib:${LD_LIBRARY_PATH:-}"
export CARGO_TARGET_DIR=/home/chitii/cmcts_target
echo "torch lib cuda files:"
ls "$TORCH_DIR/lib" | grep -E "cuda|cudart|nvrtc|c10_cuda" | head
echo "--- ldd of libtorch_cuda ---"
ldd "$TORCH_DIR/lib/libtorch_cuda.so" 2>&1 | grep -iE "not found|cuda" | head
echo "--- minimal tch cuda probe ---"
cat > /tmp/tchprobe.rs <<'EOF'
fn main() {
    println!("cuda_is_available = {}", tch::Cuda::is_available());
    println!("cuda_device_count = {}", tch::Cuda::device_count());
    println!("cudnn_is_available = {}", tch::Cuda::cudnn_is_available());
}
EOF
mkdir -p /tmp/tchprobe/src && cp /tmp/tchprobe.rs /tmp/tchprobe/src/main.rs
cat > /tmp/tchprobe/Cargo.toml <<'EOF'
[package]
name="tchprobe"
version="0.0.0"
edition="2021"
[dependencies]
tch="0.24.0"
EOF
cd /tmp/tchprobe
CARGO_TARGET_DIR=/home/chitii/tchprobe_target cargo run --quiet 2>&1 | tail -8
