#!/usr/bin/env bash
# One-shot Phase 4/5 verification. Writes a clear PASS/FAIL log to
# /tmp/phase5_check.log (read it directly — no tail buffering).
set -uo pipefail
LOG=/tmp/phase5_check.log
: > "$LOG"
exec >>"$LOG" 2>&1

source ~/.cargo/env
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
export LIBTORCH="$TORCH_DIR" LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="$TORCH_DIR/lib:${LD_LIBRARY_PATH:-}"
export CARGO_TARGET_DIR=/home/chitii/cmcts_target
ROOT=/mnt/c/dojo/catan_bot/.claude/worktrees/az-bots

echo "=== [1/5] cargo test (rng, mt, evaluator, state_internal) ==="
cd "$ROOT/catan_mcts_rs"
cargo test -p catan_mcts_rs --test rng_parity --test mt_parity --test evaluator_parity --test state_internal 2>&1 \
  | grep -E "test result|FAILED|error\[|panicked" || true

echo "=== [2/5] maturin develop (build extension w/ mcts module) ==="
cd "$ROOT/catan_mcts_rs"
maturin develop --release 2>&1 | tail -3

echo "=== [3/5] dump MCTS golden ==="
cd "$ROOT/mcts_study"
python scripts/dump_mcts_golden.py 2>&1 | tail -3

echo "=== [4/5] engine parity (Phase 4) ==="
python -m pytest tests/test_rust_engine_parity.py -q 2>&1 | tail -4

echo "=== [5/5] MCTS parity (Phase 5) ==="
python -m pytest tests/test_rust_mcts_parity.py -q 2>&1 | tail -15

echo "=== DONE ==="
