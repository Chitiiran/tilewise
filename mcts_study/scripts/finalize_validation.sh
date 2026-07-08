#!/usr/bin/env bash
# Wait for the running 100-seed gate, then run the Phase-9 production-net
# cross-check + the production-path arena parity test. All results -> a log read
# directly (no tail buffering). Long-running; launch in background.
set -uo pipefail
LOG=/tmp/finalize_validation.log
: > "$LOG"
exec >>"$LOG" 2>&1
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
export LD_LIBRARY_PATH="$TORCH_DIR/lib:${LD_LIBRARY_PATH:-}"
ROOT=/mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study
cd "$ROOT"

echo "=== waiting for 100-seed gate (/tmp/gate100.log) ==="
while ! grep -qE "[0-9]+ (passed|failed|error)" /tmp/gate100.log 2>/dev/null; do sleep 30; done
echo "=== 100-seed gate result ==="
grep -E "[0-9]+ (passed|failed|error)" /tmp/gate100.log | tail -1

echo "=== Phase-9 production-net cross-check (4 seeds, sims=200, self-play) ==="
python scripts/validate_rust_path.py --seeds 4 --n-sims 200 2>&1 | grep -vE "TracerWarning|dim_size|warnings.warn|Deprecation" | tail -12

echo "=== production-path arena parity (run_arena rust vs python, 8 games) ==="
python -m pytest tests/test_daily_rust_engine.py::test_run_arena_rust_matches_python -q 2>&1 | tail -4

echo "=== FINALIZE DONE ==="
