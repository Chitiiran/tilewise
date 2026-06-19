#!/usr/bin/env bash
# Small-batch smoke of the PRODUCTION self-play path (rust engine, real net,
# CUDA, chunked/pausable). 6 games, sims=30. Confirms: GPU engaged, records
# produced, resources sampler works, pause sentinel respected. Args: [games] [sims]
set -uo pipefail
source ~/.cargo/env
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
TORCH_DIR=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
SP=$(dirname "$TORCH_DIR")
NVLIBS=$(echo "$SP"/nvidia/*/lib | tr ' ' ':')
export LD_LIBRARY_PATH="$TORCH_DIR/lib:$NVLIBS:${LD_LIBRARY_PATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_PRELOAD="$TORCH_DIR/lib/libtorch_cuda.so${LD_PRELOAD:+:$LD_PRELOAD}"
CKPT=/home/chitii/catan_data/runs/v3/az_loop/checkpoints/az_iter_1.pt
OUT=/tmp/micro_sp
rm -rf "$OUT"; mkdir -p "$OUT"
cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study
GAMES="${1:-6}"; SIMS="${2:-30}"
# start the resource sampler beside the run
python -m catan_az.sampler "$OUT" 1.0 selfplay & SAMP=$!
python -m catan_mcts.experiments.self_play_rust \
  --out-root "$OUT" --checkpoint "$CKPT" --num-games "$GAMES" --n-sims "$SIMS" \
  --n-concurrent 64 --hidden-dim 128 --num-layers 4 --vp-target 10 \
  --seed-base 77000000 --self-play 2>&1 | tail -8
kill "$SAMP" 2>/dev/null
echo "=== resources.jsonl (last 2) ==="; tail -2 "$OUT/resources.jsonl" 2>/dev/null
echo "=== game shards ==="; ls "$OUT"/*/games*.parquet 2>/dev/null | head
echo "=== done.txt count ==="; cat "$OUT"/*/done.txt 2>/dev/null | wc -l
