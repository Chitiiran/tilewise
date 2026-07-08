#!/usr/bin/env bash
# Arena throughput gate (Phase-1 exit, Task 5): champion az_iter_1 vs itself,
# batched Rust+TorchScript arena, wall-clock timed. Measures games/min for the
# cross-game leaf-batched GPU path (Task 1-4) so the roadmap's "<=1h wall-clock
# for a 40-game/sims=200 arena" gate has a real number behind it.
#
#   ./scripts/arena_throughput_gate.sh [N_GAMES] [SIMS] [OUT_DIR]
#
#   N_GAMES  default 40    (must be a multiple of 4 -- 4 arena rotations)
#   SIMS     default 200
#   OUT_DIR  default /home/chitii/catan_data/runs/v3/arena_throughput_gate
#
# Champion checkpoint is fixed: az_loop/checkpoints/az_iter_1.pt, used as BOTH
# candidate and champion (a self-play-vs-itself measurement has no promotion
# meaning -- this script is a throughput gate, not an arena verdict).
set -uo pipefail

N_GAMES="${1:-40}"
SIMS="${2:-200}"
OUT_DIR="${3:-/home/chitii/catan_data/runs/v3/arena_throughput_gate}"
CKPT="/home/chitii/catan_data/runs/v3/az_loop/checkpoints/az_iter_1.pt"

source ~/.cargo/env
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study

mkdir -p "$OUT_DIR"

# GPU env: replicate daily.py's _rust_cuda_env() EXACTLY by calling the real
# function and exporting what it computes (rather than re-deriving the nvidia
# lib enumeration / preload path in bash, which would drift from the source
# of truth). LD_PRELOAD is only set if libtorch_cuda.so actually exists (same
# guard as _rust_cuda_env -- a missing path in LD_PRELOAD aborts the
# interpreter at startup).
eval "$(python -c '
import sys
sys.path.insert(0, ".")
from catan_az.daily import _rust_cuda_env
env = _rust_cuda_env()
for k in ("LD_LIBRARY_PATH", "LD_PRELOAD", "CUBLAS_WORKSPACE_CONFIG"):
    v = env.get(k)
    if v is not None:
        v = v.replace("\x27", "\x27\\\x27\x27")
        print(f"export {k}=\x27{v}\x27")
')"

if [[ "$LD_PRELOAD" != *libtorch_cuda.so* ]]; then
  echo "[arena-gate] WARNING: libtorch_cuda.so not found -- arena will run on CPU" >&2
fi

python - "$N_GAMES" "$SIMS" "$OUT_DIR" "$CKPT" <<'PYEOF'
import sys
import time
from pathlib import Path

n_games = int(sys.argv[1])
sims = int(sys.argv[2])
out_dir = Path(sys.argv[3])
ckpt = Path(sys.argv[4])

from catan_az.arena import run_arena
from catan_az.config import AzConfig

cfg = AzConfig(
    sims=sims,
    arena_sims=sims,
    arena_games=n_games,
    engine="rust",
    arena_max_draw_rate=1.0,   # gate measures throughput, not a promotion verdict
    arena_min_decisive=0,
)

print(f"[arena-gate] games={n_games} sims={sims} out_dir={out_dir} ckpt={ckpt}",
      flush=True)

t0 = time.monotonic()
result = run_arena(
    candidate_ckpt=ckpt,
    champion_ckpt=ckpt,
    cfg=cfg,
    out_dir=out_dir,
    seed_base=71_000_000,
    device="cuda",
    n_concurrent=cfg.max_batch,
)
elapsed = time.monotonic() - t0

gpm = (result.games / elapsed) * 60.0 if elapsed > 0 else 0.0

print(f"[arena-gate] wall_clock_seconds={elapsed:.1f}", flush=True)
print(f"[arena-gate] games={result.games} wins_cand={result.wins_cand} "
      f"wins_champ={result.wins_champ} draws={result.draws} "
      f"timeouts={result.timeouts}", flush=True)
print(f"[arena-gate] games_per_min={gpm:.3f}", flush=True)
print(f"GATE_RESULT games={result.games} seconds={elapsed:.1f} gpm={gpm:.3f}",
      flush=True)
PYEOF
