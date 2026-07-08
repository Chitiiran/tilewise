#!/usr/bin/env bash
# Fetch numpy's exact ziggurat_constants.h and parse it into a Rust constants
# module (catan_mcts_rs/src/ziggurat_tables.rs). Every digit preserved verbatim.
set -euo pipefail
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate

OUT_DIR=/mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study/scripts
HDR="$OUT_DIR/ziggurat_constants.h"

# Pin to the numpy version in the venv so tables match the runtime exactly.
NP_VER=$(python -c 'import numpy; print(numpy.__version__)')
echo "numpy version: $NP_VER"

URL="https://raw.githubusercontent.com/numpy/numpy/v${NP_VER}/numpy/random/src/distributions/ziggurat_constants.h"
echo "fetching $URL"
curl -fsSL "$URL" -o "$HDR" || {
  echo "version-tagged fetch failed; trying main"
  curl -fsSL "https://raw.githubusercontent.com/numpy/numpy/main/numpy/random/src/distributions/ziggurat_constants.h" -o "$HDR"
}
echo "header bytes: $(wc -c < "$HDR")"
head -5 "$HDR"
