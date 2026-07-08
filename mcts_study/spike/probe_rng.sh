#!/usr/bin/env bash
set -euo pipefail
source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate
python - <<'PY'
import numpy as np
g = np.random.default_rng(12345)
print("bit_generator:", type(g.bit_generator).__name__)
print("state keys:", list(g.bit_generator.state.keys()))
st = g.bit_generator.state
print("has_uint32 / state:", st.get("has_uint32"), {k: v for k, v in st["state"].items()})
# Sample the three consumers the oracle uses, with a FRESH rng each time.
g1 = np.random.default_rng(777)
print("random() x3:", [g1.random() for _ in range(3)])
g2 = np.random.default_rng(777)
print("dirichlet([0.8]*4):", g2.dirichlet([0.8]*4).tolist())
g3 = np.random.default_rng(777)
print("choice([0,2,5], p=[.2,.3,.5]) x3:",
      [int(g3.choice([0,2,5], p=[.2,.3,.5])) for _ in range(3)])
# How does default_rng map an int seed -> PCG64 state? via SeedSequence.
ss = np.random.SeedSequence(777)
print("SeedSequence(777).generate_state(4, np.uint64):",
      ss.generate_state(4, dtype=np.uint64).tolist())
PY
