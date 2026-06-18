# Rust-MCTS + TorchScript-GNN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Catan AlphaZero MCTS (tree + state + apply_action) and GNN inference entirely into Rust (via `tch-rs` loading a TorchScript-exported net), eliminating the millions of per-node Python↔Rust↔PyTorch boundary crossings that make self-play and arena latency-bound, while keeping PyTorch training and the AZ loop orchestration unchanged.

**Architecture:** A new `catan_mcts_rs` Rust crate calls the existing `catan_engine` library directly (no PyO3 per node) and runs the GNN through a `TorchScriptEvaluator` wrapping a `tch::CModule`. The Python AZ loop calls coarse PyO3 entry points (`run_selfplay`, `run_arena`) — one crossing per *stage*, not per node. A small `export_torchscript.py` step traces the trained PyTorch net into the `.ts` the Rust side loads. Every Rust unit is double-verified (golden-parity vs the Python oracle AND differential/property tests) to BIT-EXACT for all decisions.

**Tech Stack:** Rust (existing `catan_engine` + new `catan_mcts_rs`), `tch` 0.24 against the pip `torch==2.11.0+cu130` libtorch wheel, PyO3/maturin, NumPy PCG64 RNG replicated in Rust, PyTorch (training, unchanged), pytest + cargo test.

---

## Phase 0 — COMPLETE (gate passed 2026-06-17)

The spike (`mcts_study/spike/`) already proved the make-or-break gate **bit-exact**:
- `torch.jit.script(GnnModel)` FAILS (PyG `HeteroConv` varargs) — documented, not used.
- A traced plain-tensor `TensorWrapper` around `GnnModel` round-trips through `tch-rs` 0.24 with **max abs diff = 0.0** for both value and policy over 50 fixed states, and **generalizes** to unseen states (features flow, not frozen).
- Build recipe: `LIBTORCH=<venv>/.../torch`, `LIBTORCH_USE_PYTORCH=1`, `LIBTORCH_BYPASS_VERSION_CHECK=1`, `LD_LIBRARY_PATH=<torch>/lib`, `CARGO_TARGET_DIR=/home/chitii/...` (off /mnt/c).
- See `project_phase0_torchscript_spike_2026_06_17` memory for the full result.

**Consequence for this plan:** we do NOT need a fixed-topology net rewrite. The train→infer seam is the traced `TensorWrapper` (plain tensors in/out). The Phase-0 spike code is the basis for Task 1 (export) and Task 8 (evaluator). The spike is kept under `spike/` for reference; production code is written fresh under the locations below.

---

## Locked decisions (read before implementing)

These were settled during design + the spike. Do not deviate without bringing it to the user.

1. **RNG strategy = replicate NumPy PCG64 in Rust (spec §5 option A), NOT switch both sides.**
   Rationale: keeps the Python oracle *unmodified* (the verification philosophy rests on "existing Python is the oracle"); the cross-check in Phase 9 then validates historical-Python vs Rust, not modified-Python vs Rust. The RNG only feeds three consumers — `rng.random()` (chance sampling), `rng.dirichlet(alpha)` (root noise), `rng.choice(p=...)` (temperature sampling) — each independently golden-testable. `np.random.default_rng(seed)` = PCG64 seeded via `SeedSequence`. This is Phase 2 and is the single hardest unit; it is gated by golden vectors dumped from numpy.

2. **Bit-exact for all DECISIONS (visit counts, chosen action, winners, records) — zero tolerance.** Net-forward floats may match to FP-identity (proven 0.0 on CPU in Phase 0); if a CUDA kernel ever forces a relaxation it must (a) be justified by naming the kernel, (b) documented in the test, (c) NOT change any decision.

3. **`catan_mcts_rs` is a NEW workspace member** depending on `catan_engine` (as a lib) + `tch`. It exposes its OWN PyO3 module (`catan_mcts_rs`) with coarse entry points. The existing `_engine` module is untouched.

4. **Self-play records flow back to Python as structured data; the existing `SelfPlayRecorder` writes the parquet.** Rust returns `Vec<GameRecord>` over the thin boundary; Python feeds them to `SelfPlayRecorder` so the on-disk schema is identical *by construction*. (Do NOT reimplement parquet writing in Rust.)

5. **Orchestration (`catan_az.daily`, ladder, journal, window, config) is UNCHANGED** except: (a) `loop._default_train` gains a TorchScript-export call; (b) `daily.py` self-play + arena stages call the new Rust entry points instead of the Python async drivers.

6. **The GNN viewer is `current_player` (ego-centric).** The value head is ego-relative; Rust MUST rotate it to absolute-seat order before backup (`value_abs[(leaf_mover+offset)%4] = value[offset]`), exactly as `async_mcts._expand_and_evaluate` does. Terminal `returns()` is already absolute.

---

## Reference oracle facts (from reading the Python this session)

- **`async_mcts.py`** — UCB `q + c·prior·√(parent.visits)/(1+child.visits)`, c=1.4, q=0 when child unvisited. `_select_child` uses strict `>` so ties keep the FIRST child in dict-insertion (= prior/expansion) order. Root: expand+evaluate, apply Dirichlet, count root as 1 visit, then `n_sims-1` sims. Per-player `move_index`. Chance fast-path (sample via rng.random over cumulative `chance_outcomes()`). Single-legal fast-path. tau schedule: tau=1 for first `temp_moves=30` per-player decisions then tau=0 argmax (self-play); arena = eps=0 + argmax.
- **`GameResult`**: seed, terminal, winner (`argmax(returns)` if `max>0` else -1), final_vp (from `stats()["players"][i]["vp_final"]`), length_in_moves (TOTAL engine steps incl. chance/forced), action_history (`engine.action_history()`), moves[] with (current_player, move_index, legal_mask[280], visit_counts[280], action_taken, root_value).
- **Self-play RNG**: ONE `np.random.default_rng(seed)` per game drives chance + dirichlet + temperature.
- **Arena** (`arena.py`): seating `_BASE=["cand","champ","cand","champ"]`, rotation `BASE[rot:]+BASE[:rot]`. `seed_plan`: per_rot=games/4, seed=`seed_base + rot*10000 + i`. Arena MCTS rng = `default_rng(seed+11)` (cand) / `default_rng(seed+13)` (champ); arena is greedy so rng only drives MCTS-internal chance sampling. The arena GAME-level chance fast-path uses a SEPARATE `random.Random(seed)` (stdlib Mersenne!). Winner = `returns().index(1.0)` else -1; timeout/step-cap → `_vp_leader_margin` (vp, then settlements+cities, then -1) marked timed_out; `vp_margin` = top−second.
- **Engine native API** (`catan_engine/src/engine.rs`): `Engine::new(seed)` / `with_rules(seed, vp_target, bonuses)`, `legal_actions()->Vec<u32>`, `legal_mask()`, `step(u32)`, `apply_chance_outcome(u32)`, `is_terminal()`, `is_chance_pending()`, `chance_outcomes()->Vec<(u32,f64)>`, derived `Clone`, `stats()->&GameStats`, `action_history()->&[u32]`, `state.current_player`, `state.vp[p]`. Observation: `observation::build_observation(&state, viewer)` → hex[19,8], vertex[54,13], edge[72,6], scalars[59], legal_mask[280]. CHANCE_FLAG `0x8000_0000` marks chance entries in history.
- **Net**: `GnnModel(hidden_dim=128, num_layers=4)`, checkpoint loaded via `obj["model_state"]`. ACTION_SPACE_SIZE=280.

---

## File structure

**New Rust crate `catan_mcts_rs/`** (workspace member at `az-bots/catan_mcts_rs/`):
- `Cargo.toml` — deps: `catan_engine` (path), `tch`, `pyo3`, `numpy` (only for the PyO3 return types), `rand`/`rand_pcg` (PCG64 primitive).
- `src/lib.rs` — PyO3 module `catan_mcts_rs`: `run_selfplay`, `run_arena`, plus test-only parity hooks.
- `src/rng.rs` — NumPy PCG64 replica: `NpRng` with `random_f64()`, `dirichlet(&[f64])`, `choice(&[i64], &[f64])`, `standard_gamma` (dirichlet dep). Seeded via `SeedSequence` replica.
- `src/seedseq.rs` — NumPy `SeedSequence` replica → 2×u64 PCG64 (state, inc) init.
- `src/evaluator.rs` — `TorchScriptEvaluator`: wraps `tch::CModule`, `evaluate(&[&Observation]) -> Vec<(Vec<f32> /*logits[280]*/, [f32;4] /*value*/)>` batched; arena variant holds two modules.
- `src/mcts.rs` — `Node`, `AsyncMcts`-equivalent `search(root, n_sims) -> [i32;280]` visit counts, UCB, value rotation, Dirichlet, `temperature_sample`, `best_action`.
- `src/selfplay.rs` — `play_one_game` (chance/single-legal fast-paths, tau schedule, move recording) → `GameRecord`.
- `src/arena.rs` — `play_arena_game`, seating, vp tiebreak, `seed_plan`.
- `tests/` — Rust-side golden + property tests (cargo test).

**New Python files:**
- `mcts_study/catan_gnn/export_torchscript.py` — `export(checkpoint, out_ts, hidden_dim, num_layers)`: load `GnnModel`, wrap in `TensorWrapper`, `torch.jit.trace`, save `.ts`.
- `mcts_study/tests/test_export_torchscript.py` — parity of `.ts` vs eager (bit-exact, 50 states).
- `mcts_study/tests/test_rust_mcts_parity.py` — the cross-language gate (drives Rust + Python, asserts identical).
- `mcts_study/scripts/dump_rng_golden.py` — dump numpy golden vectors for Rust rng tests.
- `mcts_study/scripts/dump_mcts_golden.py` — dump Python MCTS visit-counts/action for fixed (seed, net, state).

**Modified Python files:**
- `mcts_study/catan_az/loop.py` (`_default_train`) — add `export_torchscript` call after `checkpoint_best.pt`.
- `mcts_study/catan_az/daily.py` — self-play + arena stages call `catan_mcts_rs` instead of the Python async drivers (behind a feature flag for the Phase-9 A/B).
- `az-bots/Cargo.toml` — add `catan_mcts_rs` to `members`.

---

## Phase 1 — TorchScript export bridge (Python)

### Task 1: `export_torchscript.py` with bit-exact trace + golden test

**Files:**
- Create: `mcts_study/catan_gnn/export_torchscript.py`
- Test: `mcts_study/tests/test_export_torchscript.py`

- [ ] **Step 1: Write the failing test**

```python
# mcts_study/tests/test_export_torchscript.py
"""Exported .ts must reproduce eager GnnModel bit-exact (spec §5.1)."""
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Batch
from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg
from catan_gnn.export_torchscript import export, TensorWrapper
from catan_mcts.adapter import CatanGame


def _states(n):
    import random
    out, seed = [], 0
    while len(out) < n:
        g = CatanGame(); st = g.new_initial_state(); rng = random.Random(seed); seed += 1
        for _ in range(rng.randrange(1, 60)):
            if st.is_terminal(): break
            la = st.legal_actions(); st.apply_action(la[rng.randrange(len(la))])
        if not st.is_terminal(): out.append(st._engine.observation())
    return out


def test_exported_ts_bit_exact(tmp_path):
    model = GnnModel(hidden_dim=32, num_layers=2).eval()  # small for speed
    ckpt = tmp_path / "m.pt"; torch.save({"model_state": model.state_dict()}, ckpt)
    ts = tmp_path / "m.ts"
    export(checkpoint=ckpt, out_ts=ts, hidden_dim=32, num_layers=2)
    loaded = torch.jit.load(str(ts)).eval()
    max_dv = max_dl = 0.0
    for o in _states(50):
        hx = torch.from_numpy(np.ascontiguousarray(o["hex_features"], dtype=np.float32))
        vx = torch.from_numpy(np.ascontiguousarray(o["vertex_features"], dtype=np.float32))
        ex = torch.from_numpy(np.ascontiguousarray(o["edge_features"], dtype=np.float32))
        sc = torch.from_numpy(np.ascontiguousarray(o["scalars"], dtype=np.float32))
        with torch.no_grad():
            rv, rl = model(Batch.from_data_list([state_to_pyg(o)]))
            tv, tl = loaded(hx, vx, ex, sc)
        max_dv = max(max_dv, (tv - rv).abs().max().item())
        max_dl = max(max_dl, (tl - rl).abs().max().item())
    assert max_dv == 0.0 and max_dl == 0.0, f"dv={max_dv} dl={max_dl}"
```

- [ ] **Step 2: Run test to verify it fails**

Run (WSL): `source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study && python -m pytest tests/test_export_torchscript.py -v`
Expected: FAIL — `ModuleNotFoundError: catan_gnn.export_torchscript`.

- [ ] **Step 3: Write the implementation**

```python
# mcts_study/catan_gnn/export_torchscript.py
"""Train→infer bridge: trace GnnModel into a plain-tensor TorchScript module.

torch.jit.script fails on PyG HeteroConv varargs; torch.jit.trace on the raw
model fails on the HeteroData input. So we wrap GnnModel in a plain-tensor
module (Catan is fixed-topology) and trace THAT. Proven bit-exact + generalizing
in the Phase-0 spike (project_phase0_torchscript_spike_2026_06_17).
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import _H2V_EI, _V2H_EI, _V2E_EI, _E2V_EI
from catan_mcts.adapter import CatanGame
import numpy as np


class TensorWrapper(nn.Module):
    """forward(hex[19,8], vertex[54,13], edge[72,6], scalars[59]) -> (value[1,4], logits[1,280])."""

    def __init__(self, model: GnnModel) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("h2v", _H2V_EI.clone())
        self.register_buffer("v2h", _V2H_EI.clone())
        self.register_buffer("v2e", _V2E_EI.clone())
        self.register_buffer("e2v", _E2V_EI.clone())

    def forward(self, hex_x, vertex_x, edge_x, scalars):
        data = HeteroData()
        data["hex"].x = hex_x
        data["vertex"].x = vertex_x
        data["edge"].x = edge_x
        data["hex", "to", "vertex"].edge_index = self.h2v
        data["vertex", "to", "hex"].edge_index = self.v2h
        data["vertex", "to", "edge"].edge_index = self.v2e
        data["edge", "to", "vertex"].edge_index = self.e2v
        data.scalars = scalars.view(1, -1)
        data["hex"].batch = torch.zeros(hex_x.size(0), dtype=torch.long)
        data["vertex"].batch = torch.zeros(vertex_x.size(0), dtype=torch.long)
        data["edge"].batch = torch.zeros(edge_x.size(0), dtype=torch.long)
        return self.model(data)  # (value, logits)


def _example():
    st = CatanGame().new_initial_state()
    o = st._engine.observation()
    f = lambda k: torch.from_numpy(np.ascontiguousarray(o[k], dtype=np.float32))
    return f("hex_features"), f("vertex_features"), f("edge_features"), f("scalars")


def export(*, checkpoint: Path, out_ts: Path, hidden_dim: int, num_layers: int) -> Path:
    model = GnnModel(hidden_dim=hidden_dim, num_layers=num_layers)
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)
    model.eval()
    wrapper = TensorWrapper(model).eval()
    traced = torch.jit.trace(wrapper, _example(), strict=True)
    Path(out_ts).parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_ts))
    return Path(out_ts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_export_torchscript.py -v`
Expected: PASS (`dv=0.0 dl=0.0`).

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_gnn/export_torchscript.py mcts_study/tests/test_export_torchscript.py
git commit -m "feat(gnn): TorchScript export bridge (traced plain-tensor wrapper, bit-exact)"
```

---

## Phase 2 — NumPy PCG64 RNG replica (Rust) — the hardest unit, gated by golden vectors

> All three subtasks follow: dump numpy golden → write failing Rust test embedding the golden → implement → green. The crate is created in Task 2.

### Task 2: Create `catan_mcts_rs` crate skeleton + SeedSequence→PCG64 init parity

**Files:**
- Create: `catan_mcts_rs/Cargo.toml`, `catan_mcts_rs/src/lib.rs`, `catan_mcts_rs/src/seedseq.rs`, `catan_mcts_rs/src/rng.rs`
- Modify: `az-bots/Cargo.toml` (add member)
- Create: `mcts_study/scripts/dump_rng_golden.py`
- Test: `catan_mcts_rs/tests/rng_parity.rs`

- [ ] **Step 1: Dump numpy golden for seeding + raw draws**

```python
# mcts_study/scripts/dump_rng_golden.py
"""Dump numpy PCG64 golden values for the Rust RNG replica tests.
Prints Rust-pasteable consts. Run in the venv."""
import numpy as np

def main():
    for seed in (0, 1, 777, 20_000_000):
        ss = np.random.SeedSequence(seed)
        st = ss.generate_state(4, dtype=np.uint64).tolist()
        g = np.random.default_rng(seed)
        bg = g.bit_generator.state["state"]
        print(f"// seed={seed}")
        print(f"//   seedseq.generate_state(4,u64) = {st}")
        print(f"//   pcg64 state = {bg['state']}, inc = {bg['inc']}")
        draws = [np.random.default_rng(seed).random() for _ in range(1)]
        g2 = np.random.default_rng(seed)
        first5 = [g2.random() for _ in range(5)]
        print(f"//   random() x5 = {first5}")

if __name__ == "__main__":
    main()
```

Run: `python scripts/dump_rng_golden.py` and paste the values into the Rust test below. (Numbers shown in the test are illustrative placeholders — REPLACE with the script's actual output before running.)

- [ ] **Step 2: Write the failing Rust test (golden + property)**

```rust
// catan_mcts_rs/tests/rng_parity.rs
use catan_mcts_rs::rng::NpRng;

#[test]
fn seedseq_pcg64_init_matches_numpy() {
    // GOLDEN: paste pcg64 (state, inc) for seed 777 from dump_rng_golden.py
    let rng = NpRng::from_seed(777);
    let (state, inc) = rng.raw_state();
    assert_eq!(state, 33261208707367790463622745601869196757u128 /* REPLACE if seed!=12345 */,
               "PCG64 state mismatch for seed 777");
    assert_eq!(inc, 268209174141567072605526753992732310247u128, "PCG64 inc mismatch");
}

#[test]
fn random_f64_matches_numpy() {
    // GOLDEN: random() x5 for seed 777 from dump_rng_golden.py
    let golden = [0.6110939299469712, 0.38281659045082816, 0.6000705254490022,
                  /* REPLACE next two from script output */ 0.0, 0.0];
    let mut rng = NpRng::from_seed(777);
    for (i, &g) in golden.iter().enumerate().take(3) {
        let x = rng.random_f64();
        assert_eq!(x.to_bits(), g.to_bits(), "draw {i}: {x} != {g}");
    }
}
```

- [ ] **Step 3: Add the crate + minimal `lib.rs` + SeedSequence + PCG64**

```toml
# catan_mcts_rs/Cargo.toml
[package]
name = "catan_mcts_rs"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]   # rlib so cargo test sees it; cdylib for PyO3

[dependencies]
catan_engine = { path = "../catan_engine" }
tch = "0.24.0"
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"
```

```rust
// az-bots/Cargo.toml -> members = ["catan_engine", "catan_mcts_rs"]
```

```rust
// catan_mcts_rs/src/lib.rs
pub mod seedseq;
pub mod rng;
```

Implement `seedseq.rs` as a faithful port of NumPy's `SeedSequence` (the hashmix/`generate_state` algorithm — XSHRR constants, `cycle`, `mix`/`hashmix`). Reference: numpy `_seedseq.pyx`. `generate_state(n, u64)` must match the golden. Implement `rng.rs` `NpRng` = PCG64 (`pcg_oneseq_128` / numpy's `pcg64`): state is 128-bit LCG (`MULT = 0x2360ed051fc65da44385df649fccf645`), `inc` from two of the four SeedSequence u64s (numpy seeds PCG64 with the SeedSequence's first 2 u64 as state-high/low and next 2 as inc-high/low, with the standard `pcg_setseq_128_srandom_r` bump). `random_f64()` = numpy's `next_double`: `(rng.next_u64() >> 11) * (1.0 / 9007199254740992.0)`. Expose `raw_state() -> (u128, u128)` for the init test.

- [ ] **Step 4: Build + run tests**

Run (WSL, via a script file — see `spike/run_rust_spike.sh` env pattern):
`cd catan_mcts_rs && CARGO_TARGET_DIR=/home/chitii/cmcts_target cargo test --test rng_parity -- --nocapture`
Expected: PASS (state/inc and the first 3 `random()` draws are bit-identical).

- [ ] **Step 5: Commit**

```bash
git add catan_mcts_rs/ az-bots/Cargo.toml mcts_study/scripts/dump_rng_golden.py
git commit -m "feat(mcts-rs): NumPy SeedSequence+PCG64 replica, random_f64 bit-exact vs numpy"
```

### Task 3: `dirichlet` + `choice` parity (the two remaining RNG consumers)

**Files:**
- Modify: `catan_mcts_rs/src/rng.rs`
- Modify: `mcts_study/scripts/dump_rng_golden.py`
- Test: `catan_mcts_rs/tests/rng_parity.rs`

- [ ] **Step 1: Extend the golden dump**

Append to `dump_rng_golden.py`: for seed 777, print `np.random.default_rng(777).dirichlet([0.8]*4).tolist()` and a sequence of `choice` results: `g=default_rng(777); [int(g.choice([0,2,5], p=[.2,.3,.5])) for _ in range(5)]`. Also dump `standard_gamma` building blocks if needed for debugging: `default_rng(777).standard_gamma(0.8, size=4).tolist()`.

- [ ] **Step 2: Write failing tests**

```rust
// add to catan_mcts_rs/tests/rng_parity.rs
#[test]
fn dirichlet_matches_numpy() {
    // GOLDEN: dirichlet([0.8;4]) for seed 777
    let golden = [0.3321022315847162, 0.32019937543444776,
                  0.05942953490460936, 0.28826885807622665];
    let mut rng = NpRng::from_seed(777);
    let out = rng.dirichlet(&[0.8, 0.8, 0.8, 0.8]);
    for (i, (&a, &b)) in out.iter().zip(golden.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "dirichlet[{i}]: {a} != {b}");
    }
}

#[test]
fn choice_matches_numpy() {
    // GOLDEN: choice([0,2,5], p=[.2,.3,.5]) x3 for seed 777 = [5,2,5]
    let golden = [5i64, 2, 5];
    let mut rng = NpRng::from_seed(777);
    let items = [0i64, 2, 5];
    let p = [0.2, 0.3, 0.5];
    for (i, &g) in golden.iter().enumerate() {
        assert_eq!(rng.choice(&items, &p), g, "choice {i}");
    }
}
```

- [ ] **Step 3: Implement `dirichlet` + `choice` + `standard_gamma`**

`dirichlet(alpha)`: numpy draws `y_i = standard_gamma(alpha_i)` then normalizes `y / sum(y)`. Port numpy's `standard_gamma` (the Marsaglia-Tsang method for alpha≥1 and the alpha<1 boost `standard_gamma(a+1) * U^(1/a)`; both branches use `next_double` and `standard_normal` via the ziggurat). **Critical:** numpy's `standard_normal` uses the ziggurat (`random_gauss_zig`) consuming `next_uint64`; port it exactly. `choice(items, p)`: numpy's `choice` with `p` uses `random_double` to form the cumulative-sum search (`idx = cumsum.searchsorted(random_double(), side='right')`); match the exact draw + search semantics.

- [ ] **Step 4: Run tests** — Expected: PASS, bit-identical dirichlet + choice.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(mcts-rs): dirichlet + choice + standard_gamma/normal bit-exact vs numpy"
```

---

## Phase 3 — TorchScriptEvaluator (Rust)

### Task 4: Single-net batched evaluator + parity vs eager PyTorch

**Files:**
- Create: `catan_mcts_rs/src/evaluator.rs`
- Modify: `catan_mcts_rs/src/lib.rs` (`pub mod evaluator;`)
- Test: `catan_mcts_rs/tests/evaluator_parity.rs`
- Reuse: golden `.bin` + `.ts` from `spike/` (or regenerate via `export_golden.py`).

- [ ] **Step 1: Write failing test (golden vectors from the spike)**

```rust
// catan_mcts_rs/tests/evaluator_parity.rs
// Loads spike/wrapper_traced.ts + spike/g_*.bin and asserts the evaluator's
// (logits, value) == PyTorch reference, max abs diff 0.0, over 50 states.
// (Mirrors spike/rust/main.rs but through the production TorchScriptEvaluator
// API: evaluate(&[Observation]) -> Vec<(Vec<f32>, [f32;4])>.)
```

(Full test body: read N from `spike/g_meta.txt`, build `Observation`s from the bins, call `evaluator.evaluate`, compare to `g_value.bin`/`g_logits.bin` bit-exact.)

- [ ] **Step 2: Run — Expected FAIL** (`evaluator` module missing).

- [ ] **Step 3: Implement `TorchScriptEvaluator`**

```rust
// catan_mcts_rs/src/evaluator.rs
use tch::{CModule, Device, Kind, Tensor};
use catan_engine::observation::Observation;

pub struct TorchScriptEvaluator { module: CModule, device: Device }

impl TorchScriptEvaluator {
    pub fn load(path: &str, device: Device) -> tch::Result<Self> {
        let mut m = CModule::load_on_device(path, device)?;
        m.set_eval();
        Ok(Self { module: m, device })
    }

    /// Batched forward. Returns (policy_logits[280], value[4]) per state,
    /// value left EGO-relative (rotation happens in MCTS, matching Python).
    pub fn evaluate(&self, obs: &[&Observation]) -> Vec<(Vec<f32>, [f32; 4])> {
        // Build [B,19,8] [B,54,13] [B,72,6] [B,59] tensors, ONE forward.
        // The traced wrapper is B=1; batching = stack node dims with a batch
        // vector. Phase-0 traced B=1; extend the wrapper export to accept a
        // batch (see Task 4b) OR loop forward per state for the FIRST cut and
        // optimize to true batch in Task 4b. For parity we loop (still 0.0).
        // ... (impl)
        unimplemented!()
    }
}
```

**Note on batching:** the Phase-0 traced wrapper is B=1. The single-net parity test passes by calling forward per state. TRUE batching (the throughput win) needs a batched trace — that is **Task 4b**, gated by its own parity test. Do not skip 4b; per-state forward defeats the purpose, but it is the correct *first green* for parity.

- [ ] **Step 4: Run — Expected PASS** (0.0 diff).

- [ ] **Step 5: Commit** `feat(mcts-rs): TorchScriptEvaluator loads traced .ts, per-state parity bit-exact`

> **Sequencing update (during execution):** Task 4b (batched trace) is DEFERRED
> into Task 10 (Phase 8). Rationale: batching is a pure-throughput change that
> does not affect any decision/record, so correctness gates (Phases 5-7) are
> proven first with per-state eval, then batching is added and verified to
> produce identical records against the already-trusted baseline. This de-risks
> the order: prove correct, then optimize.

### Task 4b: Batched trace + batched evaluate parity (DEFERRED to Task 10)

**Files:**
- Modify: `mcts_study/catan_gnn/export_torchscript.py` (add `BatchTensorWrapper` + `export_batched`)
- Modify: `catan_mcts_rs/src/evaluator.rs` (true batch path)
- Test: `mcts_study/tests/test_export_torchscript.py` (batched bit-exact), `catan_mcts_rs/tests/evaluator_parity.rs` (batch of 50 == per-state == PyTorch)

- [ ] **Step 1:** Failing Python test: a batched wrapper (forward takes `[B*19,8]…` + batch vectors, or a list) traced once, fed B=50, must equal eager `Batch.from_data_list` bit-exact.
- [ ] **Step 2:** Run — FAIL.
- [ ] **Step 3:** Implement `BatchTensorWrapper` (concatenate node features across graphs + construct the per-type `batch` index vectors from a passed `counts` tensor, or fix B and pad — simplest: variable-B via a `batch` index input). Trace with example B=2 and confirm it generalizes to B=50 (same generalization check as Phase 0). Implement the Rust batched `evaluate` to build these tensors.
- [ ] **Step 4:** Run both tests — Expected PASS (batch == per-state == PyTorch, 0.0).
- [ ] **Step 5:** Commit `feat(mcts-rs): batched TorchScript forward, bit-exact vs per-state and eager`.

---

## Phase 4 — Rust state wrapper (thin) + engine parity

> The engine is already Rust; "state" here is a thin adapter exposing exactly the methods MCTS needs, plus the dual-RNG chance-sampling helper, validated against the Python adapter.

### Task 5: State adapter + engine differential parity vs Python

**Files:**
- Create: `catan_mcts_rs/src/state.rs` (re-exports `catan_engine::Engine` + helpers: `legal_actions_vec`, `chance_sample(rng)`, `observation_current()`, `returns_abs()`, `vp_margin`, `vp_leader_margin`)
- Test: `catan_mcts_rs/tests/engine_parity.rs` AND `mcts_study/tests/test_rust_engine_parity.py`

- [ ] **Step 1: Write the failing differential test (Python side drives both)**

```python
# mcts_study/tests/test_rust_engine_parity.py
"""N random playouts: Rust engine (via catan_mcts_rs debug hook) agrees with
the Python adapter move-by-move on legal_actions, current_player, terminal,
returns, vp."""
import numpy as np
from catan_mcts.adapter import CatanGame
import catan_mcts_rs  # the new module

def test_engine_move_by_move_parity():
    for seed in range(20):
        py = CatanGame().new_initial_state(seed=seed)
        # catan_mcts_rs.debug_playout(seed, action_choices) replays the SAME
        # action indices and returns per-step (legal_actions, cp, terminal).
        rng = np.random.default_rng(seed)
        choices = []
        while not py.is_terminal() and len(choices) < 300:
            if py.is_chance_node():
                outs = py.chance_outcomes(); r = float(rng.random())
                cum, chosen = 0.0, outs[-1][0]
                for v, p in outs:
                    cum += p
                    if r <= cum: chosen = v; break
                choices.append(("chance", int(chosen))); py.apply_action(int(chosen))
            else:
                la = py.legal_actions(); a = la[int(rng.integers(len(la)))]
                choices.append(("step", int(a)))
                # assert Rust sees same legal set BEFORE applying
                assert catan_mcts_rs.debug_legal_actions(seed, choices[:-1]) == [int(x) for x in la]
                py.apply_action(int(a))
        assert catan_mcts_rs.debug_terminal(seed, choices) == py.is_terminal()
```

- [ ] **Step 2: Run — Expected FAIL** (`catan_mcts_rs` not importable / no debug hooks).
- [ ] **Step 3: Implement** `state.rs` helpers + temporary PyO3 `debug_*` hooks in `lib.rs` (replay a choice list on a fresh `Engine`, return the queried value). Build with `maturin develop --release` (env per Phase-0 recipe).
- [ ] **Step 4: Run** both the cargo test (Rust-internal: clone determinism, chance_outcomes sum to 1) and the pytest. Expected PASS.
- [ ] **Step 5: Commit** `feat(mcts-rs): state adapter + engine move-by-move parity vs Python adapter`.

---

## Phase 5 — Rust MCTS search + visit-count/action parity

### Task 6: `mcts.rs` search with golden visit-count + action parity

**Files:**
- Create: `catan_mcts_rs/src/mcts.rs`
- Create: `mcts_study/scripts/dump_mcts_golden.py`
- Test: `mcts_study/tests/test_rust_mcts_parity.py`, `catan_mcts_rs/tests/mcts_internal.rs`

- [ ] **Step 1: Dump Python MCTS golden**

`dump_mcts_golden.py`: build a fixed small net (`GnnModel(32,2)`, fixed seed), export it `.ts`, pick K fixed (seed, depth) states, run `AsyncMcts(evaluator, c=1.4, rng=default_rng(seed)).search(state, n_sims=S)` for S in {8, 32}, dump the resulting `visit_counts[280]` + `best_action` + `last_root_value` to JSON. Use a synchronous wrapper around the batched evaluator (or a trivial direct-eval evaluator) so the golden is deterministic. Include eps>0 (self-play, with dirichlet) AND eps=0 (arena) cases.

- [ ] **Step 2: Write the failing cross-language test**

```python
# mcts_study/tests/test_rust_mcts_parity.py
"""Fixed (seed, net, state): Rust search visit_counts AND chosen action ==
Python async_mcts, bit-exact (spec §5 'Rust MCTS search' row)."""
import json, numpy as np, catan_mcts_rs
from pathlib import Path

def test_visit_counts_and_action_parity():
    golden = json.loads(Path("tests/data/mcts_golden.json").read_text())
    for case in golden:
        vc, action, root_value = catan_mcts_rs.debug_search(
            net_ts=case["net_ts"], seed=case["seed"], history=case["history"],
            n_sims=case["n_sims"], dirichlet_eps=case["eps"],
            dirichlet_alpha=case["alpha"])
        assert list(vc) == case["visit_counts"], f"visit mismatch seed={case['seed']}"
        assert action == case["best_action"]
        assert np.float32(root_value).tobytes() == np.float32(case["root_value"]).tobytes()
```

- [ ] **Step 3: Run — Expected FAIL** (`debug_search` missing).
- [ ] **Step 4: Implement `mcts.rs`** mirroring `async_mcts.py` EXACTLY:
  - `Node { state: Engine, to_play, is_expanded, children: Vec<(u32, Node)>, prior, visit_count, value_sum }`. **Use an insertion-ORDERED children container (Vec or IndexMap), iterating in expansion order**, so UCB ties break to the first-expanded child identically to Python dict order.
  - UCB exactly as the oracle; `select_child` strict `>`.
  - `expand_and_evaluate`: chance loop (sample via `NpRng.random_f64()` over cumulative `chance_outcomes()`), eval via `TorchScriptEvaluator`, **rotate ego→absolute value**, create children in priors order (priors = softmax over legal of `logits[legal]`, computed identically: subtract max, exp, normalize — match numpy float32 ops).
  - Root expand → `apply_root_noise` (NpRng.dirichlet) → root counts 1 visit → `n_sims-1` sims → visit_counts[280].
  - `temperature_sample` (NpRng.choice) + `best_action` (argmax; ties = lowest index, matching `np.argmax`).
  Add `debug_search` PyO3 hook (replays history on a fresh Engine, runs search).
- [ ] **Step 5: Run** — Expected PASS (visit_counts + action + root_value bit-exact for all cases, eps=0 AND eps>0).
- [ ] **Step 6: Commit** `feat(mcts-rs): MCTS search bit-exact (visit counts + action + root value) vs async_mcts`.

---

## Phase 6 — Full self-play path + end-to-end 100-seed gate

### Task 7: `selfplay.rs` `run_selfplay` + per-game GameRecord parity

**Files:**
- Create: `catan_mcts_rs/src/selfplay.rs`
- Modify: `catan_mcts_rs/src/lib.rs` (`run_selfplay` PyO3 entry returning a list of dict-able records)
- Test: `mcts_study/tests/test_rust_selfplay_gate.py`

- [ ] **Step 1: Write the failing end-to-end gate test (NON-NEGOTIABLE, spec §5)**

```python
# mcts_study/tests/test_rust_selfplay_gate.py
"""THE GATE: same 100 seeds through Python AND Rust self-play produce IDENTICAL
game records (length, winner, action_history, visit_counts, root_value)."""
import numpy as np, asyncio
from catan_mcts.adapter import CatanGame
from catan_mcts.async_mcts import play_one_async_game
from catan_gnn.export_torchscript import export
import catan_mcts_rs

def _py_game(seed, ts_eager_evaluator, n_sims):
    # build the Python batched evaluator from the SAME checkpoint; run one game.
    ...

def test_100_seed_selfplay_gate(tmp_path):
    # small net for speed; SAME checkpoint -> .ts for Rust, eager for Python.
    SEEDS = list(range(100)); N_SIMS = 16
    for seed in SEEDS:
        py = _py_game(seed, ...)                      # GameResult
        rs = catan_mcts_rs.run_selfplay(net_ts=..., seeds=[seed],
                                        n_sims=N_SIMS, self_play=True)[0]
        assert rs["length_in_moves"] == py.length_in_moves
        assert rs["winner"] == py.winner
        assert rs["action_history"] == py.action_history
        assert len(rs["moves"]) == len(py.moves)
        for mr, mp in zip(rs["moves"], py.moves):
            assert mr["visit_counts"] == list(mp.visit_counts)
            assert mr["action_taken"] == mp.action_taken
            assert mr["current_player"] == mp.current_player
            assert mr["move_index"] == mp.move_index
            assert np.float32(mr["root_value"]).tobytes() == np.float32(mp.root_value).tobytes()
```

- [ ] **Step 2: Run — Expected FAIL**.
- [ ] **Step 3: Implement `selfplay.rs`** mirroring `play_one_async_game`: ONE `NpRng::from_seed(seed)` per game, chance fast-path, single-legal fast-path, per-player move_index, tau schedule (tau=1 first `temp_moves=30` per-player decisions else 0), record moves, winner = argmax(returns) if max>0 else -1, final_vp from stats, length = total steps, action_history from engine. `run_selfplay` loops seeds, returns Vec of records. (Concurrency/true-batching across games is a Phase-8 perf concern; correctness gate runs games independently — identical records regardless of batching.)
- [ ] **Step 4: Run — Expected PASS** (100/100 identical).
- [ ] **Step 5: Commit** `feat(mcts-rs): run_selfplay + 100-seed end-to-end record parity gate GREEN`.

---

## Phase 7 — Full arena path + end-to-end gate

### Task 8: `arena.rs` `run_arena` + winner/margin/winrate parity

**Files:**
- Create: `catan_mcts_rs/src/arena.rs`
- Modify: `catan_mcts_rs/src/lib.rs` (`run_arena` entry; two-net evaluator)
- Test: `mcts_study/tests/test_rust_arena_gate.py`

- [ ] **Step 1: Write the failing gate test**

```python
# mcts_study/tests/test_rust_arena_gate.py
"""Same seeds: Rust arena per-game (winner_seat, timed_out, vp_margin) ==
Python _play_arena_game; aggregate winrate identical."""
# Build two small nets -> two .ts. For each (rot, seed) in seed_plan:
#   py = asyncio.run(_play_arena_game(...))    -> (winner, timed_out, margin)
#   rs = catan_mcts_rs.run_arena(...)[i]
#   assert equal. Then assert ArenaResult winrate/wins identical.
```

Note the dual-RNG: Rust arena must use a stdlib-Mersenne-equivalent for the GAME-level chance fast-path (`random.Random(seed)`) AND NpRng(seed+11/+13) for the MCTS-internal chance. **Port Python's `random.Random` (Mersenne Twister `random()`) too** — add `MtRng` to `rng.rs` with its own golden test (dump `random.Random(seed).random()` values). This is an extra RNG; add it under Task 8 step 0.

- [ ] **Step 0: Mersenne `random()` parity** — dump `random.Random(777).random()` x5; add `MtRng` golden test; implement MT19937 `random()` (`genrand_res53`). Green before step 1.
- [ ] **Step 2: Run gate — FAIL.**
- [ ] **Step 3: Implement `arena.rs`**: seating, seed_plan, `play_arena_game` (game chance via MtRng, MCTS via NpRng, greedy argmax), vp_leader_margin + vp_margin (read `stats()`/`vp`), winner = returns index else tiebreak. `run_arena` loops the plan, returns per-game records; Python aggregates `ArenaResult` (UNCHANGED — reuse the existing dataclass).
- [ ] **Step 4: Run — Expected PASS** (per-game + winrate identical).
- [ ] **Step 5: Commit** `feat(mcts-rs): run_arena + per-game & winrate parity gate GREEN (incl. MT19937 replica)`.

---

## Phase 8 — Wire into `catan_az` + true cross-game batching

### Task 9: Export in train step + daily.py engine swap (feature-flagged)

**Files:**
- Modify: `mcts_study/catan_az/loop.py` (`_default_train`)
- Modify: `mcts_study/catan_az/daily.py` (self-play + arena stages)
- Modify: `mcts_study/catan_az/config.py` (add `engine: str = "rust"` flag; default keeps Python until Phase 9 validates)
- Test: `mcts_study/tests/test_loop_exports_ts.py`, `mcts_study/tests/test_daily_rust_engine.py`

- [ ] **Step 1:** Failing test: after `_default_train`, a `net.ts` exists next to `checkpoint_best.pt` and loads + matches the checkpoint bit-exact.
- [ ] **Step 2:** Run — FAIL.
- [ ] **Step 3:** Add `export(checkpoint=best, out_ts=best.with_suffix(".ts"), hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers)` to `_default_train`. Add a `cfg.engine` switch in `daily.py`: `"rust"` → call `catan_mcts_rs.run_selfplay` / `run_arena` (feeding records into `SelfPlayRecorder` / `ArenaResult`); `"python"` → the existing async drivers. Keep ladder/journal/window/PUBLISH untouched.
- [ ] **Step 4:** Run — PASS. Also run the full existing `catan_az` test suite to confirm orchestration is intact.
- [ ] **Step 5:** Commit `feat(az): TorchScript export in train; daily self-play+arena Rust engine (flagged)`.

### Task 10: True cross-game batched self-play/arena evaluator

**Files:**
- Modify: `catan_mcts_rs/src/selfplay.rs`, `arena.rs`, `evaluator.rs`

- [ ] **Step 1:** Property test: batched-across-games self-play produces records IDENTICAL to the per-game path (batching must not change decisions — it only changes when forwards happen). Re-run the 100-seed gate with the batched scheduler.
- [ ] **Step 2:** Run — FAIL (batched scheduler not built).
- [ ] **Step 3:** Implement a leaf-batching scheduler: run G games as state machines, collect their pending leaves, one batched `evaluate`, resume. (No asyncio — a simple Rust loop over game slots.) Arena: two evaluators, each batches its own net's leaves to full size (the fix for the 2-evaluator problem). Determinism: each game still consumes its own NpRng in the same order, so records are unchanged.
- [ ] **Step 4:** Run — PASS (records identical to Task 7/8; this is purely a perf change).
- [ ] **Step 5:** Commit `feat(mcts-rs): cross-game leaf batching (throughput) with identical records`.

---

## Phase 9 — Production validation (the spec's step 9)

### Task 11: One iter on the Rust path + Python cross-check + GPU/throughput evidence

**Files:**
- Create: `mcts_study/scripts/validate_rust_path.sh` (runs a small self-play + arena both ways; diffs)

- [ ] **Step 1:** Run a SMALL self-play batch (e.g. 8 seeds, n_sims=200) on BOTH the Python and Rust paths from the SAME checkpoint/.ts; diff the game records (must be identical — this is the live re-confirmation of the gate on the production-size net `GnnModel(128,4)`, not the toy net).
- [ ] **Step 2:** Run a SMALL arena (e.g. 8 games) both ways; diff per-game winner/margin and the aggregate winrate.
- [ ] **Step 3:** Measure: GPU utilization + power (the 28%/4W baseline should rise), per-game wall-clock (should drop well under the 600s/2400s deadlines and finish naturally), and games/hour vs the Python baseline. Record numbers with sources (per `feedback_quantitative_claims_need_sources`).
- [ ] **Step 4:** If all diffs are identical AND throughput/GPU improved, flip `cfg.engine` default to `"rust"`. Document the validation in a journal under `docs/superpowers/journals/`.
- [ ] **Step 5:** Commit `feat(az): validate Rust path bit-exact on production net; default engine=rust`. Do NOT resume the paused AZ loop or merge to main without user approval.

---

## Cross-cutting requirements (apply to every Rust task)

- **TDD red/green per step.** Failing test first, then minimal impl.
- **Double verification per unit** (spec §5): golden-parity (oracle values embedded) AND differential/property (N seeds/states agree). Both required; neither alone.
- **Bit-exact for decisions; FP-identity for the net forward** (Phase-0 proved 0.0 on CPU).
- **Rebuild discipline:** after any Rust change touching the PyO3 surface, `maturin develop --release` from the worktree (env: `source ~/.cargo/env`, activate venv, `LIBTORCH_USE_PYTORCH=1`, `LIBTORCH_BYPASS_VERSION_CHECK=1`). WSL may need `wsl.exe --shutdown` on a getpwnam error. A green pytest without rebuild is a false negative.
- **WSL nested-quoting:** put multi-step shell logic in `.sh` files and run the file (never inline `$()`/loops through `wsl.exe -- bash -lc`).
- **Build artifacts off /mnt/c:** `CARGO_TARGET_DIR=/home/chitii/...`.
- **No deleting any file without explicit per-action authorization** (`feedback_no_delete_without_permission`).
- **Commit at each green checkpoint; push branches to origin freely; ask before merging to main.**

## Self-review notes (spec coverage)

- §3 architecture (Rust MCTS, tch-rs GNN, TorchScript seam, training unchanged) → Phases 1,3,5,6,7,8.
- §4.1 `catan_mcts_rs` → Phases 2-8. §4.2 `TorchScriptEvaluator` → Phase 3. §4.3 `export_torchscript.py` → Phase 1 + Task 9. §4.4 thin PyO3 → Tasks 7,8,9. §4.5 training unchanged → only Task 9's export call added.
- §5 verification: Phase-0 ✔ (done); RNG bit-exact → Phase 2 (+ MT19937 in Task 8); per-component double verification → every task; end-to-end 100-seed gate → Task 7 (self-play) + Task 8 (arena).
- §5 value-perspective rotation → Task 6 (explicit).
- §6 rollout steps 1-9 → Phases 0-9 one-to-one.
- §7 non-goals respected (no orchestration change beyond the export call + flagged engine swap; no net architecture change — the traced wrapper avoided the fixed-topology rewrite).
