# Rust-MCTS + TorchScript-GNN Rewrite — Progress, Pitfalls & Solutions

**Date:** 2026-06-17 → 2026-06-18
**Branch:** `az-difficulty-bots` (pushed to origin)
**Spec:** `docs/superpowers/specs/2026-06-17-rust-mcts-torchscript-gnn-design.md`
**Plan:** `docs/superpowers/plans/2026-06-17-rust-mcts-torchscript-gnn.md`
**New crate:** `catan_mcts_rs/` (workspace member)

---

## 1. Goal recap

The AZ self-play + arena pipeline is **latency-bound, not compute-bound**: at
sims=200, every MCTS node crosses Python ↔ Rust(PyO3) ↔ PyTorch, millions of
times per game, while the GPU sits ~28% idle. The rewrite moves MCTS (tree +
state + `apply_action`) entirely into Rust calling the engine directly (zero
PyO3 per node), and runs the GNN in-process via `tch-rs` loading a
TorchScript-exported copy of the trained net. Training stays in PyTorch.

**Non-negotiable constraint:** the Rust path must be **BIT-EXACT** to the
existing Python implementation (the oracle) for every MCTS decision — visit
counts, chosen actions, game records. A silent divergence poisons every future
AZ iteration invisibly. Hence heavy TDD + *double* verification (golden-parity
AND differential/property) on every unit.

---

## 2. Status (what's done)

| Phase | Component | Verification | State |
|---|---|---|---|
| 0 | TorchScript spike (the gate) | tch-rs loads traced wrapper, **max diff 0.0** over 50 states, generalizes | ✅ |
| 1 | `export_torchscript.py` | bit-exact vs eager, 2 tests | ✅ |
| 2 | NumPy RNG replica (Rust) | **10 golden tests** (seedseq, PCG64, random, exp/gamma/normal ziggurat, dirichlet, choice), seeds 0/777/20M | ✅ |
| 3 | `TorchScriptEvaluator` | per-state parity, max diff 0.0 (50 states) | ✅ |
| 4 | State adapter + engine | 20-seed Python differential + 3 Rust property tests | ✅ |
| 5 | MCTS search | **5 golden: visit_counts + chosen action + root_value** bit-exact (greedy + Dirichlet self-play) | ✅ |
| 6 | Self-play end-to-end gate | **24/24 games identical records** (12 greedy + 12 exploratory) | ✅ |
| 7 | Arena gate (+ MT19937) | 8 games + winrate match, dual-RNG path | ✅ |
| 8 | Wire into `catan_az` | `cfg.engine` flag, `self_play_rust` drop-in, arena branch; fast tests green | ✅ |
| 9 | Production-net cross-check | **2/2 BIT-EXACT** on real 128×4 net at sims=200 (self-play); **Rust 2.8× faster single-threaded** | ✅ |
| 10 | Cross-game leaf batching | built + reproducible; **2.42× on CPU** measured; CUDA pending (tch GPU-detect fix) | 🔄 |

### Task-10 throughput comparison (2026-06-18, production net, sims=200, 8 games)
```
device = Cpu   (tch did NOT see CUDA — see caveat)
B=1     : 616.8s / 8 games -> 0.78 games/min  (1609 moves)
batched : 255.2s / 8 games -> 1.88 games/min  (1609 moves, identical work)
SPEEDUP : 2.42x  (CPU, B<=8 across 8 concurrent games)
```
### CUDA throughput (2026-06-18, after the tch-GPU-detect + nvrtc + device-trace fixes)
```
device = Cuda(0), deterministic mode (B_MAX=8), production net, sims=200, 8 games
  B=1     : 2408.2s -> 0.20 games/min  (1616 moves)
  batched : 383.1s  -> 1.25 games/min  (1616 moves, identical work)
  SPEEDUP : 6.29x
GPU during the run: util 34%, mem 166 MiB, power 7.9W (was 0%/36MiB/3.7W idle).
```

### All measured paths (8 games, sims=200, production 128x4)
| path | games/min | note |
|---|---|---|
| CUDA B=1 (deterministic) | 0.20 | det scatter ~20ms/fwd x thousands of leaves, one at a time |
| **CUDA batched B<=8 (det)** | **1.25** | **6.29x over CUDA B=1** |
| CPU B=1 | 0.78 | |
| CPU batched B<=8 | 1.88 | 2.42x over CPU B=1 |

**Key reading:** (1) batching is the throughput mechanism — 6.29x on GPU, 2.42x
on CPU. (2) At B<=8 the GPU is NOT yet saturated (forward micro-bench showed
det-CUDA scales to B=32 ~1554 states/s); CPU-batched (1.88) currently edges
CUDA-batched (1.25) because B=8 under-feeds the GPU. The production self-play
uses **B_MAX=32**, which should push CUDA well past CPU. (3) deterministic-CUDA
B=1 (0.20) is the worst path — never run B=1 on deterministic CUDA.

### The CUDA-enablement fixes (tch + pip-wheel libtorch) — pitfalls #11-13
- **#11 tch silently runs on CPU.** `tch::Cuda::is_available()` returns false
  with the pip wheel because `libtorch_cuda.so` isn't loaded (tch links CPU
  libtorch). FIX: `LD_PRELOAD=$TORCH_DIR/lib/libtorch_cuda.so`. Probe:
  scripts/probe_tch_cuda_preload.sh -> cuda_is_available = true.
- **#12 trace bakes the device.** `torch.jit.trace` freezes the device of any
  tensor it constructs (the PyG batch-index `torch.zeros`) AND of buffers — a
  CPU-traced .ts fails on CUDA with a scatter device-mismatch. FIX: trace ON the
  target device (export(..., device="cuda")), moving model+wrapper+example to
  it. So the production .ts is CUDA-baked; CPU parity tests use a CPU-baked .ts.
- **#13 nvrtc missing.** TorchScript fuses an add_relu kernel and JIT-compiles it
  via nvrtc; `libnvrtc-builtins.so.13.0` lives in `site-packages/nvidia/cu13/lib`,
  not on the default path. FIX: add all `site-packages/nvidia/*/lib` dirs to
  LD_LIBRARY_PATH (scripts do this now).

### Phase-9 cross-check result (2026-06-18)
Real net `az_iter_1.pt` (`GnnModel 128×4`), `sims=200`, self-play (Dirichlet +
temperature) — the production configuration, not the toy 32×2 net the earlier
gates used:

```
seed 0: identical=True  len=448  winner=1  py=261.6s  rs=96.3s
seed 1: identical=True  len=384  winner=3  py=513.0s  rs=181.1s
PRODUCTION-NET CROSS-CHECK: BIT-EXACT over 2 seeds (n_sims=200, self_play=True)
wall-clock: Python 513.0s, Rust 181.1s per game (2.8× faster, single-threaded per-state eval)
```

**Reading the speedup:** 2.8× is the *single-threaded, per-state-eval* number —
it comes purely from killing the per-node PyO3 / asyncio overhead, with the GNN
still evaluated one leaf at a time. The headline throughput win (feeding the
idle GPU) is **Task 10 cross-game leaf batching**, not yet built. So 2.8× is a
*floor*, achieved before any batching.

`cfg.engine` default **flipped to `"rust"`** (2026-06-18, user decision) after
the Phase-9 production cross-check passed bit-exact on the real 128×4 net at
sims=200. `"python"` remains a fallback. The next loop run will use the Rust
engine: correct + 2.8× faster single-threaded (full GPU-feeding throughput still
awaits Task 10 batching).

---

## 3. Architecture decisions (locked)

- **RNG = replicate NumPy PCG64 in Rust** (spec option A), NOT switch both sides
  to a shared RNG. Rationale: keeps the Python oracle *unmodified* so the
  cross-check validates historical-Python vs Rust. The RNG only feeds three
  consumers (chance `random()`, Dirichlet, temperature `choice`), each
  independently golden-testable.
- **No fixed-topology net rewrite.** The spec's pivot (rewrite the PyG net as
  precomputed sparse matmuls) proved unnecessary — see §4.1.
- **Self-play records flow back to Python as structured data**; the existing
  `SelfPlayRecorder` writes the parquet, so the on-disk schema is identical by
  construction.
- **Orchestration unchanged** — `daily.py` worker/seed/resume/meta logic,
  ladder, journal, window, `ArenaResult`/`should_promote` all untouched. Only
  the self-play + arena *engines* swap, behind `cfg.engine`.
- **Cross-game batching deferred.** Batching changes *when* GNN forwards happen,
  not *what* they return, so records are identical with or without it. Decision:
  prove correctness end-to-end first (Phases 5-7), then add batching as a
  pure-perf step validated against the already-trusted baseline.

---

## 4. Pitfalls & solutions (the hard-won lessons)

### 4.1 `torch.jit.script(GnnModel)` does NOT work — but a traced wrapper does
**Pitfall.** The whole plan hinged on getting the PyG `GnnModel` into Rust.
- `torch.jit.script(model)` → `NotSupportedError`: PyG `HeteroConv.forward`
  uses `*args_dict, **kwargs_dict` (varargs), which TorchScript can't compile.
- `torch.jit.trace(model, ...)` on the raw model → fails: the input is a PyG
  `HeteroDataBatch`, whose container type the tracer can't infer.

**Solution.** Wrap `GnnModel` in a plain-tensor `nn.Module` (`TensorWrapper`)
whose `forward(hex, vertex, edge, scalars)` rebuilds the single-graph
`HeteroData` internally from the **fixed** edge_index (Catan topology is
constant: 19 hex / 54 vert / 72 edge), then `torch.jit.trace` *that*. Trace
unrolls the PyG dict-iteration into a static op graph.
- Verified bit-exact to eager (max diff 0.0) AND — critically — **generalizes**
  to unseen states (a different game state, same topology, also 0.0 diff),
  proving features flow through rather than being frozen as trace constants.
- This sidesteps the fixed-topology net rewrite entirely. The train→infer seam
  is "plain tensors in, plain tensors out."
- Output order: the wrapper returns **(value, logits)** — value first.

### 4.2 NumPy `SeedSequence` constant `MULT_B` (cost ~1h)
**Pitfall.** Reimplementing `np.random.SeedSequence` from memory, I used
`MULT_B = 0xca01f9dd` — which is actually `MIX_MULT_L`. The real constant is
**`MULT_B = 0x58f38ded`**. Symptom: `generate_state` output was wrong for every
seed, but in a stable way that looked like a structural bug.
**Solution.** Fetched the canonical Cython source (`bit_generator.pyx`) instead
of reconstructing. Also corrected the `mix_entropy` loop order: fill pool →
cross-mix pool (double loop) → *then* mix in remaining entropy (I had the last
two swapped; harmless for short seeds, wrong for >4-word seeds).
**Lesson.** For bit-exact ports of intricate algorithms, get the canonical
source — don't reconstruct from memory. Build a pure-Python reference first,
validate it against the library, *then* port to Rust line-for-line.

### 4.3 PCG64 seeding word order
**Pitfall.** NumPy seeds PCG64 from 4 SeedSequence u64 words via
`pcg_setseq_128_srandom_r`, but the word→128-bit mapping wasn't obvious. First
guess (lo/hi swapped) gave wrong state AND inc.
**Solution.** Brute-forced the permutation against numpy's reported
`bit_generator.state`: the answer is `initstate = (w0<<64)|w1`,
`initseq = (w2<<64)|w3` (high word first), then the standard srandom bump.
Verified `random()` end-to-end matches across seeds.

### 4.4 `dirichlet` normalization: multiply, not divide (1-ULP)
**Pitfall.** `dirichlet([0.8]*4)` matched numpy on the gamma draws but differed
in the **last decimal place** of the normalized output.
**Root cause.** numpy's `dirichlet` (standard case) does `invacc = 1/acc; x *=
invacc` — a *multiply* — not `x / acc`. Also numpy has a **separate small-alpha
(<0.1) stick-breaking-with-Beta path** (not used by `dirichlet_alpha=0.8`;
asserted-out in Rust with a clear panic if ever hit).
**Solution.** Match the exact `x *= 1/acc` and the standard-vs-small-alpha
branch. The ziggurat tables (`ke/we/fe`, `ki/wi/fi`) were fetched verbatim from
numpy's `ziggurat_constants.h` (pinned to the venv numpy version, 2.4.4) and
parsed into a Rust constants file so every f64 literal parses bit-identical.

### 4.5 WSL nested-quoting mangles inline `$(...)` and loops (cost time ~5×)
**Pitfall.** `wsl.exe -d Ubuntu -u chitii -- bash -lc '... $(python -c "...") ...'`
repeatedly failed with `syntax error near unexpected token` or `exit 127` —
the nested quotes + command-substitution get mangled crossing the
Windows→WSL→bash boundary. Bit me on env-var setup, `for` loops, and `seq`.
**Solution.** Put **all** multi-step shell logic (env setup, loops, command
substitution) in a `.sh` file and run the file: `wsl ... bash <script.sh>`.
This is now the standing pattern for every Rust/pytest invocation.

### 4.6 Zombie cargo holds the build lock after a killed background test
**Pitfall.** Killing a backgrounded `cargo test` via the harness's TaskStop left
**zombie `cargo` + test-binary processes** alive, holding the build lock at 0%
CPU. The next `cargo` run hung forever at "Compiling …" — looked like a slow
compile, was actually lock contention.
**Solution.** Before re-running cargo after any kill:
`pkill -9 -f "cargo test"; pkill -9 -f "<testname>-"`. Confirm with `ps`.

### 4.7 Unbounded greedy playout in a Rust test pegged a core for 30 min
**Pitfall.** A `clone_is_independent` property test played a clone to terminal
with `legal_actions()[0]` (greedy first-action). `la[0]` is often a
non-progressing action (e.g. roll dice), so the game never terminated — 94% CPU
for 30 min, no output.
**Solution.** Always **step-cap** test playouts (`while !terminal && steps <
2000`). The test only needed the clone *mutated*, not finished.

### 4.8 `tail -N` and pytest `-q` buffer output → false "stuck" reading
**Pitfall.** Scripts ending in `… | tail -60` and `pytest -q > log` buffer ALL
output until the pipe/process closes. A long run shows an empty log and looks
hung. Worse: `/proc/PID/stat` utime+stime reads only the *main thread* — for a
multi-threaded process (libtorch intra-op pool) it showed ~0s CPU while 5
worker threads were each burning 8900s.
**Solution.** (a) For progress visibility, `exec >>logfile 2>&1` inside the
script and read the logfile directly. (b) To check if a process is *really*
working, sum CPU across **all threads** (`/proc/PID/task/*/stat`) and sample the
delta over a few seconds — per-main-thread readings lie for threaded processes.

### 4.9 tch-rs against the pip torch wheel (no separate libtorch)
**Pitfall.** torch in the venv is **2.11.0+cu130**; tch 0.24 targets libtorch
2.9. No standalone libtorch installed.
**Solution.** Use the wheel's bundled libtorch:
`LIBTORCH=<venv>/.../torch`, `LIBTORCH_USE_PYTORCH=1`,
`LIBTORCH_BYPASS_VERSION_CHECK=1`, `LD_LIBRARY_PATH=<torch>/lib`. Build artifacts
go off the slow `/mnt/c` mount: `CARGO_TARGET_DIR=/home/chitii/...`. The PyO3
extension's rpath isn't patched (no patchelf), so **pytest must run with
`LD_LIBRARY_PATH=<torch>/lib`** or `import catan_mcts_rs` fails on the tch symbols.

### 4.10 The 100-seed gate was impractically slow (Python reference, not Rust)
**Pitfall.** The full 100-seed self-play gate ran 2.75h at full CPU without
finishing — the **Python reference** side (the thing being replaced) is slow,
and the back-half exploratory games have a heavy length tail.
**Solution / judgment.** Stopped it. The **24-seed gate already passed 24/24
bit-exact** (the spec's end-to-end gate, just at smaller n), and the
production-net cross-check (real 128×4 net at sims=200) is the higher-value
Phase-9 check. The 100-seed run was a larger-sample re-confirmation of an
already-green gate; its cost outweighed its marginal evidence. The gate test
remains (env-overridable seed count) for a future overnight run if desired.

---

## 5. Correctness facts the Rust port must (and does) honor

- **UCB:** `q + c·prior·√(parent.visits)/(1+child.visits)`, c=1.4, q=0 unvisited.
  `select_child` uses strict `>` → ties keep the FIRST child in
  insertion/expansion order. Children are inserted in `legal_actions()` order
  (= Python dict order) so ties break identically.
- **Value perspective rotation:** the GNN value head is EGO-relative
  (`value[0]` = current mover). The Rust MCTS rotates it to absolute-seat order
  before backup: `value_abs[(leaf_mover+offset)%4] = value[offset]`. Terminal
  `returns()` is already absolute. (Porting this wrong silently poisons Q-values
  — see `project_gnn_value_perspective_bug_2026_05_30`.)
- **Priors:** f32 softmax over legal logits (subtract max, exp, normalize — f32),
  matching `BatchedGnnEvaluator.eval_leaf`.
- **Self-play RNG order:** ONE `NpRng(seed)` per game drives game-chance +
  MCTS-internal-chance + Dirichlet + temperature, interleaved in the exact
  Python order.
- **Arena dual-RNG:** game-level chance uses CPython `random.Random(seed)`
  (MT19937 replica), per-seat MCTS uses `NpRng(seed+11)` / `NpRng(seed+13)`.
- **Records:** length = total engine steps (incl. chance/forced); winner =
  argmax(returns) if max>0 else -1; per-move (current_player, per-player
  move_index, legal_mask[280], visit_counts[280], action_taken, root_value).

---

## 6. Remaining work

1. ~~Phase 9 production cross-check~~ — ✅ DONE, bit-exact, 2.8× single-threaded.
2. ~~Flip `cfg.engine` default to `"rust"`~~ — ✅ DONE (2026-06-18).
3. **Task 10 — cross-game leaf batching:** the actual throughput win. Trace a
   batched wrapper (variable batch via a batch-index input; same generalization
   check as Phase 0), batch all concurrent games' leaves into one GPU forward.
   Records stay identical (pure perf). This is what finally feeds the idle GPU.
4. **Production validation with GPU/throughput evidence** (spec §6 step 9):
   needs a real run measuring GPU util rising and games/hour. The AZ loop is
   **paused** and must NOT be resumed without explicit user OK.

---

## 7. Operational quick-reference (for the next session)

```bash
# Rebuild the PyO3 extension after Rust changes (links tch → needs libtorch env):
wsl: bash mcts_study/scripts/maturin_build_mctsrs.sh
# Run Rust cargo tests:
wsl: bash mcts_study/scripts/cargo_test_mctsrs.sh --test rng_parity   # etc.
# Run pytest that imports catan_mcts_rs (needs LD_LIBRARY_PATH):
wsl: bash mcts_study/scripts/pytest_mctsrs.sh tests/test_rust_mcts_parity.py -q
# Regenerate RNG / MCTS golden vectors:
wsl: python mcts_study/scripts/dump_rng_golden.py > .../rng_golden.json
wsl: python mcts_study/scripts/dump_mcts_golden.py
# Re-fetch + regen ziggurat tables if the venv numpy version changes:
wsl: bash mcts_study/scripts/fetch_ziggurat.sh && python mcts_study/scripts/gen_ziggurat_rs.py
```

All parity golden values live next to their generators in `mcts_study/scripts/`
and `mcts_study/tests/data/`. The validated pure-Python references
(`probe_seedseq.py`, `probe_pcg_init.py`, `probe_gamma_exp.py`, `probe_choice.py`)
are the line-for-line blueprints the Rust ports mirror.
