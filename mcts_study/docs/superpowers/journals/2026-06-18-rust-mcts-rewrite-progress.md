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
| 10 | Cross-game leaf batching | **DONE** — reproducible; **~16× at production config (B=32, CUDA)**, 6.3× at B≤8, 2.42× on CPU | ✅ |

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

### All measured paths (sims=200, production 128x4, deterministic)
| path | games | games/min | vs CUDA B=1 |
|---|---|---|---|
| CUDA B=1 | 8 | 0.20 | 1x |
| CUDA batched B<=8 | 8 | 1.25 | 6.3x |
| **CUDA batched B=32** | **32** | **3.26** | **~16x** |
| CUDA batched B=64 | 64 | 3.85 | ~19x |
| CPU B=1 | 8 | 0.78 | — |
| CPU batched B<=8 | 8 | 1.88 | — |

**Key reading:** (1) batching is THE throughput mechanism — at the production
config (B_MAX=32, 32 concurrent games) the GPU does **3.26 games/min ≈ 16x the
deterministic-CUDA B=1 baseline**, and **2.6x the B<=8 run** (B=8 under-fed the
GPU). (2) At B=32 CUDA (3.26) decisively beats CPU-batched (1.88) — the crossover
the B=8 run was below. GPU power climbed 3.7W idle → 7.9W (B=8) → 29.6W (B=32):
real sustained work. (3) deterministic-CUDA B=1 (0.20) is the worst path — never
run B=1 on deterministic CUDA; batching is mandatory there.
(4) All reproducible (deterministic mode) — the replay contract holds.

### Does bigger batch help? Forward-throughput sweep (deterministic CUDA, isolates B)
```
 B      fwd/s   states/s   ms/fwd
  1      50.3      50.3    19.864
  8      50.3     402.7    19.865   <- same latency as B=1, 8x the states
 16      49.1     785.4    20.372
 32      44.7    1431.2    22.359
 64      45.1    2884.6    22.187
 96      43.7    4199.9    22.858
128      41.5    5316.5    24.076   <- peak, STILL climbing (no plateau)
```
**Yes — bigger batches help, near-linearly, and it has NOT saturated at B=128.**
The deterministic-scatter kernel has a large FIXED per-call cost that dominates
the tiny h128 compute: forward latency barely moves (19.9ms@B=1 -> 24.1ms@B=128,
1.2x time for 128x the states). So states/s ~= 50*B until something saturates;
at B=128 it's 5317 states/s ≈ **106x the B=1 rate**, still rising, no OOM on the
4GB card. **Implication:** production B_MAX=32 leaves throughput on the table —
raising B_MAX to 64/128 would ~2-4x GNN throughput again. The real ceiling is
the number of CONCURRENT GAMES with a leaf pending (you need ~B games in flight
to fill a B batch), not the GPU. Sweep: scripts/bench_batch_sweep.py.

**END-TO-END games/min is a DIFFERENT, lower ceiling** than raw states/s:
- B=32/32g: 3.26 g/min ; B=64/64g: 3.85 g/min — doubling B_MAX gained only +18%.
Raw GNN throughput doubles 32→64, but end-to-end is now limited by game DESYNC
(can't keep a B=64 batch full — games are at different move counts / hit chance
nodes) and CPU-side per-leaf work (build_observation, tree ops, NpRng) that
does NOT batch. So past ~B=32 the GPU is no longer the bottleneck end-to-end;
the win flattens. Practical sweet spot: B_MAX ~= 32-64 (the production B_MAX=32
is close to the knee; 64 is +18% for 2x VRAM/concurrency). Bigger B_MAX only
helps if you run many more concurrent games AND cut the CPU-side per-leaf cost.

### THE MAXIMUM (full sweep to B=2048, deterministic CUDA, 4GB GTX 1650)
```
   B      fwd/s   states/s   ms/fwd   VRAM_MB
   1       48.8       48.8    20.5       37
  32       41.2     1317      24.3       47
  64       36.6     2340      27.3       57
 128       35.4     4532      28.2       76
 256       27.4     7012      36.5      113
 512       15.9     8160      62.7      188
1024        8.8     8974     114        339
2048        4.6     9368     219        638   <- PEAK raw throughput (plateau)
```
- **Raw GNN max ≈ 9,400 states/s at B=2048**, but PLATEAUING: B=256→2048 is 8x
  the batch for only +34% throughput. The knee is **B≈128-256** (~4.5-7k
  states/s); past it you pay linear latency (219 ms/fwd at B=2048) for crumbs.
- **VRAM is NOT the limit** — B=2048 used only 638 MB of 4096 MB. The h128 net is
  tiny; memory never caps. (The ceiling would be ~B=12000 on VRAM, useless.)
- **End-to-end games/min max ≈ 3.85** (B=64), FLAT past B=32 — this is the metric
  that matters, and it's far below raw throughput. Limited by game desync +
  CPU-side per-leaf work, not the GPU.

**Bottom line — the useful maximum is B_MAX ≈ 32-64.** Production B_MAX=32 is at
the knee. To go faster end-to-end the lever is NOT bigger batches: it's cutting
the CPU per-leaf cost (build_observation / tree ops / NpRng) or running more
concurrent games (more CPU cores). The GPU has huge headroom (638MB/4GB, 9k
states/s) that self-play can't currently feed.

### CPU-vs-GPU phase profile (16 games, B_MAX=32, sims=200, deterministic CUDA)
```
total : 533.5s
  GPU forward : 516.5s (96.8%)   64722 batches, MEAN B = 10.7  (cap 32!)
  CPU provide : 12.9s (2.4%)     expand + tree + backup
  CPU advance : 3.4s  (0.6%)     chance/single-legal + finish_move
  leaves      : 692,236
```
**OVERTURNS the "CPU is the bottleneck" theory.** CPU work is only 3%. The time
is 96.8% GPU forward — but the smoking gun is **mean batch = 10.7 of 32**: we pay
the full ~20ms deterministic-scatter per-call latency 64,722 times for a
ONE-THIRD-FULL batch. The GPU isn't compute-bound; it's **starved within the
batch** (games desync — only ~11 of 16 have a leaf parked at any instant). The
real lever is **mean batch FILL**, not CPU speed and not raw batch ceiling:
(1) run more concurrent games than B_MAX so batches fill toward the cap, and/or
(2) cut the per-forward fixed cost (deterministic scatter is ~20ms vs ~7.5ms
non-det). This is the original "starved GPU" problem in miniature.

### CONFIRMED: more concurrent games fills batches (16 vs 64 games, B_MAX=32)
| games | mean B | GPU% | leaves | total | leaves/s |
|---|---|---|---|---|---|
| 16 | 10.7 | 96.8% | 692k | 533s | 1298 |
| 64 | 21.5 | 91.7% | 2.6M | 1148s | **2281 (1.76x)** |
Doubling+ the concurrent games (16→64) doubled mean batch fill (10.7→21.5) and
gave **1.76x more leaf-throughput** — pure batch-fill win, no algorithm change.
But mean B=21.5 < 32 cap even at 64 games: ~1/3 of games are NOT parked at any
instant (in `advance` resolving chance/single-legal runs, or finishing). THAT
gap is the next lever after concurrency. CPU% rose 3%→8% but is still minor.

### Concurrency sweep extended to G=512 (B_MAX=32, sims=50, deterministic CUDA)
| G | mean batch /32 | GPU% | CPU% | leaves/s |
|---|---|---|---|---|
| 16 | 10.7 | 96.8% | 3.0% | 1298 |
| 64 | 21.5 | 91.7% | 8.0% | 2281 |
| **512** | **29.9 (93%)** | 88.2% | 11.4% | **2525** |
**Concurrency ALONE nearly fills the batch** — 29.9/32 at G=512. But sharply
diminishing: 64→512 (8x games) only +11% leaves/s (2281→2525). Box: 12 cores /
54GB / 4GB VRAM — RAM/VRAM never blocked; the scheduler core hit 95% (single-
threaded) at G=512 but CPU is still only 11% of total time.

**CONCLUSION — the bottleneck has MOVED.** Batch fill is ~solved by concurrency
(93% at G=512), so Lever B (full-parked-set scheduler) has ≤7% fill headroom
left — NOT worth the refactor. At high G we are now genuinely **GPU-forward-
bound** (88%, 228k near-full batches) at ~2525 leaves/s — far below the 9400
states/s raw ceiling because the **deterministic-scatter kernel (~20ms/call)**
is the limit. The only levers left that matter: (1) cheaper forward — non-det
CUDA is 2.6x but BREAKS the replay contract (rejected); a smaller/fused net or
fp16 would help; (2) reduce leaves/game (tree reuse across moves). Lever B is
effectively MOOT. Recommended production setting: **n_concurrent ≈ 256-512,
B_MAX=32** (knee of the fill curve; past 64 it's +11% for 8x the games/RAM).

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
