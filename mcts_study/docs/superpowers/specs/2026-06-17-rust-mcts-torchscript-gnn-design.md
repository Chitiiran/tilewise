# Rust-MCTS + TorchScript-GNN Architecture — Design Spec

**Date:** 2026-06-17
**Status:** Approved design, ready for implementation planning
**Author handoff:** This spec is written for an agent with NO prior context from the design conversation. Read it fully before planning.

---

## 1. Problem Statement

The AlphaZero self-play + arena pipeline is **latency-bound, not compute-bound**. Measured during live runs on the GTX 1650 (4 GB):

- GPU utilization **28%**, power draw **4.2 W of 75 W (~6%)**, VRAM **134 / 4096 MiB (3%)**, CPU load **0.85 / 12 cores**. Nothing is saturated.
- Each MCTS game spends its wall-clock **waiting on round-trips**: Python coroutine → batch queue → tiny GPU forward (microseconds) → back through PyO3 → engine `apply_action`. At sims=200 and ~170–575 moves/game, that is **millions of Python↔Rust↔PyTorch boundary crossings per game**.
- The GPU is **starved**: forwards finish in microseconds, then it idles waiting for the next batch of leaves to arrive through the Python/asyncio/PyO3 path.

### Measured evidence (concurrency sweep, `measure_arena_concurrency.py`, cand=iter_7 vs champ=az_iter_1, sims=200, no per-game deadline)

| n_concurrent | result |
|---|---|
| 16 | **0 games** finished in 11.6 min (matches the production arena's 100% timeout) |
| 8 | 8/8 finish naturally, per-game **658→1491 s, median 1200 s**, 8/8 over the 600 s cap, 0.3 games/min. Times **climb across the run** (phase-drift desync). |
| 4 | 586, 1062, 1111 s (partial) — still mostly >1000 s |

**Conclusion: tuning concurrency cannot fix this.** Even c=4 produces 1000 s+ games. The 3 "obvious" fixes all fail:
- More evaluators → GPU already idle; more pipes fragment batching further.
- Lower concurrency → games still exceed the cap; phase-drift dominates.
- Raise the cap alone → works but each 300-game arena takes many hours on a structurally-starved pipeline.

The arena is **worse** than self-play because it runs **two evaluators** (candidate net + champion net); their leaves cannot batch together, so each net sees only ~half the concurrent games' leaves (mean_batch ~9 vs self-play ~16).

### Why it is not O(c)

Ideal would be `O(total_turns / c)` with perfect batching (the user's intuition). It degrades because: (1) game **length** varies 170–575+ turns (`max_steps` is a cap, not a fixed count); (2) **phase drift** — concurrent games desync via chance nodes and forced moves, so their leaves don't arrive together and batches thin out; (3) the **two-evaluator arena** halves achievable batching. Effective parallelism is ~c/2 at best and degrades toward serial as games desync.

---

## 2. Goal

**Both** maximum throughput (games/hour) **and** low per-game latency, as a **scalable foundation** (would benefit from a bigger GPU or more cores later). Not a quick config patch — a structural fix.

---

## 3. Chosen Architecture

Move **MCTS (tree, UCB selection, expansion, state management, `apply_action`) entirely into Rust**, calling the existing Rust engine directly with **zero PyO3 crossings per node**. The GNN runs via **`tch-rs` (libtorch)** loading a **TorchScript-exported** copy of the trained net. **Training stays in PyTorch, unchanged.**

Per iteration: train in PyTorch → `torch.jit.script` the net → Rust self-play/arena load the scripted module and run **fully in-Rust** (zero Python in the hot loop). The millions of per-node crossings collapse to **zero hot-loop crossings**; MCTS calls the net in-process via tch, batching all concurrent games' leaves into full GPU forwards. The idle GPU finally gets fed.

### Decisions locked during design

| Axis | Decision | Rationale |
|---|---|---|
| Where MCTS lives | **Rust** (tree + state + apply_action) | Kills the per-node PyO3 cost; the dominant latency |
| Where GNN inference lives | **Rust via tch-rs** (Python out of the inference loop) | Fastest, cleanest hot loop; no per-leaf cross-process call |
| Where GNN **training** lives | **PyTorch, unchanged** | Net stays trainable; we only move *inference* |
| Train→infer seam | **TorchScript export** (`torch.jit.script(GnnModel)`) | Rust runs the *identical traced graph* — parity by construction, no hand-port |
| Net parity guarantee | TorchScript (not hand-ported tch net) | Sidesteps reimplementing PyG message-passing in Rust |
| PyG→TorchScript risk | **Spike first (Phase 0)** before committing | The whole plan hinges on PyG scripting cleanly |
| Cross-language parity bar | **BIT-EXACT** (not statistical) | MCTS is deterministic given (seed, net, RNG); no reason to accept drift; bit-exact surfaces divergence loudly |
| Orchestration layer | **UNCHANGED** (daily.py, ladder, journal, window logic) | We swap the self-play + arena *engine* underneath, not the loop |

---

## 4. Components

Five units with clean interfaces. **The Python AZ orchestration layer (`catan_az.daily`, ladder, journal, window selection — all recently fixed) is untouched.** Only the self-play and arena *engines* swap from Python-async to Rust.

### 4.1 `catan_mcts_rs` — Rust MCTS crate (NEW)
Owns the search tree, UCB selection, node expansion, state management — calling the existing Rust engine directly (no PyO3 per node).
- Public surface (coarse PyO3, one crossing per **stage**):
  - `run_selfplay(net_path, n_games, sims, config, seed_base) -> Vec<GameRecord>`
  - `run_arena(net_a_path, net_b_path, n_games, sims, seed_base, seating) -> Vec<ArenaResult>`
- Depends on: the engine crate + `TorchScriptEvaluator` (4.2).
- Internally mirrors the existing Python `async_mcts.py` algorithm: chance fast-path, single-legal fast-path, UCB with c=1.4, per-player move-index, temperature sampling for self-play (Dirichlet root noise + tau schedule), argmax for arena.

### 4.2 `TorchScriptEvaluator` — the net, in Rust (NEW)
Wraps a `tch::CModule` (TorchScript-exported GNN). The ONLY thing touching libtorch.
- Interface: `evaluate(batch: &[State]) -> Vec<(PolicyLogits, Value)>`
- Internally: serialize states → graph tensors → one `forward()` on GPU → return. Batches across all concurrent games.
- For arena: holds **two** CModules but batches each net's leaves to full size (recovering the lost parallelism — the fix for the 2-evaluator problem).

### 4.3 `export_torchscript.py` — the train→infer bridge (NEW, Python)
After PyTorch training: `torch.jit.script(GnnModel)` → save `net.ts`. One small script, run once per iteration. Wired into `catan_az.loop._default_train` after `checkpoint_best.pt` is written.

### 4.4 PyO3 surface — thin (MODIFIED)
`catan_az.daily` calls into `catan_mcts_rs` via the coarse entry points above — one crossing per *stage*, not per node. Returns parquet-ready results matching the current `SelfPlayRecorder` / arena `results.jsonl` schema so downstream (dataset, ladder, journal) is unchanged.

### 4.5 Existing PyTorch training — UNCHANGED
`catan_gnn.train`, the dataset, the AZ loop orchestration all stay. Only the self-play and arena engines swap.

---

## 5. Verification Strategy (CORE OF THIS PLAN — DO NOT SHORTCUT)

**Principle: the Python implementation is the ORACLE.** Every Rust component must prove BIT-EXACT match against Python before it is trusted. **Double verification = two independent check classes per unit** (golden-parity AND differential/property); never just one. A Rust MCTS that silently diverges from the proven Python one poisons every future iteration invisibly — there is no loud failure, just slowly-wrong search.

### Phase 0 — Spike + parity gate (BEFORE committing to the full build)
1. `torch.jit.script(GnnModel)` → load in tch → run **50 fixed states** through BOTH the live PyTorch net and the TorchScript-in-Rust path → assert outputs match.
   - **Standard for the NET forward specifically:** TorchScript runs the same libtorch kernels as PyTorch, so policy logits and value should match to **floating-point identity** on the same device. Assert **max abs diff = 0** (CPU) — if and only if GPU non-determinism in a specific kernel makes exact-0 impossible, the bound may relax to **≤ 1e-6**, but this MUST be (a) justified by identifying the exact kernel, (b) documented in the test, and (c) NOT propagated to the MCTS/decision layer.
   - **Standard for MCTS DECISIONS (visit-counts, chosen action, game records):** **BIT-EXACT, zero tolerance.** Discrete outputs (argmax, visit counts, action ids, winners) must be identical — there is no float-epsilon excuse for a different *decision*. The net-forward epsilon (if any) must not change which action is chosen; if it does, that is a failure to investigate, not to tolerate.
2. If PyG message-passing won't script cleanly: **STOP and pivot** to a fixed-topology net (the Catan graph is fixed-shape — 54 vertices / 72 edges / 19 hexes — so message-passing can be precomputed sparse matmuls with no PyG). This decision is made at spike time with real evidence, not assumed now.

### Bit-exact determinism requirement
MCTS with the same (seed, net, RNG stream) must be reproducible in BOTH languages. **Replicate numpy's RNG (PCG64 / the exact generator `async_mcts.py` uses via `np.random.default_rng(seed)`) in Rust, OR switch both sides to a single shared deterministic RNG**, so visit-counts and chosen actions match bit-for-bit. Bit-exact is REQUIRED everywhere — statistical equivalence is NOT an acceptable fallback (user decision). Replicating the RNG is part of the work.

### Per-component double verification

| Unit | Check 1 — golden-parity (oracle) | Check 2 — differential/property |
|---|---|---|
| TorchScript net | 50-vector golden test vs live PyTorch | random states: Rust == PyTorch (bit-exact) |
| Rust state / apply_action | fixed seeds → same legal_actions, VP, terminal as Python engine | N random playouts: board state agrees move-by-move |
| Rust MCTS search | fixed seed + fixed net → **identical visit-counts AND chosen action** as Python MCTS | M seeds: chosen action agrees 100% |
| Full self-play game | one seed → same length, winner, action_history | N games: length/winner/value-target distributions identical |
| Full arena | fixed seeds → same per-game winner + vp_margin | winrate over N games identical to Python |

### End-to-end gate (NON-NEGOTIABLE)
Before any Rust-generated games train a real net: run **the same 100 seeds** through BOTH Python and Rust self-play paths and prove the resulting game records (length, winner, action_history, recorded visit-counts, value targets) are **identical**. Only then does Rust self-play feed the loop. Same gate for arena before a Rust arena verdict is trusted for promotion.

---

## 6. Rollout / Sequencing

1. **Phase 0 spike** (gate): TorchScript export + tch round-trip + 50-vector parity. Pivot decision on PyG if needed.
2. **RNG parity**: replicate numpy generator in Rust; bit-exact unit test vs numpy.
3. **Rust state/apply_action wrapper** + golden + differential tests vs Python engine.
4. **Rust MCTS search** + visit-count/action parity vs `async_mcts.py`.
5. **TorchScriptEvaluator** (batched, single-net then two-net for arena).
6. **Full self-play path** + end-to-end 100-seed gate.
7. **Full arena path** + end-to-end gate.
8. **Wire into `catan_az`**: coarse PyO3 entry points; `export_torchscript.py` in the train step; swap self-play + arena engines. Orchestration untouched.
9. **Production validation**: one full iter_8 on the Rust path; confirm games finish naturally (no timeout), GPU utilization rises, throughput improves, and the verdict matches what the Python path would have produced (run both once for the final cross-check).

Each step is TDD (red/green) AND double-verified per §5 before the next begins.

---

## 7. Out of Scope / Non-Goals
- No change to the AZ loop orchestration, ladder, journal, window logic, or training code (beyond adding the TorchScript export call).
- No multi-GPU / multi-machine (the design should not *preclude* it, but it is not built now).
- No net architecture change UNLESS the Phase 0 spike proves PyG won't TorchScript (then a fixed-topology rewrite, validated to match current strength).

---

## 8. Context / Related Findings (from memory; verify before relying)
- `project_arena_latency_bound_2026_06_17` — the measurements above.
- `project_mcts_pyo3_boundary_bottleneck` — the original "fix in Rust" thesis (predates GNN; the rollout-to-Rust fix it describes is moot now — there are no random rollouts, the GNN is the evaluator — but the boundary-cost thesis holds).
- `project_batched_eval_gate1_2026_05_30` — batching works but became CPU/orchestration-bound; motivates removing Python from the loop.
- `project_gnn_value_perspective_bug_2026_05_30` — the GNN value head is EGO-relative; the Rust MCTS must rotate value to absolute seat before backup, exactly as the fixed Python async MCTS does. **Port this correctly or Q-values are poisoned.**
- `feedback_rust_rewrite_tdd_double_verify` — the verification mandate (this §5).
- `feedback_worktree_swap_breaks_pyo3_install` — rebuild PyO3 (`maturin develop`) after Rust changes; WSL may need `--shutdown`.

## 9. Environment Notes (for the implementing agent)
- Worktree: `C:\dojo\catan_bot\.claude\worktrees\az-bots\mcts_study\` (Windows) = `/mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study/` (WSL).
- Run Python in WSL via the venv: `source /home/chitii/catan_mcts_venvs/mcts-study/bin/activate`. Invoke WSL as user: `wsl.exe -d Ubuntu -u chitii -- bash -lc '...'`.
- WSL nested-quoting mangles inline `$()` and loops — **put multi-step shell logic in a script file and run the file**.
- Rebuild after Rust changes: `maturin develop --release` from the worktree (source `~/.cargo/env` first).
- The existing reference implementations to match: `catan_mcts/async_mcts.py` (MCTS), `catan_az/arena.py` (`_play_arena_game`, seating rotations), `catan_mcts/experiments/self_play_async.py` (self-play driver), `catan_gnn/gnn_model.py` (`GnnModel`).
- AZ loop runs are at `/home/chitii/catan_data/runs/v3/az_loop/` (WSL symlink off the C: drive).
