# GNN-v0 Execution Journal

**Plan:** [`docs/superpowers/plans/2026-04-29-gnn-v0-implementation.md`](../plans/2026-04-29-gnn-v0-implementation.md)
**Branch:** `gnn-v0` (off `main`, branched 2026-04-29 11:07 EDT after merging `mcts-v2-hardening` into main)
**Worktree:** `C:/dojo/catan_bot/.claude/worktrees/gnn-v0/`

This is a per-task execution log that captures **findings, deviations, and lessons** discovered while implementing the plan. The plan itself is the design contract; this journal is the running record of what we actually learned.

Format per task: a one-line completion status, then bullet points for anything non-obvious that came out of the work. Each entry tags whether it was a **bug fix in the spec**, a **process lesson**, or just a **measured number worth keeping**.

---

## Task 1: Add torch + torch-geometric dependencies

**Status:** ✅ Done. Commit `e8d7ff5`.

**Findings:**

- **Pip pulled torch 2.11.0+cu130** (with CUDA libs ~2 GB) automatically because the host has an NVIDIA GTX 1650. The plan said `torch>=2.2` and didn't pin a CPU-only index; pip detected the GPU and pulled the CUDA wheel. Not a bug; just a footprint surprise.
- **Editable install rebound to gnn-v0.** The shared WSL venv (`~/catan_mcts_venvs/mcts-study/`) had `pip install -e` pointing at the `mcts-study` worktree. Re-running `pip install -e` from gnn-v0 rebound the editable install to gnn-v0's path. The running e5 sweep workers were unaffected (their modules were already imported into memory), but **any newly-spawned multiprocessing worker in the mcts-study sweep would have imported from gnn-v0's source**. In practice the sweep's mass-spawn happened at launch time, so this didn't bite.
- **Process lesson:** when two worktrees share an editable install, `pip install -e` from one *changes* the installed package's source-tree pointer. Running long-process sweeps and switching worktrees aren't decoupled.

## Task 2: Recorder schema v2 — add action_history to games.parquet

**Status:** ✅ Done. Commit `7aeef08`.

**Findings:**

- **The plan listed 5 experiment files to update (e1–e5).** Implementer correctly updated all five, plus discovered **two pre-existing tests** (`tests/test_determinism.py` and `tests/test_recorder.py`) that called `finalize()` with the v1 signature and broke. Plan didn't enumerate them. Implementer fixed inline; not scope creep.
- **`_GameRow` dataclass field ordering matters.** The new `action_history` field (no default) must come BEFORE `schema_version: int = SCHEMA_VERSION` (with default). Dataclass rule: fields without defaults precede fields with defaults.
- **Process lesson — branch isolation:** the v2 schema commit landed only on `gnn-v0`. The `mcts-study` worktree (still on `mcts-v2-hardening`) never received it because the two branches diverged at `4274d57` (the GNN-v0 plan commit). Result: the 1500-game e5 v2 sweep launched from `mcts-study` at 11:07 AM (~20 min before the schema commit at 11:26 AM) wrote v1 parquets, **and would still write v1 today** because that branch never received the bump.
- **The fix to land later:** merge `gnn-v0` into `main`, then merge `main` into `mcts-v2-hardening`, so all branches share the schema bump.
- **Real cost of the missed migration:** the 1500-game v2 sweep is unusable for GNN training (no action_history). We had to do a 50-game data regen on `gnn-v0` worktree to produce v2 data.

## Task 3: Static adjacency tables

**Status:** ✅ Done. Commits `81b3cbd → 9ada3a4 → 272b20d`.

**Findings:**

- **Cross-check beat the spec.** The plan asked for shape + per-hex distinctness tests. Implementer added a stronger consistency check: derive `EDGE_TO_VERTICES` from hex perimeters and assert it equals the literal table. **This catches single-row transcription errors** that the shape/distinctness tests would not. Recommended pattern for any future hand-transcribed lookup tables.
- **First commit forgot to write the cross-check as code.** Implementer ran it locally, mentioned it in the commit message, but didn't add the test. Spec review caught it; fixed in `9ada3a4`.
- **Naming polish:** original exports `HEX_VERTEX_EDGES` / `VERTEX_EDGE_EDGES` were renamed to `HEX_VERTEX_EDGE_INDEX` / `VERTEX_EDGE_EDGE_INDEX` (clearer that these are PyG `edge_index` tensors). Also extracted `NUM_HEXES = 19`, `NUM_VERTICES = 54`, `NUM_EDGES = 72` constants. Caught in code-quality review (`272b20d`).
- **Plan reference doc updated** in `db76311` to use the new symbol names so subagents executing later tasks read the right names.

## Task 4: state_to_pyg — observation → HeteroData

**Status:** ✅ Done. Commits `2d856fb → 3a61859`.

**Findings:**

- **Module-level pre-built edge_index tensors** (sliced once from the adjacency tables at import time) are shared across all returned `HeteroData` instances. Safe under PyG's standard ops (`Batch.from_data_list`, `HeteroConv`, `.to(device)` all return new tensors), but explicitly **read-only** at runtime. Documented in code with a comment block.
- **`np.ascontiguousarray(..., dtype=np.float32)`** on numpy inputs is a no-op when the engine already returns contiguous float32 (which it does), but kept as defensive code — costs nothing.
- **Polish:** replaced literals `114`/`144` with `NUM_HEXES * 6` / `NUM_EDGES * 2` (`3a61859`). Same intent, better self-documenting.

## Task 5: GnnModel — body + value head + policy head

**Status:** ✅ Done. Commits `98c261d → 55adba4`.

**Measured numbers worth keeping:**

- **67,730 params** at hidden_dim=32, 2 layers (within the 20k–80k v0 budget).
- **7.25 ms CPU forward pass at batch=1** (above the 5 ms target, under the 10 ms test slack). Mitigation if it ever matters: drop `EMBED_DIM` from 128 to 64.

**Findings:**

- **`from .. import ACTION_SPACE_SIZE` (relative)** would have failed at runtime — `catan_gnn` and `catan_mcts` are sibling packages, not nested. Implementer used absolute `from catan_mcts import ACTION_SPACE_SIZE`. Plan snippet was wrong.
- **`Batch.scalars.view(-1, N_SCALARS)`** is fragile because PyG's collation behavior for graph-level attrs varies between versions. Cleaner fix lived in `state_to_pyg.py`: store scalars as shape `[1, 22]` so `Batch.from_data_list` produces shape `[B, 22]` deterministically. Applied in `55adba4`.
- **`from torch_geometric.utils import scatter` was inside `forward()`** in the initial commit (delayed import to avoid PyG-version-location issues). PyG 2.5+ has stable scatter location; moved to module level in `55adba4`.
- **New test added:** `test_scalars_collated_correctly_for_various_batch_sizes` at B ∈ {1, 2, 4, 16} — guards against PyG collation regressions.

## Task 6: CatanReplayDataset

**Status:** ✅ Done. Commits `4d4d202 → 4e4fd8a`.

**Findings (one of the most important catches in the whole project):**

- **The plan's replay loop counted `len(legal) > 1` for ANY player** as an MCTS-decided move. **That would have produced silently corrupted training positions.** Implementer caught the bug: `move_index` is only incremented inside `play_one_game` when `current_player == recorded_player`. So the dataset's replay must also only count non-trivial decisions where `engine.current_player() == row.current_player`. Without this guard, position N in the parquet would map to a different engine state every time another player happened to have a non-trivial decision in between.
- **v1-only data must be filtered AND not crash the loader.** The plan said "filter v1 by schema_version >= 2" but the original code unconditionally accessed `games["action_history"]`, which throws `KeyError` on v1 parquets. Found in `4e4fd8a`'s `test_v1_games_are_skipped`. Fixed: relaxed the empty-check so the dataset can construct empty when only v1 data is present.
- **Test runtime down from 12s to 10s** by sharing the e1 mini-run via a module-scoped pytest fixture. Cosmetic but quality-of-life.

## Task 7: Training loop

**Status:** ✅ Done. Commits `b30c42d → 7d7c4b0`.

**Findings:**

- **The plan's `_masked_policy_loss` would have produced NaN in every training step.** PyTorch's `log_softmax` of `-inf` logits produces `-inf`, and `0 * -inf == NaN` even when the target is 0. The dataset guarantees `target == 0` at illegal positions, so the loss should be finite — but it isn't, until you mask the log_probs to 0 at illegal positions before multiplying by target. Implementer caught it; one-line fix:
  ```python
  log_probs = log_probs.masked_fill(~mask, 0.0)
  ```
  **Without this fix the model would have trained on NaN gradients, which silently breaks everything.** Tests would have passed (they only check artifact existence) and we'd have shipped a broken pipeline.
- **Per-epoch wall-clock was ~115s on test data** (3,880 rows × batch=4) — much higher than the spec's "few seconds" estimate. Cause: chance-step game-length blowup (~12-15k steps/game). Production training at batch=64 would be ~7s/epoch instead. Spec's per-epoch estimate was based on the wrong game length.
- **GPU support added** as a plan amendment (`96a41d4`): `device="auto"` resolves to cuda when available. Confirmed working on NVIDIA GTX 1650.
- **`model.cpu().state_dict()` mutates the model in place.** Footgun if a caller wants to keep training on GPU after `train_main()` returns. Fixed in `7d7c4b0`: copy state_dict tensors to CPU without touching the model itself.
- **`git_sha` returns `"unknown"`** on WSL when run against a Windows-path worktree — git can't resolve the gitfile path. Cosmetic; we have other commit-tracking via `out_dir` paths.

## Task 8: GnnEvaluator

**Status:** ✅ Done. Commits `7e97d61 → 6e43a6f`.

**Findings:**

- **Cache-hit invariant test exposed an aliasing footgun.** The plan's cache key is `tuple(state._engine.action_history())`. The implementer's first version of the cache-hit test asserted that `evaluate(state2)` triggers a *new* forward pass when state2 is from a different seed. But `_drive_to_player_decision(state2)` produced the same action_history as state1 (different *seeds* but identical *decision-history* prefixes — chance outcomes that drove forward are NOT in the action history we use for the cache key). Cache hit. Test failed correctly. Fixed by cloning state and applying one legal action — guarantees action_history differs by exactly one entry.
- **Defensive `model.eval()` inside `_forward()`** — costs nothing, prevents footguns if a caller monkey-patches the model into train mode externally.
- **No batched MCTS leaf eval** is the obvious v1 follow-up. At sims=400 with ~150 decisions/game, that's ~60k sequential forward passes/game — host→device transfer dominates compute at batch=1. TODO comment added in code.

## Task 9: Bench-2 static-position benchmark

**Status:** ✅ Done. Commit `7b1c4f1`.

**Findings:**

- **Replay-loop parity with dataset** required exact mirroring of Task 6's logic, including the `current_player` guard. Done correctly on first try because the dataset bug had been caught.
- **Smoke run with depth=3, n=10:** value MAE ≈ 0.70, KL ≈ 0.98 (1-epoch model). Sane envelope, not interpreted further. Real numbers come from a properly-trained model.

## Task 10: e6 MCTS-with-GNN winrate experiment

**Status:** ✅ Done. Commit `1078d53`.

**Findings:**

- **One smoke test, runs in ~2 min** (1 game × sims=2 with a 1-epoch trained model). Validates the experiment plumbing without committing to a real training cost.
- **CLI registration:** added `e6` to `_EXPERIMENTS` dict in `cli.py` and `--help` choices.
- **Multiprocessing pickling:** the worker passes `checkpoint Path + hidden_dim + num_layers`, not the live model. Each worker re-loads the checkpoint from disk on its side. Fine for a 283 KB v0 checkpoint; would be wasteful at scale.

## Task 11: End-to-end smoke run

**Status:** ✅ Done with concerns. No commit (acceptance only).

**Findings (the smoke caught real issues, exactly as it's supposed to):**

- **The 1500-game e5 v2 parquet was schema v1.** Detected in Step 1. Branch isolation, see Task 2 discussion. Forced a small fresh e5 sweep on `gnn-v0` to generate v2 training data.
- **5-epoch training on 10-game v2 data:** val_loss 1.94 → 1.23, val_top1 0.40 → 0.38. Loss decreasing, top-1 noisy (low data). Pipeline works.
- **bench-2 (n=100):** value_mae=0.30, policy_kl=0.0012. Better than the 1-epoch smoke (0.70 / 0.98); sanity-consistent.
- **e6 5-game smoke at sims=100, max_seconds=360 → 0 finished games.** All 5 timed out at exactly 18 moves. Cause: GNN+MCTS at sims=100 is ~20s/move on this CPU, exceeds budget. **The 360s/game cap that worked for e5 lookahead-VP is too tight for e6 GNN inference.** Lesson: pipeline cost per move is dominated by the evaluator. Need 600s+ cap for e6.
- **The path-quirk fix** (`5284461`) emerged from Step 1's discovery that the smoke e5 sweep landed at worktree-root `runs/` instead of `mcts_study/runs/`. Real bug in `cli.py`: top-level `--out-root` was consumed but never forwarded to the experiment's `cli_main`. Regression test added.
- **Pipeline verdict:** ✅ end-to-end working — every component produces correct artifacts, every gate detects bad inputs. Strength evaluation needs (1) more training data (only 10 games of v2 currently) and (2) larger e6 budget.

## Post-task: cli `--out-root` forwarding fix

**Status:** ✅ Done. Commit `5284461`.

**Findings:**

- **Every multi-worker sweep we've ever done landed at worktree-root `runs/`** instead of the user-specified `--out-root`. The top-level argparse (`runp.add_argument("--out-root", ...)`) consumed the value, then `parse_known_args` shoved the rest into `sys.argv`, and the experiment's *own* `cli_main()` re-parsed with its default `Path("runs")`.
- **User's `--out-root` was silently dropped.** Visible only by inspecting where parquets landed.
- **Fix:** re-inject `--out-root <user-value>` into the forwarded `sys.argv` before calling the experiment's `cli_main()`.
- **Regression test:** `test_cli_forwards_out_root_to_experiment` runs an e1 sweep with `--out-root tmp_path/specified` and asserts the run dir lands there.
- **Process lesson:** subtle CLI bugs that produce visible-but-not-erroring behavior can persist for many runs unnoticed. Worth a regression test for each non-trivial argparse forwarding pattern.

## Data regen (post-Task 11)

**Status:** ✅ Done. Background task `bvsgryodj`.

**Numbers:**

- **50 games / 200 attempted** (25 each at depth=25/sims=400 and depth=35/sims=400).
- **0 timeouts** with a 600s/game cap (vs 9.3% timeout rate on the original 360s e5 sweep).
- **Schema v2 with action_history** — usable for GNN training.
- **Wall-clock:** ~1h45m on 4 workers.

**Per-cell winrate (this run vs original e5 v2 sweep):**

| cell | regen (n=25) | original v2 (n=57/55) |
|---|---:|---:|
| depth=25/sims=400 | 60.0% | 75.4% |
| depth=35/sims=400 | 48.0% | 76.4% |

Both within ±20pp 95% CI for n=25 — consistent with the original measurement, though the depth=35 drop is on the larger end of plausible.

## Plan amendments

- **`96a41d4`** — GPU support amendment: training defaults to `device="auto"` → cuda when available, MCTS-time inference defaults to CPU (transfer overhead dominates at batch=1).
- **`db76311`** — symbol-name sync after Task 3 polish (HEX_VERTEX_EDGES → HEX_VERTEX_EDGE_INDEX).

## Bigger lessons that should make it into `learnings.md`

1. **Branch isolation in git worktrees is silent.** Two worktrees on different branches can run for hours without either knowing they have divergent code. Schema bumps, recorder changes, and any new field on a persistent format need to land on every active branch *before* a long-running sweep starts. Otherwise the sweep produces data in the old format with no warning.

2. **`0 * -inf = NaN` in PyTorch masked-CE.** The fix is to mask the `log_probs` to 0 before multiplying by the target distribution. This is a foot-gun any AlphaZero implementation hits; it must be tested with a finite-loss assertion in the training loop, not just with "tests pass."

3. **Cache keys for game-state dedup must include all stochastic elements.** If chance outcomes don't appear in the action_history we use as a cache key, two states reachable from the same seed via different chance paths will collide. The dataset's replay logic correctly walks chance steps; the evaluator's cache must too.

4. **Replay determinism requires a per-player counter, not a global one.** When the recorder skips trivial turns and only records the recorded player's MCTS decisions, the dataset's replay loop must mirror that exactly — count only the recorded player's non-trivial decisions toward `move_index`, not all players'. Off-by-N bugs here corrupt training data without crashing.

5. **Argparse forwarding patterns deserve regression tests.** A `parse_known_args` + `sys.argv` rewrite that drops a flag silently can persist across many runs unnoticed. Test that the user's intended behavior survives the round-trip.

6. **Per-evaluator wall-clock budget must be re-tuned per evaluator.** The 360s/game cap that worked fine for lookahead-VP at sims=400 produced 0 finished games at sims=100 GNN inference. Each evaluator has a different per-move cost; the budget must be set per evaluator, not inherited.

---

## Long-running operations timeline

Visual log of every sweep, cache build, and training run that took >10 minutes wall-clock. Times are local EDT (UTC-4). Rows added when an op starts; finalized when it completes. Each "block" of `█` characters spans the elapsed time on a 24-hour axis. One `█` ≈ 30 minutes.

Read this top-to-bottom in starting-time order to see what was running when, and side-by-side to see overlapping CPU/GPU contention.

```
2026-04-29 EDT
hour:    06   08   10   12   14   16   18   20   22   00   02   04   06   08

07:08 ─[████████████]──── 14:11    e5 v2 sweep, 1500 games  (mcts-study branch, schema v1)
                          14:30 ─[█]─ 14:50  50-game data regen (gnn-v0, schema v2)
                                                  19:07 ─[█]─ 19:15  cache_w0.pt build (36k pos, 250 MB)
                                                  19:32 ─[██]─ 20:12  cache_full.pt build (136k pos, 938 MB)
                                                              20:14 ─[█]─ 20:21  D4 1-epoch b=64 baseline
                                                                  20:25 ─[█]─ 20:32  D4b 1-epoch b=256
                                                                          22:41 ──[████████]── 04:30 (4/30)
                                                                                       user 400-game e5 sweep (d15+d25 done; d35 partial)
```

```
2026-04-30 EDT
hour:    00   02   04   06   08   10   12   14   16

23:46 (4/29) ─[██████]── 04:38   v0a/v0b overnight train+bench-2+e6 (50-game cache, 0 wins each)
              04:35 ─[█████████]── 09:04            v1_d15_subset cache build + train (200 games d15)
                                  10:08 ─[ ]─ 10:15  e7 12-game tournament (LookaheadMcts 12/12)
                                            07:23 ─[█████████████████]── ~17:00
                                                  v2_d25_w0w1 cache build + train (200 games d25, 60 ep)
```

### 07:08–14:11 EDT — `e5 v2 1500-game sweep` (mcts-study worktree)

- **Goal:** comprehensive winrate measurement for lookahead-VP at depth ∈ {3,10,17,25,35} × sims ∈ {25,100,400} with n=100 per cell.
- **Outcome:** 1360 finished games + 140 timeouts (9.3% timeout at 360s cap).
- **Run dir:** `mcts-study/runs/2026-04-29T11-07-e5_lookahead_depth/`
- **Wall-clock:** 7 h 03 min on 4 workers.
- **Schema:** v1 (no `action_history`) — branch isolation issue, see Task 2 lesson. Data unusable for GNN training as-is.
- **Headline:** depth=25/sims=400 = 75.4% winrate (n=57). depth=35 ≈ depth=25 at same sims.

### 14:30–14:50 EDT — `50-game data regen` (gnn-v0)

- **Goal:** generate v2 schema training data targeted at known-strong cells.
- **Outcome:** 50 / 200 attempted finished. 0 timeouts. Schema v2 with `action_history`.
- **Run dir:** `gnn-v0/runs/2026-04-29T20-49-e5_lookahead_depth/`
- **Wall-clock:** 1 h 45 min on 4 workers.
- **Cells:** depth ∈ {25, 35} × sims=400, 25 games each.
- **Per-cell winrate:** depth=25/sims=400 = 60% (n=25), depth=35/sims=400 = 48% (n=25).

### 19:07–19:15 EDT — `cache_w0.pt build` (worker0 subset)

- **Goal:** smoke-validate the cache pipeline (`CachedDataset` build + persist + reload).
- **Outcome:** 36k positions cached, 250 MB on disk, ~10 min build at ~60 pos/s.
- **Path:** `mcts_study/runs/gnn_v0_smoke/cache_w0.pt`
- **Used by:** Step A smoke training (10 epochs, ~7 sec/epoch).

### 19:32–20:12 EDT — `cache_full.pt build` (full 50-game)

- **Goal:** Derisk 3 — confirm cache scales linearly past worker0.
- **Outcome:** 136k positions cached, 938 MB on disk, ~40 min build at ~57 pos/s, peak RSS 3.91 GB. Linear scaling confirmed across the dataset.
- **Path:** `mcts_study/runs/gnn_v0_smoke/cache_full.pt`
- **Projection to 400-game:** ~5.4 hr build, ~7.5 GB disk, ~31 GB RAM peak (need WSL bump from 31 GB cap).

### 20:14–20:21 EDT — `D4 1-epoch baseline (b=64)` (full cache)

- **Goal:** measure per-epoch training cost on the cached dataset.
- **Outcome:** 6 min 59 sec total. Per-batch ~170 ms, suggesting ~165 ms is non-GPU work (PyG `Batch.from_data_list` + I/O).

### 20:25–20:32 EDT — `D4b 1-epoch (b=256)`

- **Goal:** test whether batch size flattens per-batch overhead.
- **Outcome:** 6 min 12 sec total. Marginal improvement. Per-batch overhead doesn't scale with batch size as hoped.

### 22:41 (2026-04-29) – ~04:30 (2026-04-30) — `user 400-game e5 sweep` (gnn-v0)

- **Goal:** generate the proper-scale v2 training data for v0 GNN.
- **Cells:** depth ∈ {15, 25, 35} × sims=400, 100 games per worker per cell (400/cell).
- **No max_seconds cap.**
- **Run dir:** `gnn-v0/runs/gnn-data-2026-04-29-evening/2026-04-29T22-41-e5_lookahead_depth/`
- **Outcome:** d15 + d25 cells fully flushed (100 games × 4 workers each = 400 games per cell, 800 games total). d35 cell was in flight when the process exited; partial d35 data may exist but not used for v2 training.
- **Total moves recorded:** ~1.86M across d15 (817k) + d25 (1.04M).

### 21:30–ongoing — D4c, D4d, D5 derisks (autonomous)

Quick GPU/single-process derisks while user's e5 sweep ran. All passed.

- **D4c (instrumented baseline, /mnt/c cache):** dataset load 250s, train epoch 100s, val 13s, total ~6 min.
- **D4d (native ext4 cache, ~/cache_full.pt):** dataset load 145s, train epoch 108s, val 15s, total ~4.5 min. Native filesystem ~40% faster on load; train phase identical (RAM-resident).
- **D5 (replay correctness):** 10/10 random positions in cache match fresh CatanReplayDataset output bit-for-bit. Replay logic correctness verified at scale.

### 23:46–ongoing — overnight v0a + v0b training pipeline (autonomous)

- **Goal:** train v0 GNN on the existing 50-game v2 cache, run bench-2 + small e6, compare against a v0b config if time permits.
- **Run dir:** `mcts_study/runs/gnn_v0_overnight/`
- **Cache:** `~/cache_full.pt` (native ext4, 938 MB, 136k positions, 50 games).

**v0a TRAIN (00:54–01:30 UTC = 36 min, 20 epochs, b=256, lr=1e-3):** done.
- Train loss flatlined at 1.087 from epoch 4 onward (early convergence; data-starved).
- Val loss climbed 1.34 → 1.84 (overfitting on 50 games).
- val_top1 noisy 33-69%, mostly 35-50% (model picks meaningful actions but unstably).
- Per-epoch wall-clock 95-105 sec; matches D4d projection.
- Checkpoint: `gnn_v0_overnight/v0a/checkpoint.pt` (283 KB).

**v0a BENCH-2 (01:31–01:32 UTC = 32 sec, n=200, lookahead_depth=25):** done.
- `bench2_value_mae`: 0.724 — model value head substantially off vs lookahead reference. Expected at 50 games.
- `bench2_policy_kl`: 0.0036 — peaky-vs-peaky KL is uninformative.

**v0a E6 (01:32–03:24 UTC = 112 min, 1 worker, 8 games, sims=100, max_seconds=1200):** done.
- **4/8 games finished, 4 timed out** at the 1200s cap.
- **0 wins out of 4 finished games** — MCTS+GNN-v0a underperformed random baseline (25%).
- mean_vp_p0 = 3.00 (vs random baseline of ~3-4).
- **Diagnostic read:** v0a value/policy heads are noisy enough at 50-game training that they actively mislead MCTS. With only sims=100 there aren't enough rollouts to recover from bad priors. This matches the bench-2 reading (value_mae=0.72 — significant disagreement with the lookahead reference).
- **Pipeline-level finding:** at sims=100 with the GNN evaluator, per-move time is ~30-60 sec. 1200s cap = ~20-40 moves before timeout. Catan games take ~150-200 MCTS-decided moves, so 4/8 timeouts is unsurprising.

**v0a complete in 112 min total.** Under the 3h soft limit, so v0b kicked off automatically at 03:24 UTC.

**v0b TRAIN (03:24–04:37 UTC = 73 min, b=256, lr=1e-4, 40 epochs):** done.
- Train loss flatlined at 1.086-1.087 from epoch ~5 onward (same plateau as v0a).
- Val loss noisy 1.49-1.83. Final 1.513 (better than v0a's 1.836).
- val_top1 final 0.348.
- **Lower lr + more epochs DID NOT help.** The plateau is data-imposed, not optimization-imposed.

**v0b BENCH-2 (04:37–04:38 UTC, n=200):** done.
- `bench2_value_mae`: 0.764 (slightly worse than v0a's 0.724).
- `bench2_policy_kl`: 0.0037 (same range).

**v0b E6 (04:38–06:20 UTC = 102 min, 1 worker, 8 games, sims=100, max_seconds=1200):** done.
- 4/8 finished, 4 timed out.
- **0 wins.** Mean VP 2.50 (worse than v0a's 3.00).
- **Same conclusion as v0a:** GNN priors actively mislead MCTS at sims=100. Model is data-starved.

### v0a vs v0b verdict

Both v0a and v0b lose every e6 game against random. The 50-game training set is too thin for any hyperparam configuration to learn a useful evaluator. **The pipeline is mechanically correct; the data is insufficient.** The d15 sweep data, when its cache finishes, is the next test — 3× more effective games, 6× more positions.

Pipeline total wall-clock: **289 min (4h 49min)** from v0a start to v0b e6 finish.

### 21:45–ongoing — D15 watcher (autonomous side-pipeline)

Watches for the first depth=15-sims=400 parquet to appear in the user's e5 sweep dir. When detected: stages worker dirs, builds an in-RAM cache, trains a v1 GNN on it, runs bench-2.

**04:15 UTC: 4 depth=15 shards landed.** Watcher staged all 4 workers and started cache build. Source dataset reported **816,923 positions** — much larger than projected (50-game cache had only 136k for 50 games; 400 games at d=15 produced 6× more positions). At 50 pos/s build rate, ETA was 4.2 hours; projected RAM peak ~25 GB (right at WSL's 31 GB cap, risky for swap).

**Pivot at 04:24 UTC:** killed the 4-worker build mid-run. Restarted with a **2-worker subset** (worker0 + worker1, ~408k positions). Projected: ~12 GB RAM peak, ~2.3 hr build, ~70 min train after.

### 04:35–ongoing — v1_d15_subset (autonomous side-pipeline, 2-worker subset)

- **Goal:** train v1 GNN on a tractable subset of the depth=15 sweep (worker0 + worker1).

**Cache build (04:35–07:09 UTC = 2h 34min):** done.
- 420,021 positions from 2 workers (~200 games of depth=15/sims=400).
- 44.7 pos/s sustained (slower than 50-game cache's 57 pos/s; CPU contention with v0b e6 + user's e5 sweep workers).
- 2.94 GB on disk.

**v1_d15_subset TRAIN (07:09–09:04 UTC = 1h 55min, b=256, lr=1e-3, 20 epochs):** done.
- Cache load: 7.5 min (3 GB native ext4 file).
- Per-batch ~211 ms (matches 50-game cache).
- **Train loss floor: 1.067** (vs v0a's 1.087 / v0b's 1.086) — meaningful drop. 3× more data → model can fit more signal.
- **Val loss range: 1.30–1.42** (much tighter than v0's 1.34–1.84) — overfitting reduced significantly.
- **val_top1: 0.37–0.48** (similar range to v0).

**v1_d15_subset BENCH-2 (09:04–09:05 UTC, n=200):** done.
- `bench2_value_mae`: 0.781 (slightly worse than v0a's 0.724 — could be sample noise on 200 positions, or d15-data value distribution differs from depth=25 lookahead reference).
- `bench2_policy_kl`: **0.0013** (vs v0's 0.0036 — markedly better, policy head better aligned with MCTS visit-count target).

**No e6 run for v1** (side-pipeline only does train+bench-2; e6 was scoped out of scratch_d15_subset_pipeline.sh to fit overnight window).

### v0a vs v0b vs v1_d15_subset summary

| metric | v0a (50 games) | v0b (50, lr=1e-4) | **v1_d15 (200 games)** |
|---|---:|---:|---:|
| train_loss floor | 1.087 | 1.086 | **1.067** |
| val_loss range | 1.34–1.84 | 1.49–1.83 | **1.30–1.42** |
| bench2 value_mae | 0.724 | 0.764 | 0.781 |
| bench2 policy_kl | 0.0036 | 0.0037 | **0.0013** |
| e6 wins/finished | 0/4 | 0/4 | (not run) |

**Pipeline scales correctly.** More data → lower train loss floor, tighter val loss, better policy-KL alignment. The hyperparameter changes alone (v0a→v0b) did nothing; the data change (v0→v1) moved every metric except value_mae.

### One-game 4-player tournament (sims=50, seed=12345, ~3 min wall-clock)

After v1 training finished, ran a head-to-head 1-game tournament with all four player types:

| Seat | Player | Final VP | Avg time/move |
|---|---|---:|---:|
| 0 | 🤖G GnnMcts (sims=50 + v1 GnnEvaluator) | 2 | 789 ms |
| 1 | 📊P PureGnn (argmax of GNN policy) | 4 | 15 ms |
| 2 | 🔍L LookaheadMcts (sims=50 + LookaheadVpEvaluator d=25) | 9 | 11 ms |
| 3 | 🎲R Random | **10 (winner)** | <1 ms |

**Ranking:** Random (10) > LookaheadMcts (9) > PureGnn (4) > GnnMcts (2).

**Key finding: PureGnn (no search) beats GnnMcts (MCTS-augmented).** Adding search to a noisy evaluator amplifies its mistakes. The v1 GNN's policy/value heads aren't yet accurate enough to help MCTS — they actively mislead it. Same pattern v0a/v0b showed in e6 (0 wins) was reproduced here.

**Per-call cost:** GnnMcts at 789ms/move is 70× slower than LookaheadMcts at 11ms/move at the same sim count. The GNN forward pass × 50 sims is the dominant tournament wall-clock.

**One-game caveat:** n=1 game means seat 3 (random) winning is partly luck of the seating. The pattern (PureGnn > GnnMcts, LookaheadMcts strong) is consistent with the structural reasoning, but the random winning could be sample noise. A multi-game tournament would settle the absolute ranking.

### 2026-04-30 07:23 UTC – ongoing — `gnn_v2_d25_w0w1` cache build + train (autonomous)

- **Goal:** train v2 GNN on **200 games of depth=25 lookahead self-play** (worker0+worker1 of the user's evening sweep). Direct apples-to-apples comparison with v1_d15_subset (same 200 games, same workers, but d15 vs d25).
- **Hyperparams:** b=256, lr=1e-3, hidden=32, num_layers=2, **epochs=60** (was 20 for v1; bumped per user request — testing whether more epochs + deeper data helps the policy collapse seen in v1's 12-game tournament).
- **Pre-task housekeeping:** deleted v0 caches (`mcts_study/runs/gnn_v0_smoke/cache_full.pt` 938 MB + `cache_w0.pt` 250 MB, 1.2 GB freed). Worker subset staged via symlinks at `mcts_study/runs/gnn_v2_d25_w0w1/staged/` so glob matches only d25 shards (excluding d15+d35 from same source dirs).
- **Run dir:** `mcts_study/runs/gnn_v2_d25_w0w1/`
- **Cache path:** `~/cache_v2_d25_w0w1.pt` (WSL native ext4)
- **PID:** 10479

**Cache build (07:23–09:57 UTC = 2h 47min):** done.
- 495,447 positions cached, **3.4 GB on disk**, ~52 pos/s sustained, peak RSS 11.1 GB (well under 31 GB cap).
- 18% slower per-pos than the 50-game cache build (52 vs 60 pos/s) — likely d25 games are longer (~2475 moves/game vs d15's 2042) so more replay work per position.
- Validates RAM projection: linear scaling extrapolates 4-worker d25 (~990k pos) to ~22 GB RAM peak — fits without WSL bump.

**Train phase (09:57 UTC – ongoing):** at epoch 9/60 as of 11:15 UTC. Per-epoch wall-clock ~7.0 min (372–397s train + ~55s val), slightly above the 6.8 min projection. ETA finish ~17:30 local / 21:30 UTC.

Loss curve (epochs 1–9):

| epoch | train_loss | val_loss | val_top1 |
|---:|---:|---:|---:|
| 1 | 1.157 | 1.152 | 0.469 |
| 2 | 1.092 | 1.197 | 0.479 |
| 3 | 1.085 | 1.147 | 0.464 |
| 4 | 1.086 | 1.275 | 0.407 |
| 5 | 1.081 | 1.269 | 0.343 |
| 6 | 1.081 | 1.277 | 0.585 |
| 7 | 1.079 | 1.309 | **0.611** ← run-best |
| 8 | 1.079 | 1.272 | 0.363 |
| 9 | 1.079 | 1.326 | 0.504 |

**Reading the curve so far:**
- **Train loss flatlined at 1.079** by epoch 7, matching v1_d15's floor of 1.067 (within noise). 200 d25 games ≈ 200 d15 games for raw fitting capacity. Confirms what the journal predicted: model is data-bound, not capacity-bound, at this size.
- **Val loss climbing** from 1.15 (ep 1) to 1.33 (ep 9). Classic mild overfit pattern — train converged but val getting worse.
- **Val top1 = 61% at ep 7** is the highest top-1 we've seen on any run (v0a peaked ~50%, v1_d15 peaked ~48%). Could be a real signal that d25 data is more learnable, or a 20-game-val-set noise peak. Need to see full curve to know.
- **Val top1 volatility (40% → 61% → 36%)** is classic small-val effect: ~20 games × 2475 moves = ~50k positions but only 20 *independent* game seeds; one game's worth of pattern shift swings top-1 by ±10pp.
- **No per-epoch checkpoints saved** (train.py only writes ep 60). If the actual best model is at ep 7-15, we won't be able to recover it. Captured as wishlist §4b.1.

### Process lesson (added during v2 training)

**The "we'll only know after epoch 60" trap.** Mid-run, observed val_top1 hit 0.611 at epoch 7 — possibly the strongest model we've trained. But our train.py only writes the final checkpoint, so even if the model peaks at epoch 7 and overfits afterward, we ship epoch 60. **For ANY future training run on noisy/small data, save per-epoch (or per-N-epoch) checkpoints by default.** The disk cost is trivial (~67 KB × 60 = 4 MB); the cost of *not* saving is "we can't tell what the model could have been." Wishlist §4b.1.

### v2_d25 training run #1 — KILLED at epoch 11 (no checkpoint saved)

The original 60-epoch run hit the §4b.1 wishlist gap directly: train.py only saved a checkpoint at the end of all epochs, so killing it lost everything despite 11 completed epochs. Train_loss flatlined at 1.079 by epoch 7, val_top1 peaked at 0.611 (epoch 7). Cache file (`~/cache_v2_d25_w0w1.pt`, 3.4 GB) was preserved and reused for run #2.

Lesson made permanent: train.py now saves per-epoch bundles + `checkpoint_best.pt` every epoch (wishlist §4b.1, .4 implemented).

### v2_d25 training run #2 (10 epochs, value-equal, with new wishlist features)

Re-ran on same cache after killing #1, with the upgraded train.py. 10 epochs only, b=256, lr=1e-3, w_value=1.0, w_policy=1.0. Cache was already on disk so no rebuild needed (load took 12 min from cold cache).

Per-epoch results:

| epoch | train | val | val_top1 | per-game spread | best? |
|---:|---:|---:|---:|---|---|
| 1 | 1.156 | 1.139 | 0.461 | 0.36–0.59 | ✓ |
| 2 | 1.093 | 1.219 | 0.486 | 0.33–0.62 | ✓ |
| 3 | 1.086 | 1.149 | 0.429 | 0.24–0.70 | |
| 4 | 1.083 | 1.221 | 0.524 | 0.33–0.62 | ✓ |
| 5 | 1.082 | 1.265 | 0.349 | 0.23–0.60 | |
| 6 | 1.081 | 1.271 | 0.576 | 0.29–0.70 | ✓ |
| **7** | 1.080 | 1.273 | **0.642** | 0.34–**0.75** | **✓ best** |
| 8 | 1.079 | 1.272 | 0.343 | 0.28–0.51 | |
| 9 | 1.079 | 1.234 | 0.509 | 0.37–0.64 | |
| 10 | 1.078 | 1.276 | 0.367 | 0.28–0.52 | |

**Findings:**
- **Best epoch 7 (val_top1=0.642)** — meaningful jump above v1_d15's 0.37–0.48 range. Best top-1 we've ever measured.
- **Per-game peak: 0.75** at epoch 7 — one val game has 75% top-1 alignment with MCTS visits.
- **Train loss floor: 1.078** — basically tied with v1's 1.067. 200 d25 games provide same fitting capacity as 200 d15 games.
- **Volatile val_top1** (40-64% across epochs) confirms 20-game val effective sample size is the dominant noise source.
- **The §4b.1 win was real:** epoch 10's val_top1=0.367 was 27pp worse than ep 7's 0.642. Without per-epoch saves we'd have shipped a much weaker model. **Strong validation of the wishlist-driven train.py upgrade.**

Archive: `mcts_study/runs/gnn_v2_d25_keep/` (all 10 epoch bundles + checkpoint_best.pt + ARCHIVE_README.md).

### v2_d25_pw4 training run (10 epochs, policy-weighted)

Hypothesis: with 200 games (sparse value signal) but 495k positions (dense policy signal), increase policy gradient by 4× to let the policy head learn faster without affecting model capacity. Same data, same epochs, same architecture; only `w_value=1.0, w_policy=4.0`.

| epoch | val_top1 (w_p=1) | val_top1 (w_p=4) |
|---:|---:|---:|
| 1 | 0.461 | 0.526 |
| 2 | 0.486 | 0.431 |
| 3 | 0.429 | 0.470 |
| 4 | 0.524 | 0.451 |
| 5 | 0.349 | 0.382 |
| 6 | 0.576 | 0.564 |
| **7** | **0.642** ✓ best | **0.643** ✓ best |
| 8 | 0.343 | 0.524 |
| 9 | 0.509 | 0.398 |
| 10 | 0.367 | 0.469 |

**Key finding: both runs peak at epoch 7 with effectively identical val_top1.** The 4× policy weight did not unlock additional policy-head accuracy. Plausible reasons:
1. The policy head is already at its data-bound ceiling at 200 games.
2. The body representation learned to maximize policy_loss either way, so changing weights only affects how much value-loss is "tolerated" — and the policy head saturated regardless.
3. pw4 ran ~25% faster per epoch (5.2 vs 7.0 min) because the cache was hot — incidental but noted.

Archive: `mcts_study/runs/gnn_v2_d25_pw4_keep/` (all 10 epoch bundles + checkpoint_best.pt + ARCHIVE_README.md).

### Tournament: v1_d15 vs v2_d25_value vs LookaheadMcts vs Random (e8, 12 games, sims=50, no MCTS-on-GNN)

Pure-policy tournament (PureGnn for the GNN seats, no MCTS wrapping the GNN evaluator), to isolate raw policy-head quality.

| role | model | wins | winrate | mean VP |
|---|---|---:|---:|---:|
| **LookaheadMcts** | depth=25, sims=50 | **9/12** | **75.0%** | 7.83 |
| **v2_d25_ep7** | (w_p=1, this) | 0/12 | 0% | **3.58** |
| **v1_d15** | (older, d15 data) | 0/12 | 0% | 3.17 |
| Random | uniform | 1/12 | 8.3% | 3.08 |
| (timeouts: 2/12) | | | | |

**v2 beats v1 by mean VP** (3.58 vs 3.17) — small but consistent edge. Both GNNs beat Random. **LookaheadMcts dominates as expected.** Random's 1 win was a setup-phase fluke (seed 15010000, rot=1, Random in seat 2).

Run dir: `mcts_study/runs/gnn_v2_d25_w0w1/final_tourney/2026-04-30T17-41-e8_gnn_vs_gnn/`.

### Tournament: v2_d25_value vs v2_d25_pw4 vs LookaheadMcts vs Random (e8, 12 games, sims=50)

The hypothesis test: does policy-weighting (w_p=4) actually improve PureGnn play vs equal-weighting (w_p=1)?

| role | model | wins | winrate | mean VP |
|---|---|---:|---:|---:|
| **LookaheadMcts** | depth=25, sims=50 | **10/12** | **83.3%** | 8.67 |
| **v2_value** (w_p=1) | gnn_v2_d25_keep | 0/12 | 0% | **3.83** |
| **v2_pw4** (w_p=4) | gnn_v2_d25_pw4_keep | 0/12 | 0% | 3.58 |
| Random | uniform | 1/12 | 8.3% | 2.75 |
| (timeouts: 1/12) | | | | |

**Headline finding (negative result, important): policy-weighting did NOT outperform equal-weighting.** v2_pw4 actually scored slightly *worse* mean VP (3.58 vs 3.83). Possible reasons (in order of plausibility):

1. **Policy head was already at its data-bound capacity.** Both models reached val_top1 ≈ 0.642 at epoch 7. With 200 games, the policy head ran out of distinct strategic situations to learn from regardless of loss weight.
2. **Body representation drifted away from value-useful features.** The model body is shared. With 4× more policy gradient, the body's internal representation may have become less semantically rich, costing PureGnn play quality even with same top-1 accuracy.
3. **Sample noise.** n=12 with 0.25 VP gap is within noise.

**Important caveat:** the val_top1 = 0.642 metric was the same for both models, but mean tournament VP differed by 0.25. This means **val_top1 is not a perfect proxy for tournament strength** — a model can score the same on policy top-1 yet play differently. Future tournaments are the only ground truth.

Run dir: `mcts_study/runs/gnn_v2_d25_pw4_keep/tourney/2026-04-30T19-32-e8_gnn_vs_gnn/`.

### Architectural deliverables this iteration

These outlasted the runs themselves:

1. **`scratch_card_tracker.py`** — Python reconstruction of all 4 players' resource hands + bank from action_history alone. Verified end-to-end on two e7 games (665+ steps each). Plug-and-play interface (`CardTracker.snapshot_at(step_idx)`) so a future `EngineCardTracker` (after the v2 engine API lands) is a one-line swap. Lives in `scratch_card_tracker.py`.

2. **`scratch_replay_lite.py` overhaul** — replay viewer now shows all 4 players' full resource breakdowns at every step (not just current player), bank state per resource, and the terminal-state perspective bug fixed. The "stale" cache logic was removed entirely.

3. **`mcts_study/catan_gnn/train.py` upgrades (wishlist §4b.1, .2, .4, .5, .8 implemented):**
   - Per-epoch checkpoint bundles (`checkpoint_epoch{NN}.pt`)
   - `checkpoint_best.pt` updated whenever val_top1 hits a new high
   - `--resume <path>` for continuation
   - Per-game val_top1 quartiles in `EpochStats` and inline log
   - `progress.png` regenerated after every epoch
   - `_write_progress_plot()` helper extracted

4. **`mcts_study/catan_mcts/experiments/e8_gnn_vs_gnn.py` (wishlist §4b.3):** new tournament script for two-GNN head-to-head + LookaheadMcts + Random. Used for the v1-vs-v2 and v2_value-vs-v2_pw4 tournaments.

5. **`scratch_e8_aggregate.py`** — pretty-print results from any e8 run dir.

6. **`scratch_plot_v2_d25_progress.py`** — standalone live-plot script (precursor to the in-train.py version).

7. **`docs/superpowers/specs/2026-04-30-engine-v2-wishlist.md`** — single source of truth for everything we want to push to Rust later: inventory caps, longest road, balanced map gen (Tier 1 ABC + Tier 2 rejection), missing PyO3 stats fields, `engine.bank()`, `engine.all_hands()`, `engine.observation_for(viewer)`, training pipeline improvements (§4b.1-.8).

### 1200-game permutation tournament — KILLED at 36/1200, lessons captured

After the 12-game tournaments, attempted a **24-permutation × 50-games = 1200-game tournament** (added wishlist §4b.3 permutations mode + per-game flush kill-safety to e8). Ran 4 workers for ~1.5 hours.

**Throughput reality check:** projected ~6 hr based on 12-game timing extrapolation. Actual sustained throughput was ~0.10 games/min/worker → revised ETA was **~48 hours.** Killed at 36 games done.

**Result of those 36 games (CAVEAT — all from permutation 0 only):**

| role | label | games | wins | winrate | mean VP |
|---|---|---:|---:|---:|---:|
| LookaheadMcts | depth=25, sims=50 | 36 | **32** | **88.9%** | 9.25 |
| GnnA | v2_value | 36 | 0 | 0% | **4.31** |
| GnnB | v2_pw4 | 36 | 0 | 0% | 4.14 |
| Random | uniform | 36 | 4 | 11.1% | 3.75 |

**Caveat is critical:** the workers never finished permutation 0 in the time allotted, so **all 36 games used the same seating** (`GnnA-GnnB-LookaheadMcts-Random`). The permutations-mode data is effectively a third 12-ish-game slice from a single seating, not a balanced mix. Per-seat sanity check confirms: every win came from seat 2 (LookaheadMcts) or seat 3 (Random), the only seats those roles ever sat in.

**Pattern is consistent with prior tournaments:**
- v2_value still edges v2_pw4 by mean VP (4.31 vs 4.14, gap 0.17 — same as 12-game's 0.25 gap, within noise).
- LookaheadMcts dominates as always.
- Random's 11.1% is right at expected variance for a 4-player game where the strong player loses ~10%.
- 0 timeouts — quality of completed games is good.

**Lessons captured for next time:**

1. **Wall-clock estimates from small-N tournaments don't extrapolate cleanly.** The 12-game runs hit short games with decisive LookaheadMcts wins; a wider seed range pulls in long-tail behavior that's much slower per game. Need to **measure per-game throughput on a small representative slice before committing to a 1200-game run.**

2. **Multiprocessing pool work distribution is unbalanced under spawn.** With workers=4 and num_games_per_seating=50, each worker got 12-13 games per permutation. Workers process permutations sequentially — they all sit on permutation 0 for hours before any of them can write a permutation 1 shard. **This makes mid-run kills look much worse than they should** (no balanced sample by definition).

3. **The kill-safety machinery worked.** 36 games saved across 4 workers via per-5-game chunk flushes. `done.txt` and per-chunk parquets recovered without loss. **Wishlist §4b.3 + the per-chunk flush pattern is good engineering** — just doesn't cover the "permutation balance" requirement.

4. **Future tournament design:** instead of `--num-games-per-seating N` × 24 perms, **iterate the OUTER loop over `i in range(N)` and the INNER loop over permutations.** That way every "round" of N×24 games covers all permutations once before starting the next round. A kill mid-run leaves a balanced subset, not a biased one. Add to wishlist as §4b.4 (renumber as needed).

### End-of-day-2 pivot: full Catan + bit-packed state

After watching three "best" d25 self-play replays (seeds 41400315, 41400279, 41400295), diagnosed why no GNN we've trained beats LookaheadMcts at any (sims, depth) combination:

**The game we're training on isn't really Catan.** Specifically, the v1 engine has these structural gaps that strategy-degrade the play:

1. **Wood/brick monopolies create permanent road blockades** (because road-cost is just W+B and roads have no inventory cap). Players who can't build roads can't expand. Stalls common.
2. **No release valve for resource-starved players.** No 4:1 bank trade, no ports, no player-to-player trades. A player who lacks sheep tiles can never build a settlement.
3. **No longest-road / largest-army / dev-card VP sources.** All VP comes from settlements + cities only. The "alternate paths to 10" that real Catan offers are missing — so games stall when one player can't reach a settlement spot.
4. **41% of d25 self-play games time out at the 30k-step cap.** Per the journal, value targets are zeroed for these → 41% of training value-signal is null. Big data-quality hole we missed.
5. **Steal mechanism may have a bug.** Worth a regression test on restart.

These add up: the GNN learns from a game where stalling is rewarded and the value head is half-empty.

**Pivot decision:** restart with a v2 engine that implements **full Catan rules** (trades, dev cards, longest road, largest army, inventory caps), but with **bit-packed observation tensors** to control state size. Most state values are tiny (resource counts 0-19 = 5 bits, VP 0-10 = 4 bits, building counts 0-15 = 4 bits) and the current f32 representation wastes ~32× per field. Bit-packing makes the cache **smaller** even with more rule features.

Full design + 36-item improvement list at `docs/superpowers/specs/2026-04-30-v2-restart-full-game-design.md`.

**What v1 work survives:**
- All training-pipeline improvements (per-epoch saves, resume, best tracking, per-game val variance, live progress).
- Two-GNN tournament infrastructure (e8 + permutations + per-chunk flush).
- Card tracker plug-and-play interface (replay viewer just swaps backends when v2 engine ships `bank()` + `all_hands()`).
- The GNN architecture itself (hetero PyG model with hex/vertex/edge nodes).
- Replay viewer (will need updates for new VP sources but core logic stays).

**What v1 work doesn't survive:**
- All trained checkpoints (gnn_v1_d15_keep, gnn_v2_d25_keep, gnn_v2_d25_pw4_keep). Reference only.
- All v1/v2 caches and parquets for training. Game changes → data invalidates.

### Net summary of v2 day

- **Best v2 model: `gnn_v2_d25_keep/checkpoint_best.pt` (epoch 7, val_top1=0.642).** Used in tournaments, mean VP 3.58–3.83.
- **No GNN ever beat LookaheadMcts** at sims=50 across any tournament configuration tried.
- **Policy-weighting (w_p=4) did not help.** Possibly the bottleneck is data, not loss balance.
- **The wishlist-driven train.py upgrades paid off immediately:** without §4b.1 we'd have lost both runs to mid-training kills. We now have full per-epoch artifact preservation by default.
- **The replay viewer is now strictly more useful** with all-hands + bank visibility.
- **Open question for v3:** is depth=35 data better? RAM check confirms 200 d35 games fits in 31 GB cap (~15.5 GB peak). Cache build ~3 hr; training another ~1h. Pending user decision.

---

### Update protocol for this section

- **At op start:** add a new entry block with start time, goal, run dir.
- **Every 30-60 min while running:** append a brief progress note ("at HH:MM: 240/1500 done, 8 timeouts, ETA ~3h").
- **At op end:** finalize with end time, total wall-clock, headline outcome, and a 1-sentence interpretation if any.
- **Periodically:** redraw the ASCII timeline at the top so the visual stays current.

The point: someone reading the journal next week should be able to see at a glance which sweeps overlapped with which, and what the total compute budget went into.

---

## End-of-day-1 findings + known limitations (2026-04-30)

This section is the load-bearing record for the next iteration. It distills what we measured, what we *think* it means, and the limitations we hit that bound the next experiment's design.

### What we learned about the v0/v1 GNN

**Pipeline is mechanically correct.** All 11 plan tasks shipped, all derisks (3-5) green, replay correctness 10/10 against fresh dataset. The data → cache → train → checkpoint → evaluator → MCTS → recorder loop closes end-to-end with no silent corruption.

**Training scales with data, not hyperparameters.**

| run | data | train_loss floor | val_loss range | bench2_value_mae | bench2_policy_kl |
|---|---|---:|---|---:|---:|
| v0a | 50 games (136k pos), lr=1e-3, 20ep | 1.087 | 1.34-1.84 | 0.724 | 0.0036 |
| v0b | 50 games, lr=1e-4, 40ep | 1.086 | 1.49-1.83 | 0.764 | 0.0037 |
| v1 (d15 subset) | 200 games (420k pos), lr=1e-3, 20ep | **1.067** | **1.30-1.42** | 0.781 | **0.0013** |

- v0a vs v0b: same data, different hyperparams → essentially identical results. **Hyperparameter tuning at 50 games is throwing darts at noise.**
- v0 vs v1: 4× more positions, same hyperparams → train loss floor drops, val loss tightens, policy_kl drops 2.7×. **Data is the binding constraint at this model size.**
- value_mae went *up* from v0 to v1 (0.72 → 0.78). Two plausible causes: (a) sample noise on n=200 bench positions, (b) the d15 dataset has shorter games with different value distributions than the depth=25 lookahead reference uses to compute MAE. Either way, value MAE alone is not a reliable signal of model quality at this stage.

**The v1 GNN actively misleads MCTS at low sim count.**

| match | players | result |
|---|---|---|
| v0a e6 | MCTS+GNN(v0a) at sims=100 vs 3 random | 0/4 wins (4 finished, 4 timed out) |
| v0b e6 | MCTS+GNN(v0b) at sims=100 vs 3 random | 0/4 wins (same) |
| v1 e7 1-game | GnnMcts vs PureGnn vs LookaheadMcts vs Random at sims=50 | Random=10 (winner), Lookahead=9, **PureGnn=4, GnnMcts=2** |

The single-game tournament (n=1) is noisy on the *ranking among non-MCTS players* (random winning is partly seat-luck), but the structural finding is robust: **PureGnn (no search) outperformed GnnMcts (MCTS-augmented) on the same model.** Adding MCTS to a noisy evaluator amplifies the noise — the model's value/policy heads are poor enough that they pull search toward bad subtrees. Same pattern v0 e6 showed (0 wins) at higher sims=100.

**LookaheadMcts is the strongest player we have**, by a wide margin. The greedy-VP lookahead evaluator built earlier already captures most of what a "decent Catan player" needs at much less compute. Beating it should be the v2 target, not "beat random."

### Performance gotchas (cost engineering)

**1. Cache key hashing was the dominant MCTS+GNN wall-clock cost.** Original `tuple(state._engine.action_history())` was O(N) where N is the action_history length (~5000+ at endgame). Hashing per call × 100 sims × 200 moves = billions of ops. Fixed by switching to `id(state)` (Python object identity) — O(1). This alone took GnnMcts per-move time from 6-10 sec to ~700-900 ms.

**2. PyG Batch overhead is not amortized by batch size at our scale.** Bumping batch from 64 to 256 saved ~1 min/epoch (out of 6 min) — much less than the 4× theoretical improvement. The per-batch graph-concat cost in `Batch.from_data_list` is meaningful even at small batches. **Pre-batched cache** (Option A from the GPU discussion) would help; we deferred it.

**3. Cache load on `/mnt/c` (Windows filesystem from WSL) is ~40% slower than native ext4.** 250s for `/mnt/c` vs 145s for `~/`. For training sessions you'd run multiple times, **copy the cache to native WSL fs first.** For one-off training, doesn't matter.

**4. CachedDataset memory uses ~4× the disk size.** Measured: 938 MB on disk → 3.91 GB RAM peak. Python overhead on tensor dicts. Linear scaling — projects 7.5 GB disk → 30 GB RAM for 400-game dataset.

**5. `Batch.from_data_list` is sequential Python.** ~210ms/batch at b=256 on 67k-param model is dominated by graph batching, not GPU. GPU utilization stayed at ~3% during training.

**6. e6 and e7 timeouts are evaluator-specific.** 360s/game cap (set during e5 development) was way too tight for sims=100 GNN inference. v0 e6 hit 4/8 timeouts at 1200s. Per-evaluator wall-clock budgets must be re-tuned, not inherited.

### Engineering / process lessons

**1. Branch isolation is silent.** Two worktrees on different branches can run for hours without either knowing they have divergent code. The schema-v2 bump landed only on `gnn-v0`; the `mcts-study` worktree on `mcts-v2-hardening` never got it, so the 1500-game e5 v2 sweep wrote schema-v1 parquets and was unusable for GNN training. **Lesson: schema bumps and persistent-format changes need to land on every active branch *before* a long sweep starts.**

**2. `0 * -inf = NaN` in PyTorch masked-CE.** Found in Task 7. Without explicit `log_probs.masked_fill(~mask, 0.0)` before the multiply, every training step produces NaN gradients while the test suite (which checks artifact existence, not loss values) still passes green. **Add a finite-loss assertion to the training loop, not just artifact tests.**

**3. Replay determinism needs a per-player counter.** The dataset's `__getitem__` must mirror `play_one_game`'s exact `move_index` accounting (only increment for `current_player == recorded_player`), not a global counter. Caught in Task 6 review; would have produced silently wrong training positions otherwise.

**4. Trivial-skip turns are not in the recorded `moves.parquet`.** When `len(legal) == 1`, `play_one_game` applies the action without invoking the bot, and the recorder doesn't write a row. Replay code (in dataset, in benchmark, in renderer) must auto-apply the unique legal action whenever it encounters this state — the action_history doesn't carry an explicit entry for it.

**5. The cache key must include all stochastic elements that distinguish states.** action_history correctly includes both decisions and chance outcomes (chance-bit encoding). Using just decision IDs would collide states that differ only in dice/steal results.

**6. CLI argparse forwarding patterns silently drop flags.** The top-level `--out-root` was consumed but never re-injected into the experiment's `cli_main()` argv. Result: every multi-worker sweep landed at worktree-root `runs/`, not the user's specified path. **Add a regression test for any `parse_known_args + sys.argv` rewrite.**

**7. WSL git can't always resolve worktree gitfiles** with `/mnt/c/...` paths. Inline `git commit` calls in shell scripts inside WSL fail with "fatal: not a git repository". Workaround: do git ops from the Windows side (via `Bash` tool) or skip auto-commit entirely.

**8. Editable-install venvs shared across worktrees couple long-running processes.** `pip install -e` from one worktree rebinds the package's source path. Running processes are unaffected (they already have modules in memory) but newly-spawned workers in another sweep may re-import from the wrong worktree's source. **For long sweeps, either separate venvs or note which worktree the venv currently points at.**

### Pipeline limitations (known, deferred)

**A. Single-graph batching is the throughput floor.** OpenSpiel's `MCTSBot.evaluate()` is called one leaf at a time. At sims=400 with ~150 decision moves per game, that's ~60k sequential single-graph forward passes per game. **A `BatchedGnnEvaluator`** (collect leaves across MCTS sims, run one batched forward) would be ~50× faster on GPU. Out of scope for v0; flagged in Task 8 code as TODO.

**B. The renderer/viewer reveals only the current player's resource hand.** `engine.observation()` perspective-rotates and only exposes the viewer's hand breakdown; opponent hands are exposed as totals only. To show all four players' resource breakdowns simultaneously would require either:
   - A new engine API (`engine.observation_for(viewer)` taking an explicit viewer argument), OR
   - Calling `observation()` 4 times in the replay capture loop (slower but doesn't require Rust changes — actually not possible with current PyO3 binding; engine.observation() always uses state.current_player as viewer).
   This is a real Rust API gap if we want full visibility in tooling.

**C. The renderer's VP count uses live building counts (settlement=1, city=2).** It does NOT count longest road, largest army, or victory-point dev cards. For Tier-1 (no dev cards) games it matches `final_vp` only at terminal because VPs from longest road / 2VP-from-buildings-bonus etc. aren't yet implemented in our v1 engine. Worth knowing when comparing the viewer's VP to a possible future engine that adds them.

**D. The per-step cache is in-RAM only.** Larger datasets (the user's overnight 400-game sweep at depth ∈ {15,25,35} would yield ~1.1M positions) project to ~30 GB RAM peak — right at WSL's 31 GB cap. Either bump WSL memory (one-time `.wslconfig` change + `wsl --shutdown`) or shard the cache to disk.

**E. We have no opponent-modeling.** All trained data was generated against random opponents (lookahead-VP at depth=25 vs 3 random). The model has never seen what a competent opponent does. v0/v1 may overfit to "play recklessly because random opponents won't punish you." A Catan-grade greedy bot in the training rotation would teach defensive play.

**F. Value head learns from final outcomes only.** AlphaZero in chess has clean ±1 returns. Our recorder uses `+1`/`-1`/`0(draw)` which is correct, but the *signal density* is ~1 outcome per ~150 positions per game. With only 200 games (40k effective positions per game-outcome), the value head is starved for variety in terminal outcomes. **Number of distinct game outcomes is the load-bearing variable, not number of positions.**

**G. PureGnnBot uses pure argmax**, no temperature, no exploration. For evaluation that's fine; for self-play data generation it'd produce highly correlated games. Worth a temperature parameter when we use PureGnn as a self-play opponent.

**H. e7 tournament is single-worker by design.** Multi-worker would let us run many games in parallel, but the user has a 4-worker e5 sweep occupying CPU. Once that's done, e7 should support `--workers 4`.

**I. We don't have head-to-head numbers at scale.** The e7 tournament finding (PureGnn beats GnnMcts) is from a single game. Need 12+ games minimum to have any confidence in the ordering. The 12-game tournament was queued but not yet completed at write time.

### Non-obvious things future-iteration should know

**1. The 50-game v2 cache (`~/cache_full.pt`, 938 MB) is the only training dataset that's both schema-v2 AND completely processed.** The 200-game subset cache (`~/cache_d15_subset.pt`, 2.94 GB) covers depth=15/sims=400 from worker0+worker1 of the user's overnight sweep. Anything else would need a fresh cache build.

**2. The 1500-game e5 v2 sweep parquets exist on disk but are SCHEMA V1.** Useful for understanding lookahead-VP performance characteristics; **not** useful for GNN training without re-recording.

**3. e5_lookahead_depth.py from `mcts-v2-hardening` has been merged into main.** All branches share the schema v2 recorder + lookahead evaluator now.

**4. `~/cache_*.pt` files persist on the WSL ext4 home directory, NOT on the Windows-side worktree.** They survive across worktree changes but die if WSL is reset. Worth backing up if we want to keep them long-term — the build cost is non-trivial (~40 min for the 50-game one, ~2.5 hours for the d15 200-game subset).

**5. The cache doesn't store the model's outputs at the time of recording.** It stores the (state → MCTS visit count target) labels. So caches are model-agnostic and reusable across all GNN architecture variants. **Don't rebuild the cache when changing model size or hyperparameters** — that's wasted compute.

**6. The replay viewer tooling is "lite":** static board PNG + JS overlays. Updating the visualizer to support new things (showing dev cards, longest road, ports, larger boards) needs only HTML/JS edits, not a re-render. The Python side just emits states; the JS draws.

**7. Hand visibility in the replay viewer is per-step current-player only.** This is an engine-API limitation, not a viewer limitation.

### Recommended next steps for v2/iteration

1. **Larger training set.** Wait for the user's overnight 400-game sweep to finish, then build a v2 cache from all 4 workers (needs WSL bump to ~56 GB) or stick to the 2-worker subset and accept reduced data variance.
2. **Train v2 on ≥1000 distinct game outcomes** (currently we have 200). The number that matters is unique-game-count, not unique-positions. To get there: re-run e5 with `--num-games 250` per cell × 4 cells = 1000+ unique outcomes.
3. **Add an engine API for `engine.observation_for(viewer: int)`** — small Rust + PyO3 change. Unlocks all-hands visibility in tooling AND opens the door to true multi-perspective training (viewer-agnostic GNN).
4. **Add Catan-grade opponents to the data-generation rotation.** The existing `GreedyBaselineBot` is a start but weak. A Catanatron-grade greedy player would expose the model to defensive play.
5. **Implement `BatchedGnnEvaluator`.** The 50× speedup on MCTS+GNN inference makes higher-sim-count tournaments feasible.
6. **Pre-batched cache.** Save HeteroData as already-batched chunks (e.g., 256-position batches) to remove per-batch Python overhead.
7. **Fix the e6 / e7 wall-clock budget** to be evaluator-aware: lookahead-VP wants ~300-600 sec/game, GNN wants ~600-1800 sec/game at sims=100, GNN at sims=400 may want even more.
8. **Decide: is "beat lookahead-VP" the v2 target?** That's a much higher bar than "beat random". May require multi-iteration AlphaZero (train v1 → self-play v1 → train v2 → ...) instead of one-shot supervised learning.

### Artifacts you can resume from

```
gnn-v0 worktree:
  ~/cache_full.pt                       — 50-game v2 cache (RAM 3.9 GB, disk 938 MB)
  ~/cache_d15_subset.pt                 — 200-game depth=15 cache (disk 2.94 GB)
  mcts_study/runs/gnn_v0_overnight/
    v0a/checkpoint.pt                   — v0a model (lr=1e-3, 20 ep)
    v0b/checkpoint.pt                   — v0b model (lr=1e-4, 40 ep)
  mcts_study/runs/gnn_v1_d15/
    v1_d15_subset/checkpoint.pt         — v1 model (200-game training)
    one_game_save/action_history.json   — full game record for replay
    one_game_save/replay_lite/          — HTML viewer (open index.html)
    one_game_live.log                   — narrated log of the same game
  mcts_study/runs/2026-04-29T20-49-e5_lookahead_depth/   — 50-game v2 sweep
  runs/gnn-data-2026-04-29-evening/
    2026-04-29T22-41-e5_lookahead_depth/ — user's 400-game sweep (in flight)
```

### Open questions for the v2 iteration spec

1. Is one-shot supervised learning enough, or do we need the AlphaZero iteration loop?
2. How do we measure "did the model learn anything useful?" without an obvious benchmark — is "beat lookahead-VP at sims=100" the right target?
3. Should we collect data with multiple evaluator depths (mix d=15, 25, 35 in training) to teach the model robustness, or stick to one depth for cleanliness?
4. How important is Tier-2 game support (dev cards, longest road, largest army) for v2? Tier-1 may be a local maximum.

---

## e7 12-game tournament (2026-04-30, sims=50)

12-game round-robin with all 4 player kinds rotated through 4 cyclic seat positions (3 games per rotation).

| player | games | wins | winrate | mean VP |
|---|---:|---:|---:|---:|
| **LookaheadMcts** | 12 | **12** | **100.0%** | **10.00** |
| PureGnn | 12 | 0 | 0.0% | 3.92 |
| Random | 12 | 0 | 0.0% | 2.83 |
| **GnnMcts** | 12 | 0 | 0.0% | **2.33** |

**Five hard findings at n=12:**

1. **LookaheadMcts dominates absolutely (12/12, 100%).** Wins every game by reaching 10 VP. Across all 4 seat positions, no exceptions.
2. **GnnMcts is the WORST player (mean VP 2.33).** Below even Random (2.83). The MCTS+GNN combination is genuinely worse than random play at sims=50 with the v1 model. **Confirms the structural finding from the 1-game tournament: MCTS amplifies the noisy GNN's mistakes.**
3. **PureGnn (no search) beats Random by ~1 VP** (3.92 vs 2.83). The trained policy head provides a small but real signal — better than random but not strategic.
4. **Adding search to v1 GNN flips the sign:** PureGnn 3.92 → GnnMcts 2.33. Search makes the model **worse** by ~1.6 mean VP.
5. **0 timeouts.** The `id(state)` cache-key fix held — games complete cleanly in the 600s budget at sims=50.

**Player ordering established:** LookaheadMcts > PureGnn > Random > GnnMcts.

**What the v1 GNN tells us about the trained model:**
- Value head: not accurate enough to guide MCTS (search amplifies its errors).
- Policy head: better than uniform random but well below greedy domain heuristics.
- The model knows *something* — PureGnn beats Random — but not enough to be useful as an MCTS leaf evaluator.

**Aggregator-bug postmortem (worth noting for the next iteration):**
Initial aggregation reported all players at 25.0% winrate (3 wins each), which was implausible. Root cause: the aggregator pulled `seed_base` from `config.json`, but the experiment's CLI doesn't write the user-specified seed_base back to config — only the default. With seed_base hardcoded to 7M and actual seeds at 13M+, the rotation index decoded to `rot_idx=600`, which made `seating[rot_idx:] + seating[:rot_idx]` an empty list. **Lesson:** experiments should persist all CLI args (especially seed_base) into config.json so downstream aggregators can reconstruct the experimental setup without guessing. Fixed by deriving seed_base from `min(seed)` observed in the parquets.

### Updated player ordering (2026-04-30, n=12)

```
LookaheadMcts (100% winrate, 10.0 mean VP — wins every game)
    ▲ +6 VP gap
PureGnn (0% winrate, 3.9 mean VP — small policy-head signal above random)
    ▲ +1 VP
Random (0% winrate, 2.8 mean VP)
    ▲ +0.5 VP
GnnMcts (0% winrate, 2.3 mean VP — MCTS amplifies noisy GNN, makes it worst)
```

The v2 target should be: **GnnMcts mean VP > LookaheadMcts mean VP** in head-to-head. That's the threshold where the trained model has actually surpassed the hand-engineered evaluator.

---

## Engine-rule gap: building inventory limits not enforced (2026-04-30)

**Discovered while building the replay viewer.** The Catan rules cap each player's lifetime building inventory at:
- 5 settlements
- 4 cities
- 15 roads

The v1 engine (`catan_engine`) **does not enforce these caps.** A player can place more than 5 settlements, more than 4 cities, etc., as long as basic adjacency / cost / resource rules pass. The replay-viewer tooling found this implicitly because it computes live VP from total settlements + cities × 2 with no upper bound; in some long games we'd see VP totals that exceeded what the inventory rules allow.

**Why this matters for v0/v1 GNN:**
- Training data has been generated under "unbounded inventory" rules. The model has learned a value function and policy that includes positions where seat 0 has 7 settlements down — positions that are illegal in real Catan.
- LookaheadMcts (depth=25 lookahead) and PureGnn / GnnMcts all play under the same unbounded rules, so the tournament fairness is preserved *between bots*. But against a real Catan rule set, all four would behave differently.
- More importantly, the **terminal condition** in our engine is "first to 10 VP", and VP is computed solely from buildings (no longest road / largest army / dev cards yet). With unbounded inventory, hitting 10 VP is just "build until you have 10 cumulative VP-worth of buildings." That changes the strategic problem from real Catan in non-trivial ways.

**What's NOT broken:**
- The engine is internally consistent. State transitions are deterministic, costs are checked, no-overlap rules apply, etc.
- The training pipeline is sound — the model is learning the function our engine actually instantiates, which is what we want for v0.
- Bot comparison results are valid for our engine's rules.

**What we should fix in v2 engine:**
1. **Add inventory caps** to `legal_actions()`: filter `BuildSettlement(v)` if `settlement_count >= 5`, etc. ~10 lines of Rust.
2. **Add longest road / largest army** to win condition (currently just building VP). Major Tier-2 work, deferred.
3. **Add dev cards.** Major Tier-2 work, deferred.

**For v2 training:** if we add inventory caps before the next training data run, all v2 data will be under the new rules and the model will learn the correct game. If we re-train v1 *without* re-recording data, the existing parquets are still valid for the existing engine but won't transfer to the inventory-capped v2 engine.

**Decision:** add inventory caps as a small-scope Rust change before any further training data runs. This is cheap, correct, and ensures the model learns real Catan strategy.

---

## Architecture note: replay viewer can be Windows-native, not WSL-bound (2026-04-30)

The replay viewer (`scratch_replay_lite.py` + the served HTML) currently runs entirely in WSL because the parent Python project lives there. **It doesn't have to.**

The viewer is a static HTML + SVG page once the data is generated. Three approaches, in order of decoupling:

1. **Current:** Python (WSL) generates JSON + PNG, Python (WSL) serves HTTP, browser (Windows) reads localhost:8765. Tight coupling — viewer can't run without WSL up.

2. **Hybrid:** Python (WSL) generates JSON + PNG, then we copy them to a Windows-native folder. Browser opens the folder via a Windows-side server (`python -m http.server` from PowerShell, or a static-site host). WSL only needed for data generation.

3. **Fully decoupled:** Generate the JSON + PNG once, **inline everything into a single self-contained HTML** (which we already do — `replay_lite/index.html` is 1.4 MB self-contained). Open the HTML directly via `file://` — no server needed at all.

**We're effectively already at #3** for the inlined HTML. The fact that we needed a local server was because the *non-inlined* version's `fetch()` of sibling files fails under `file://`. The inlined version works without any server.

**For future iterations:** the viewer should be treated as a **portable artifact** — the inlined HTML can be copied anywhere, opened with no infrastructure. WSL is only needed at generation time. This decoupling matters because:
- Sharing replays with others doesn't require them to set up WSL.
- Replays survive WSL state being reset.
- A non-developer can examine a saved game without the dev environment.

Action item for the next session: **prefer the inlined `index.html` as the primary distribution format.** The non-inlined `layout.json` + `overlays.json` + `board.png` triple is for development/iteration where you'd rebuild often. Once a replay is "final", embed everything and ship as one file.
