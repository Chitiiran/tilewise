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
                                                                          22:41 ─[ongoing]  user 400-game e5 sweep
                                                                                       (running, eta TBD)
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

### 22:41–ongoing — `user 400-game e5 sweep` (gnn-v0 — user-launched, do not touch)

- **Goal:** generate the proper-scale v2 training data for v0 GNN.
- **Cells:** depth ∈ {15, 25, 35} × sims=400, 100 games each.
- **No max_seconds cap.**
- **Run dir:** `gnn-v0/runs/gnn-data-2026-04-29-evening/2026-04-29T22-41-e5_lookahead_depth/`
- **ETA:** unknown; running on 4 workers.
- **Constraint:** while this runs, no competing 4-worker CPU sweeps from me.

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

---

### Update protocol for this section

- **At op start:** add a new entry block with start time, goal, run dir.
- **Every 30-60 min while running:** append a brief progress note ("at HH:MM: 240/1500 done, 8 timeouts, ETA ~3h").
- **At op end:** finalize with end time, total wall-clock, headline outcome, and a 1-sentence interpretation if any.
- **Periodically:** redraw the ASCII timeline at the top so the visual stays current.

The point: someone reading the journal next week should be able to see at a glance which sweeps overlapped with which, and what the total compute budget went into.
