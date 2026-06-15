# AZ Distillation + Difficulty-Tiered Web Bots — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline — long WSL compute runs need babysitting from the main session). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship difficulty-tiered GNN Catan bots in the existing web interface, then run one (or two) AlphaZero-style policy-improvement iterations overnight — distill the GnnMcts@200 teacher (51.7% vs LookV3) into a search-free PureGnn student via sharpened visit-count targets.

**Architecture:** Phase 1 wires curated difficulty presets into `bot_registry`/`server.py` using *existing* checkpoints (deliverable-safe before any long compute). Phase 2-3 run the already-built async batched self-play stack (`self_play_async.py`, full-Catan defaults) to generate a teacher corpus, then train a student with a new `--policy-sharpen` (visits², the journal-recommended distillation target) and gate it head-to-head vs LookV3. Phase 4 iterates if the gate passes.

**Tech Stack:** Rust engine (PyO3/maturin), PyTorch + PyG GNN (h128/L4), FastAPI web app, WSL Ubuntu (54 GB RAM, GTX 1650 4 GB), async batched MCTS evaluator.

**Key prior facts (sourced):**
- GnnMcts@200 (Cell 6 net) = 51.7% vs LookV3 31.7%, 120-game async control (`2026-06-01-puregnn-deploy-investigation-FINAL.md` §2).
- Raw argmax PureGnn ≈ 8–18%; cheap search (sims 8/16/32) is WORSE (the valley). Never ship mid-sims tiers.
- Distillation target form: **sharpened visits² renormalized** (recommended, FINAL journal §4); hard-argmax and +value-aux are fallbacks.
- Self-play data MUST be full Catan (`vp_target=10, bonuses=True`) — v3 rules actively mis-train (`2026-05-27-fullcatan-deep-behavioral-analysis.md` §"Practical implications").
- Gate measurement: head-to-head async harness, <5% timeouts, never val_top1 (`feedback_use_headtohead_not_midtournament`, `feedback_val_top1_misleads_under_loss_aug`).
- Teacher net: `runs/v3/rl_checkpoints/round0_Cell6.pt` (h128, L4).

---

## Task 0: Worktree environment setup

**Files:** none (environment only)

- [ ] Create `mcts_study/runs` dir in worktree; symlink `runs/v3` → `/home/chitii/catan_data/runs/v3` (WSL-side `ln -s`; Windows sees broken link, that's expected).
- [ ] In WSL venv (`~/catan_mcts_venvs/mcts-study/`): `maturin develop --release` from the worktree (editable PyO3 install points at whatever worktree built last — memory `feedback_worktree_swap_breaks_pyo3_install`).
- [ ] Re-point editable installs (`pip install -e`) at the worktree if `pip show` reveals a stale path.
- [ ] Baseline: `pytest mcts_study/tests -x -q -m "not slow"` green before any edits.
- [ ] Add `.claude/worktrees/` + `.superpowers/` to `.gitignore`; commit.

## Task 1: Difficulty presets in the web bot registry (TDD)

**Files:**
- Modify: `mcts_study/catan_mcts/web/bot_registry.py`
- Modify: `mcts_study/catan_mcts/web/server.py` (`/api/bots` + seat-spec resolution)
- Modify: `mcts_study/catan_mcts/web/static/` lobby (difficulty dropdown)
- Test: `mcts_study/tests/test_web_difficulty.py`

**The ladder (each entry justified by a measured winrate):**

| id | label | spec | basis |
|---|---|---|---|
| `beginner` | Beginner | `{type: Random}` | floor |
| `easy` | Easy | `{type: Greedy}` | weak heuristic |
| `medium` | Medium | `{type: PureGnn, ckpt: Cell6 ep10}` | best argmax GNN (~8–18% vs LookV3 field) |
| `hard` | Hard | `{type: LookaheadMctsV3}` | ~70% in 4-way full-Catan tournaments |
| `expert` | Expert | `{type: GnnMcts, ckpt: Cell6, sims: 200, device: cpu}` | 51.7% vs LookV3 head-to-head — strongest known |

No mid-sims GnnMcts tier — the valley makes sims 8–32 *worse* than argmax.

- [ ] **Step 1: failing tests** — `list_difficulties()` returns the 5 presets with `id/label/spec` fields; `resolve_seat_spec({"difficulty": "expert"}, checkpoints_dir)` returns a buildable spec with absolute checkpoint path; unknown id raises `ValueError`; `{"type": ...}` specs pass through unchanged (back-compat).
- [ ] **Step 2: implement** `DIFFICULTIES` table + `resolve_seat_spec()` in `bot_registry.py`; preset checkpoints stored as paths relative to `checkpoints_dir` (resolved at build time). Missing checkpoint → clear `ValueError` listing the expected relative path.
- [ ] **Step 3: server wiring** — `/api/bots` response gains `"difficulties"`; `create_game` resolves each seat through `resolve_seat_spec` before `GameSession`.
- [ ] **Step 4: lobby UI** — difficulty dropdown as the default seat picker; "Advanced" reveals the existing type+checkpoint controls.
- [ ] **Step 5: e2e smoke** — FastAPI TestClient: create a game with 3 difficulty bots (beginner/easy/hard — no torch needed), human seat 0, assert game starts and a bot move advances. Playwright browser check of the lobby if time allows.
- [ ] **Step 6: commit + push** after each green step.

## Task 2: Throughput probe + overnight teacher data-gen

**Files:**
- Create: `mcts_study/scripts/run_distill_teacher.sh` (launch wrapper, nohup + log)
- Output: `runs/v3/distill/teacher_corpus_<ts>/` (WSL Linux fs)

- [ ] **Step 1: probe** — `python -m catan_mcts.experiments.self_play_async --checkpoint runs/v3/rl_checkpoints/round0_Cell6.pt --num-games 8 --self-play --n-sims 200` (full-Catan defaults). Record games/min, mean achieved batch, RAM. **Cite all numbers in the journal.**
- [ ] **Step 2: size the run** — corpus target = measured rate × ~8 h × process count. Try 2 concurrent processes (distinct seed ranges, shared GPU) only if probe shows GPU headroom; CPU is 6c/12t so >2-3 processes thrash (user's "10 workers" applies to the tournament harness, which already takes `--workers`; data-gen concurrency = `--n-concurrent 64` coroutines in-process).
- [ ] **Step 3: launch** with `--self-play` (Dirichlet+temperature — canonical AZ exploration), `--seed-base 21000000`, resumable out-dir, `--ram-budget-mb 40000`. Per-game parquet flush + done.txt are already built in.
- [ ] **Step 4: babysit** — check log every ~30 min (per-batch observability is built in); commit a RUNBOOK note with launch command + PID.

## Task 3: `--policy-sharpen` distillation target in train.py (TDD)

**Files:**
- Modify: `mcts_study/catan_gnn/train.py`
- Test: `mcts_study/tests/test_policy_sharpen.py`

Transform lives in the train loop (like Cand 7's `class_balanced_target`) so cached datasets stay valid: `target' = target^p / Σ target^p` per row, masked to legal actions. `(v/s)^p ∝ v^p` so operating on the normalized target is equivalent to sharpening raw visits.

- [ ] **Step 1: failing tests** — `sharpen_policy_target(t, p=2)`: `[0.5,0.5,0]→[0.5,0.5,0]` (ties preserved), `[0.6,0.3,0.1]→[0.783,0.196,0.022]` (≈, sharper), `p=1` is identity, all-zero row stays zero (no NaN), gradient-free.
- [ ] **Step 2: implement** + thread `policy_sharpen: float = 1.0` through `train()` and CLI (`--policy-sharpen`). Applied to train AND val targets (consistent loss scale).
- [ ] **Step 3: green + commit.**

## Task 4: Train student + arena gate

- [ ] **Step 1: train** — warm-start from `round0_Cell6.pt`, `--policy-sharpen 2.0`, value aux on (multi-task), 10 epochs, per-batch progress logging (hard rule `feedback_training_observability`). One-epoch timing test on a small fixture before the full run.
- [ ] **Step 2: gate** — e10g-style async arena: student-argmax vs raw-Cell6-argmax vs LookV3, 120 games, shared seeds, <5% timeouts. Report 95% CI.
  - **Promote** (student → web `medium`… and consider `hard`) if student-argmax beats raw-argmax outside CI.
  - **Hold** if within CI: document, keep presets as-is — the difficulty ladder still ships on existing checkpoints.
- [ ] **Step 3: journal + commit** regardless of outcome (honest negatives are project currency).

## Task 5: AZ iteration 2 + canonical tournament (conditional)

- [ ] Only if Task 4 promotes AND ≥3 h of night remain: regenerate self-play with the student as evaluator net (teacher = GnnMcts@200 on student), retrain, re-gate.
- [ ] Final: 1200-game canonical tournament for the shipped lineup if time allows; else 120-game gates stand and 1200-game run is documented as follow-up.

## Task 6 (stretch): UI polish

- [ ] Only after Tasks 1-4 are done and committed: bot "thinking" indicator, game-end summary, lobby polish.

---

## Self-review notes
- Spec coverage: difficulty bots (T1), AZ-style learning using existing engine/pipeline/tournament (T2-5), worktree+commits (T0, every task), 10 workers (tournament harness `--workers 10` where the async harness supports it; data-gen concurrency is coroutine-based by design), UI stretch (T6). PR opened after first commit.
- No mid-sims difficulty tiers (valley). No v3-rules data-gen. No val_top1 gating.
- Risk: WSL disk-mount flakiness observed at session start (vhdx on D: failed once, recovered after retry) — RUNBOOK includes resume commands; all long runs resumable by design.
