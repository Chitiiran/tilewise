# Phase 1: Batched Arena + SHA Stamping + Observability Minimums — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** give the arena the same cross-game leaf batching self-play got (fixing the ~25h/300-game
B=1 bottleneck), fix git-SHA stamping under WSL+worktree, and add the two observability minimums —
so Phase 2/3 of `docs/superpowers/plans/2026-07-08-distillation-first-roadmap.md` (main worktree)
have a fast, trustworthy measurement pipeline.

**Architecture:** reuse `SearchSession` verbatim (it is net-agnostic: emits `NeedEval(obs)`,
consumes `provide(out)`). New code = an arena `Slot` (MtRng game-chance + per-seat NpRngs +
greedy move pick) and a two-queue scheduler `play_arena_games_batched` that routes each game's
parked leaf to its mover's net-queue (cand/champ), flushing each queue at `b_max`. Python side:
`run_arena_games` gains optional batched params; `arena.py:_ts` exports the batched
device-suffixed `.ts`. Contract per Task-10 precedent: bit-exact REPRODUCIBILITY across runs +
evaluator faithfulness ≤1e-4 + winner-level agreement audit vs the B=1 oracle (which stays).

**Tech stack:** Rust (`catan_mcts_rs`), tch/TorchScript, PyO3, pytest + cargo test. All builds/tests
in WSL via `mcts_study/scripts/maturin_build_mctsrs.sh` (canonical env: LIBTORCH_USE_PYTORCH=1,
LIBTORCH_BYPASS_VERSION_CHECK=1, CARGO_TARGET_DIR=/home/chitii/cmcts_target).

## Global constraints

- After ANY Rust change: `bash mcts_study/scripts/maturin_build_mctsrs.sh` (and root
  `maturin develop --release` if `catan_engine` changed). pytest without rebuild = false negative.
- Cargo test env: `export LIBTORCH=<venv>/torch LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 LD_LIBRARY_PATH=<venv>/torch/lib CARGO_TARGET_DIR=/home/chitii/cmcts_target`.
- The B=1 arena path (`play_arena_game`, arena.rs:64-135) is the ORACLE — do not modify it.
- Python contract is FROZEN: `run_arena_games(cand_ts, champ_ts, pairs, sims, vp_target, bonuses)`
  returns per-game dicts with exactly `seed, rot, winner_seat, winner_role, timed_out, vp_margin`
  (python.rs:351-358); Python injects `ts` and owns chunking/PAUSE (arena.py:288-308).
- Fixtures: `mcts_study/spike/wrapper_traced.ts` (B=1, CPU-traced) and `wrapper_batched.ts`
  (B_MAX=8, CPU) — regenerate ONLY via `scripts/export_spike_batched.py` / an equivalent
  CPU-device export from `az_iter_1.pt`. Never commit cuda-traced fixtures (2026-07-08 lesson).
- Commit per task on `az-difficulty-bots`; push freely; no merge to main without approval.

## Code map (from the 2026-07-08 seam analysis — file:line cited)

- `SearchSession` state machine: `catan_mcts_rs/src/mcts.rs:242-470`. `new(root_engine, n_sims, c, dirichlet_alpha, dirichlet_eps)` (:285), `pump(&mut self, rng) -> SessionStep` (:351, call once), `provide(&mut self, out, rng) -> SessionStep` (:411), `take_visit_counts()` (:467), `last_root_value` (:280). `SessionStep::{NeedEval(Observation), Done}` (:256).
- Self-play donor scheduler: `play_games_batched(ev, seeds, cfg)` `selfplay.rs:298-335`; per-game `Slot` `selfplay.rs:170-256` (`advance_to_search` :205 = desync handler; `finish_move` :246).
- Arena B=1 oracle: `play_arena_game(ev_cand, ev_champ, seed, seating_cand, sims, vp_target, bonuses, max_steps)` `arena.rs:64-135`. Game chance = `MtRng::from_seed(seed)` + cumulative-probability walk (:75, :86-97). MCTS RNGs: cand `NpRng::from_seed(seed+11)`, champ `seed+13` (:77-78). Greedy `best_action` (mcts.rs:473). Seating `seating_is_cand(rot)` (:24); `seed_plan` (:34).
- Evaluator: `load(path, device)` (:41), `load_batched(path, device, b_max)` (:50), `evaluate_one` (:62), `evaluate_batch(&[&Observation])` (:122, pads to b_max, ONE forward). One CModule per instance — two instances = two nets.
- Python entries: `run_arena_games` python.rs:327-358 (loads B=1 via `load`, serial loop — the thing to extend); `run_selfplay`'s batched branch python.rs:255-265 (the pattern to mirror).
- Python caller: `arena.py:_run_arena_rust` :254-308; `_ts()` :272-278 (exports `.{dev}.ts` B=1);
  batched-export pattern to mirror: `self_play_rust._ensure_batched_ts` (self_play_rust.py:63-73,
  writes `.{dev}.b{b}.batch.ts`).
- Test templates: `tests/batched_selfplay.rs` (bit-exact reproducibility via `records_equal`, fixture load_batched B=8 CPU), `tests/session_parity.rs` (pump/provide contract), `tests/batched_evaluator.rs` (faithfulness ≤1e-4).

---

### Task 1: Arena `Slot` — per-game state with the arena's RNG scheme

**Files:**
- Modify: `catan_mcts_rs/src/arena.rs` (add `ArenaSlot` below `play_arena_game`)
- Test: `catan_mcts_rs/tests/arena_slot.rs` (new)

**Interfaces:**
- Produces (used by Task 2's scheduler):
  ```rust
  pub struct ArenaSlot {
      pub engine: Engine,
      pub chance_rng: MtRng,          // game-level chance (arena contract)
      pub rng_cand: NpRng,            // seed+11 — MCTS chance for cand searches
      pub rng_champ: NpRng,           // seed+13 — champ searches
      pub seating_cand: [bool; 4],
      pub seed: u64,
      pub rot: usize,
      pub session: Option<SearchSession>,
      pub cur_is_cand: bool,          // net owning the CURRENT session
      pub steps: u32,
      pub done: bool,
      pub result: Option<ArenaGameResult>,
  }
  impl ArenaSlot {
      pub fn new(rot: usize, seed: u64, vp_target: u8, bonuses: bool) -> Self;
      /// Advance through chance (MtRng cumulative walk, byte-copy of
      /// arena.rs:86-97) and single-legal fast-paths; at a decision node start
      /// a SearchSession (greedy: dirichlet_eps=0.0, alpha=0.8, c=1.4) and
      /// pump it with the MOVER's rng; set cur_is_cand. Returns the parked
      /// leaf obs, or None when the game reached terminal/max_steps (sets
      /// done + result exactly like arena.rs:119-134).
      pub fn advance_to_search(&mut self, sims: u32, max_steps: u32) -> Option<Observation>;
      /// Session Done: take_visit_counts -> best_action -> engine.apply;
      /// clears session, bumps steps.
      pub fn finish_move(&mut self);
      /// The rng matching cur_is_cand (for provide()).
      pub fn cur_rng(&mut self) -> &mut NpRng;
  }
  ```
- Consumes: `SearchSession` (mcts.rs, UNCHANGED), `MtRng` (mt.rs), `NpRng`, `ArenaGameResult` (arena.rs:53).

- [ ] **Step 1: Write the failing test** — `tests/arena_slot.rs`:

```rust
//! ArenaSlot must reproduce play_arena_game's game trajectory exactly when
//! driven with the same net outputs: same chance walk (MtRng), same per-seat
//! rngs, same greedy picks. Driving the slot with evaluate_one makes it a
//! B=1 re-encoding of the oracle -> full-game equality is REQUIRED here
//! (same kernels, same RNG streams, only the control flow differs).
use catan_mcts_rs::arena::{play_arena_game, seating_is_cand, ArenaSlot};
use catan_mcts_rs::evaluator::TorchScriptEvaluator;
use catan_mcts_rs::mcts::SessionStep;
use std::path::PathBuf;
use tch::Device;

fn spike() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("mcts_study").join("spike")
}

#[test]
fn slot_driven_b1_equals_oracle() {
    let ts = spike().join("wrapper_traced.ts");
    if !ts.exists() { eprintln!("skip: fixture missing"); return; }
    let ev = TorchScriptEvaluator::load(ts.to_str().unwrap(), Device::Cpu).unwrap();
    for (rot, seed) in [(0usize, 42u64), (1, 43), (2, 44), (3, 45)] {
        let seating = seating_is_cand(rot);
        let oracle = play_arena_game(&ev, &ev, seed, seating, 8, 10, true, 5000);
        let mut slot = ArenaSlot::new(rot, seed, 10, true);
        loop {
            match slot.advance_to_search(8, 5000) {
                None => break,
                Some(mut obs) => {
                    loop {
                        let out = ev.evaluate_one(&obs);
                        match slot.session.as_mut().unwrap()
                                  .provide(&out, slot_cur_rng(&mut slot)) {
                            SessionStep::NeedEval(o) => obs = o,
                            SessionStep::Done => break,
                        }
                    }
                    slot.finish_move();
                }
            }
        }
        let got = slot.result.expect("slot finished without result");
        assert_eq!(got.winner_seat, oracle.winner_seat, "rot={rot} seed={seed}");
        assert_eq!(got.timed_out, oracle.timed_out);
        assert_eq!(got.vp_margin, oracle.vp_margin);
    }
}

// helper because provide borrows session and rng from the same struct:
fn slot_cur_rng(slot: &mut ArenaSlot) -> &mut catan_mcts_rs::rng::NpRng {
    // implement in the test with split borrows via the ArenaSlot API decided
    // in implementation (e.g. a method `provide_cur(&mut self, out) -> SessionStep`
    // that internally routes to the right rng is ALSO acceptable — if chosen,
    // update this test to call slot.provide_cur(&out) instead).
    unimplemented!("replace with the API implemented in Task 1")
}
```

Note: the borrow of `session` and `cur_rng` together will not compile as sketched — the
implementer should expose `provide_cur(&mut self, out: &NetOutput) -> SessionStep` on
`ArenaSlot` (routing to the internal rng) and use that in the test. Keep the assertion
structure identical.

- [ ] **Step 2: Run test to verify it fails** — `cargo test -p catan_mcts_rs --test arena_slot` → FAIL (unresolved import `ArenaSlot`).
- [ ] **Step 3: Implement `ArenaSlot`** in arena.rs. `advance_to_search`: copy the chance walk verbatim from arena.rs:86-97 (MtRng `random_f64()` + cumulative probability), single-legal fast-path as in selfplay.rs:205-230 but WITHOUT recording; start `SearchSession::new(engine.clone(), sims, 1.4, 0.8, 0.0)`; `pump` with the mover's rng; terminal/step-cap → build `ArenaGameResult` exactly as arena.rs:119-134 (vp-leader on timeout, +1.0 scan otherwise). `finish_move`: `take_visit_counts` → `best_action` → apply.
- [ ] **Step 4: Rebuild + run test** — build script, then `cargo test -p catan_mcts_rs --test arena_slot` → PASS. This test IS the parity gate for Task 1 (full-game equality vs oracle at B=1).
- [ ] **Step 5: Commit** — `git add catan_mcts_rs/src/arena.rs catan_mcts_rs/tests/arena_slot.rs && git commit -m "feat(arena-batch): ArenaSlot — pausable per-game arena state, B=1-oracle-equal"`

### Task 2: Two-queue scheduler `play_arena_games_batched`

**Files:**
- Modify: `catan_mcts_rs/src/arena.rs`
- Test: `catan_mcts_rs/tests/batched_arena.rs` (new)

**Interfaces:**
- Produces:
  ```rust
  pub fn play_arena_games_batched(
      ev_cand: &TorchScriptEvaluator,   // load_batched
      ev_champ: &TorchScriptEvaluator,  // load_batched
      pairs: &[(usize, u64)],           // (rot, seed)
      sims: u32, vp_target: u8, bonuses: bool, max_steps: u32,
  ) -> Vec<ArenaGameResult>             // in pairs order
  ```
- Consumes: `ArenaSlot` (Task 1), `evaluate_batch` (evaluator.rs:122).

- [ ] **Step 1: Write the failing tests** — `tests/batched_arena.rs` with two tests:
  (a) `batched_arena_reproducible`: run `play_arena_games_batched` twice over
  `seed_plan(7000, 8)` with both evaluators = `load_batched(wrapper_batched.ts, Cpu, 8)`;
  assert the two result vectors are field-identical (winner_seat, timed_out, vp_margin, per
  index) — mirror `records_equal` in batched_selfplay.rs:17-33.
  (b) `batched_arena_agreement_vs_oracle`: run the batched scheduler AND the B=1
  `play_arena_game` oracle over the same 8 (rot,seed) pairs (oracle uses
  `load(wrapper_traced.ts)`); count winner_seat matches; assert `matches >= 7` and print any
  mismatch. (Batched kernels reassociate floats ~1e-7 → a rare argmax flip is tolerated at
  padding boundaries; a systematic divergence is not. If matches < 8, file the diff in the
  Task-5 journal with the seed for manual trace before proceeding.)
- [ ] **Step 2: Run to verify both fail** (unresolved `play_arena_games_batched`).
- [ ] **Step 3: Implement.** Mirror selfplay.rs:298-335: init slots from pairs; `parked: Vec<Option<Observation>>` seeded by `advance_to_search`; loop — partition active indices by `slots[i].cur_is_cand` into `q_cand`/`q_champ`; for each queue `chunks(ev.b_max())` → `evaluate_batch` → `provide_cur` per result; `NeedEval` re-parks, `Done` → `finish_move` + `advance_to_search` re-park-or-retire. Break when both queues empty. Return `slots.map(|s| s.result.unwrap())`.
- [ ] **Step 4: Rebuild + run** → both PASS.
- [ ] **Step 5: Commit** — `feat(arena-batch): two-queue cross-game batched arena scheduler`

### Task 3: PyO3 entry — batched params on `run_arena_games`

**Files:**
- Modify: `catan_mcts_rs/src/python.rs:327-358`
- Test: `mcts_study/tests/test_rust_arena_batched.py` (new)

**Interfaces:**
- Produces: `run_arena_games(cand_ts, champ_ts, pairs, sims, vp_target, bonuses, batched_cand_ts=None, batched_champ_ts=None, b_max=None)` — when the batched kwargs are given, `load_batched` both on `infer_device()` and call `play_arena_games_batched`; else the existing serial B=1 loop (oracle path, unchanged). Same per-game dict schema (python.rs:351-358), same order.

- [ ] **Step 1: Failing pytest** — `test_rust_arena_batched.py`:

```python
import pytest
from pathlib import Path

SPIKE = Path(__file__).resolve().parents[1] / "spike"

@pytest.mark.skipif(not (SPIKE / "wrapper_batched.ts").exists(), reason="fixture missing")
def test_batched_kwargs_same_schema_and_dedup_key():
    import catan_mcts_rs
    pairs = [(0, 9001), (1, 9002), (2, 9003), (3, 9004)]
    b1 = str(SPIKE / "wrapper_traced.ts")
    bb = str(SPIKE / "wrapper_batched.ts")
    recs = catan_mcts_rs.run_arena_games(
        b1, b1, pairs, 8, 10, True,
        batched_cand_ts=bb, batched_champ_ts=bb, b_max=8)
    assert [r["seed"] for r in recs] == [s for _, s in pairs]
    for r in recs:
        assert set(r) == {"seed", "rot", "winner_seat", "winner_role",
                          "timed_out", "vp_margin"}
        assert r["winner_role"] in ("cand", "champ", None)
```

- [ ] **Step 2: Run** — `python -m pytest mcts_study/tests/test_rust_arena_batched.py -q` → FAIL (unexpected keyword).
- [ ] **Step 3: Implement** in python.rs (mirror `run_selfplay`'s batched branch at :255-265; keep positional signature prefix identical so arena.py's existing call keeps working).
- [ ] **Step 4: Rebuild (`maturin_build_mctsrs.sh`) + pytest** → PASS.
- [ ] **Step 5: Commit** — `feat(arena-batch): run_arena_games batched kwargs (B=1 path preserved as oracle)`

### Task 4: arena.py — batched `.ts` export + wiring

**Files:**
- Modify: `mcts_study/catan_az/arena.py:262-301` (`_run_arena_rust`)
- Test: `mcts_study/tests/test_az_arena.py` (add one test)

**Interfaces:**
- Consumes: Task 3 kwargs; `export_batched` (catan_gnn/export_torchscript.py) via the
  `_ensure_batched_ts` pattern (self_play_rust.py:63-73 — device-suffixed `.{dev}.b{b}.batch.ts`).
- Produces: `_run_arena_rust` exports BOTH nets' batched `.ts` next to their checkpoints and
  passes `batched_cand_ts/batched_champ_ts/b_max=cfg.max_batch`. results.jsonl/PAUSE/chunk
  semantics unchanged (arena.py:288-308).

- [ ] **Step 1: Failing pytest** — monkeypatch `catan_mcts_rs.run_arena_games` with a recorder
  fake returning valid dicts; monkeypatch the export to a no-op writing a marker file; call
  `_run_arena_rust` with tmp checkpoints; assert the fake received `batched_cand_ts` ending
  `.b{cfg.max_batch}.batch.ts` and `b_max == cfg.max_batch`, and results.jsonl got one line
  per game with `ts` injected.
- [ ] **Step 2: Run** → FAIL (kwargs never passed).
- [ ] **Step 3: Implement** (small: an `_ts_batched(ckpt)` helper beside `_ts`, plus the call-site kwargs).
- [ ] **Step 4: pytest the arena test file** → PASS (including the existing pinned tests).
- [ ] **Step 5: Commit** — `feat(arena-batch): arena.py exports+passes batched nets (chunk/PAUSE semantics unchanged)`

### Task 5: Throughput gate + journal (Phase-1 exit)

**Files:**
- Create: `mcts_study/scripts/arena_throughput_gate.sh` (env recipe + 40-game run)
- Create: `docs/superpowers/journals/2026-07-XX-phase1-batched-arena.md`

- [ ] **Step 1:** Script: champion `az_iter_1` vs itself, 40 games, sims=200, GPU env
  (`daily.py`'s `_rust_cuda_env` recipe), wall-clock timed.
- [ ] **Step 2:** Run. Gate: **≤ 1 h wall-clock** (roadmap Phase-1 exit; expectation ~15-30 min
  at 1.5-3 games/min — extrapolated from self-play 3.26 g/min shared across two nets).
- [ ] **Step 3:** Journal: measured g/min, GPU util, batch fill, agreement-test results, any
  oracle mismatches traced. Commit.

### Task 6: BUG B — git SHA stamping under WSL+worktree

**Files:**
- Modify: the SHA-capture site in `mcts_study/catan_az/` (locate via `grep -rn "rev-parse\|git" mcts_study/catan_az/*.py` — the shake-out journal §6 names the failure: worktree `.git` file holds a Windows `gitdir:` path, meaningless in WSL)
- Test: `mcts_study/tests/test_az_hardening.py` (add)

- [ ] **Step 1: Failing test** — call the SHA helper with a monkeypatched broken `git` (env
  `GIT_DIR` pointing at a Windows-style path) and assert it falls back to (a) `AZ_GIT_SHA`
  env var if set, else (b) parsing `<repo>/.git/worktrees/<name>/HEAD` textually, and never
  returns empty silently (returns `"unknown"` + logs at worst).
- [ ] **Step 2-4:** Implement (order: env var → `git -C <windows-main-repo-as-wsl-path>` → textual HEAD parse → "unknown"), rebuild not needed (pure Python), pytest → PASS.
- [ ] **Step 5:** Wire `AZ_GIT_SHA` into the launch path: `daily.py` sets it for workers from its own capture. Commit — `fix(az): SHA stamping survives WSL+worktree (BUG B, shake-out journal §6)`

### Task 7: Observability minimums (roadmap Phase-1c)

**Files:**
- Create: `mcts_study/catan_az/data_quality.py` — `summarize_selfplay_dir(dir) -> dict`
  (winners/seat, draw+timeout counts, length p50/p90/max, games) + `degeneracy_verdict(summary, cfg) -> "ok"|"degenerate"` (>20% timeouts or >40% draws or 0 winners → degenerate)
- Modify: the self-play completion path in `catan_az/loop.py` (write `data_quality.json` next to SELFPLAY.done; refuse to mark healthy when degenerate)
- Modify: `catan_gnn/train.py` — log per-epoch `val_value_mse` and `val_value_sign_acc`
  next to `val_top1` (the value head currently has NO metric — shake-out journal §3)
- Test: `mcts_study/tests/test_data_quality.py` (new), extend an existing train-metrics test

- [ ] **Step 1: Failing tests** — (a) fixture parquet dir (reuse the tiny fixtures other az
  tests use) → summary fields + a crafted all-timeout dir → "degenerate"; (b) train one epoch
  on the existing toy fixture → training_log.json rows contain `val_value_mse` and
  `val_value_sign_acc` floats.
- [ ] **Step 2-4:** Implement; run `pytest mcts_study/tests/ -q` FULL suite → all green.
- [ ] **Step 5:** Commit — `feat(az): data-quality gate + value-head val metrics (observability minimums)`

---

## Self-review notes

- Task 1's test compiles only after choosing the `provide_cur` API — flagged inline; the
  implementer must reconcile the test with the chosen borrow-safe API (assertions unchanged).
- Task 2(b)'s ≥7/8 agreement threshold encodes the accepted float-reassoc tolerance; a
  failure below it is a STOP-and-investigate, not a threshold to loosen.
- Tasks 6-7 are independent of 1-5 and may be done first if GPU is busy.
- Phase-1 exit = Task 5 gate green + full pytest + full cargo suite green. Then: PR
  `az-difficulty-bots` → main for user review (approved 2026-07-08), and Phase 2
  (re-anchor baselines) per the roadmap.
