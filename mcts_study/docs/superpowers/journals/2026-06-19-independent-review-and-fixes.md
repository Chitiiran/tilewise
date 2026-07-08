# Independent Review of the Pipeline Build — Findings & Fixes

**Date:** 2026-06-19
**Why:** The pipeline build (observable/pausable/maxed) had only SELF-review. Two
independent agents were dispatched — an adversarial code reviewer and a fresh-eyes
dashboard-clarity reviewer — because self-review misses exactly these blind spots.
Both found real issues. This documents what they caught and how it was fixed.

---

## Code review — findings & resolution

| ID | Finding | Severity | Status |
|---|---|---|---|
| C1 | `_all_selfplay_dirs` still globbed `*self_play_async*` → on the rust engine, prior-iter dirs invisible → **training window collapsed to ONE iteration** (replay buffer silently defeated) | HIGH | **FIXED** — glob `*self_play_*`; regression test `test_daily_rust_dirs` added |
| C2 | "byte-identical resume" was FALSE: rust resume stacked seed-offset + done-skip (could dup/lose games / stall quota) | HIGH | **FIXED** — rust uses ONE deterministic resume dir + `--resume-dir` (done.txt dedup), no seed-offset; regression test added |
| H1 | `_default_train` wrote a plain CPU `.ts` no consumer reads; consumers re-traced device-suffixed every launch | MED-HIGH | **FIXED** — `_default_train` exports device-matched `.{dev}.ts` + `.{dev}.b{bmax}.batch.ts`, the exact paths consumers reuse |
| M4 | A PAUSED arena fell through to `_aggregate_arena` → could publish a promote/hold verdict on a PARTIAL arena | MED | **FIXED** — pause now raises `ArenaPaused`; `run_day` stops without a verdict; test asserts raise |
| M1 | PAUSE sentinel never cleared → resume immediately re-paused (silent no-op); undocumented | MED | **FIXED** — `run_day` checks PAUSE at the top, stops, and tells the operator to remove it before resume |
| H3 | Sampler could orphan on `kill -9`; docstring claimed "stops on STOP/PAUSE sentinel" but `run_forever` only checked SIGTERM | MED | **FIXED** — `run_forever` now also checks a STOP/PAUSE sentinel (orphan self-terminates); test added |
| M2 | re-exec used a fragile LD_PRELOAD substring check that could mask a broken-path preload (interpreter abort) | LOW-MED | **FIXED** — gate solely on the `_AZ_GPU_REEXEC` one-shot guard; `_rust_cuda_env` only injects the preload if the `.so` exists, else warns |
| L2 | self-play exported a plain B=1 `.ts` it never used (the batched path ignores it) | LOW | **FIXED** — dropped the dead `_ensure_ts` call in self-play |
| H2 | grad-norm logged AFTER opt.step() | (claim) | **NOT A BUG** — reviewer confirmed grads valid there (zero_grad is at loop top). Left as-is; it's sampled-not-every-batch (label is honest enough). |
| M3 | per-epoch-seeded shuffle reproducibility claim too strong under `num_workers>0` + `rotate_mode=random` | MED (latent) | **DOCUMENTED** — `_default_train` uses neither; the contract holds for the production config. Not changed (latent). |

**Verdict the reviewer credited as actually-fine:** the re-exec loop guard doesn't
loop / preserves args; preflight runs once; the Rust batched scheduler is
RNG-faithful so chunking alone doesn't perturb games; grads at the grad-norm site
are valid.

---

## Dashboard clarity review — findings & resolution

| # | Finding | Status |
|---|---|---|
| 1 | Infra cards (Machine/Training) placed ABOVE the decision-relevant Arena card → buries "is the bot improving?" | **FIXED** — moved both BELOW Arena (they answer "is the box busy") |
| 2 | Machine sparkline overlaid util% + power-W/75 on one axis, no legend → amber line unreadable / misleading | **FIXED** — chart now shows GPU util only on a fixed 0–100 axis; power is a stat |
| 3 | Training loss sparkline auto-normalized to its own max → a flat-but-high loss LOOKED like it was descending | **FIXED** — `spark()` now takes an explicit `[ymin,ymax]`; loss is **zero-anchored**; header shows start→now delta so the eye isn't the only judge |
| 5 | `-1.0` failure sentinel rendered as a plausible real value ("util -1%") | **FIXED** — `_ok()`/`_fmt()` treat `<0`/null as "—" |
| 4 | Hardcoded "/75W" and "of 12" would lie on another machine | **PARTIAL** — removed the misleading "/75" from the chart; labels now "GPU power draw" / "CPU load (1-min avg)" (no false denominators). Backend power-limit/nproc emission deferred. |
| spark | no axes/gridlines | **IMPROVED** — `spark()` now draws ymin/ymax gridlines + labels |
| 6 | "nothing running" vs "broken" not distinguished | NOTED — not changed this pass (cards hide when idle, as before); a future "idle" stub is a nice-to-have |
| grad-norm unexplained | caption now explains "≈ steady is healthy, spikes = instability" |

**Reviewer credited as already-clear:** the glossary, liveness expectation-setting,
the CI bar (the model the new charts now imitate), rotation-fairness honesty,
table tooltips, Elo small-range guard. New cards already matched the visual style.

---

## Net result
All HIGH/MED code findings fixed + regression-tested (the C1 glob bug had NO test
before — now it does). The two actively-misleading dashboard charts (auto-normalized
loss, dual-unit GPU overlay) are corrected to honest fixed-axis charts, reordered
below the decision cards. Full plain-pytest pipeline sweep green after the fixes.

Lesson: self-review verified that the code RAN; it did not catch that the window
silently collapsed to one iteration (C1) or that the loss chart could lie (dash #3).
Independent adversarial + fresh-eyes review caught both. Worth doing by default.
