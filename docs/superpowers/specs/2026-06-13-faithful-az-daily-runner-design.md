# Faithful AZ Daily Runner + Resilience + Dashboard — design

**Date:** 2026-06-13
**Status:** approved (user, 2026-06-13)
**Branch:** `az-difficulty-bots` (PR #3)
**Builds on:** `2026-06-11-az-loop-design.md` (the `catan_az` loop this wraps)

## 1. Goal & framing

Turn the working `catan_az` loop into a **faithful-AlphaZero daily trainer**
that can be run every day (manually now, cron later) and survive unpredictable
interruptions losing **at most one game** of work. The organizing principle is
**resilience + observability**, not a fixed schedule: time budget is a soft hint;
the runner always does the next useful unit of work and stops cleanly whenever
told to.

User decisions baked in:
- **Option 1 (faithful AZ):** train each candidate on a window **dominated by
  current-champion self-play** (`fresh_ratio = 0.70`).
- **Manual + cron-ready** trigger.
- **Pauseable, ≤1-game loss, observable.**
- **Fresh-ratio targeting, resumable.**
- **Resource guards stay in** (environment failures are physical reality, not
  bugs — see §3).
- **Archive after a full cycle**, keep window games on fast disk, move
  out-of-window raw parquet to HDD.
- **Hard-abort on busy GPU.** Dashboard shows the **live web-playable champion
  link.**

## 2. Three-category failure model (the design's backbone)

| Category | Recurs? | Response |
|---|---|---|
| **Mechanical bugs** (run-dir collision, non-idempotent publish, straggler stall, stale install, orphan procs) | No — fixed once, tests lock them | Make impossible by construction |
| **Environment / resource** (WSL/disk drop, RAM/VRAM OOM, GPU contention) | **Yes — physical reality** | Preflight + in-loop guards: detect, halt safely or degrade, never corrupt |
| **Algorithmic outcomes** (HOLD, high draw-rate, stagnation) | They're *results*, not failures | Record + surface to metrics/dashboard; never block the loop |

All mechanical failures hit this session are already fixed (dedup, make_run_dir
PID dirs, active_game_count, wall-clock cap, PUBLISH idempotency, VP-tiebreak).
This spec ensures category 2 is survived and category 3 is surfaced.

## 3. Architecture

```
run_az_day.sh                  cron-ready entry: preflight -> daily driver -> archive
  catan_az/preflight.py        env + resource guards (abort hard / degrade soft)
  catan_az/daily.py            resumable daily driver (fresh-ratio self-play, soft --until,
                               stagnation detect); calls loop.run_iteration() per cycle
  catan_az/archive.py          post-cycle: move out-of-window raw parquet to HDD
  catan_az/dashboard/          static HTML + small FastAPI route over journal/status/ladder
```

The existing `catan_az/loop.py` stays the iteration engine; `daily.py`
orchestrates many iterations. Minimal hooks added to existing code: a `rules_id`
+ `champion_name` tag per run dir, and a fresh-ratio path in `buffer.py`.

## 4. Resumability — ≤1 game loss

Layers, each already or newly checkpointed:
- **Per game:** parquet shard + `done.txt` line flushed on completion (exists).
- **Per stage:** `SELFPLAY/TRAIN/ARENA/PUBLISH.done` markers (exist).
- **Per cycle:** new `daily_state.json` manifest — `{iter, stage, champion,
  fresh_target, fresh_done, rules_id, ts}`, atomically rewritten at each
  transition.

On restart `daily.py` reads `daily_state.json` and resumes the exact stage:
- Self-play counts existing fresh done-games and generates only the **deficit**
  to the fresh target (never regenerates).
- Train/arena resume via their done-markers + arena `results.jsonl`.
- A `STOP` sentinel file → graceful stop after the in-flight game flushes
  (worst case: lose that one game).

Power-loss mid-write is survived by atomic writes (tmp+rename) for all JSON
state; a half-written parquet shard is skipped+logged on next read, not fatal.

## 5. Fresh-ratio self-play (the Option-1 fix for window dilution)

iter-2 held because 63% of its window was older cell6-era games. Fix:
- Each run dir tagged `champion_name` + `rules_id` (engine-rules version).
- `daily.py` generates current-champion games until they are
  `≥ fresh_ratio (0.70) × window_games` of the window, **then** trains.
- `buffer.select_window` only mixes **same-`rules_id`** games, so a future
  engine-fidelity bump (robber, trades) cleanly flushes stale-rules games
  instead of poisoning the window.

Per-iteration self-play volume is therefore derived, not fixed:
`fresh_target = ceil(fresh_ratio * window_games)`; resumable toward it.

## 6. Guards (preflight + in-loop)

Run before each cycle. **Hard** = abort with a clear logged reason + notify;
**Soft** = degrade and continue.

| Guard | Type | Action |
|---|---|---|
| WSL up + `runs/` writable | hard | abort (env down) |
| Fast-disk free ≥ `min_fast_gb` (default 10) | hard | abort |
| HDD free ≥ `min_hdd_gb` (default 20) | hard | abort (no room to archive) |
| Editable install path == this worktree | self-heal | `maturin develop --release` if stale, else proceed |
| Stale `catan_az`/`self_play_async` procs from a dead run | self-heal | reap them |
| PID-file lock already held | hard | refuse double-run |
| GPU busy (mem-used or util > threshold, or foreign proc) | **hard** | abort — don't fight another GPU job |
| RAM free < `ram_budget` | soft | cap concurrent self-play procs from the budget |

## 7. Observability + dashboard

**Enriched `status.json`** (per-stage): iter, stage, fresh_done/target, ETA,
last-cycle verdict + winrate + draw-rate, free disk/RAM/GPU, stagnation flag.

**Stagnation + anchor:** `stagnation_holds` consecutive HOLDs (default 5) →
flag in status + one notification (not an error — a surfaced result). Periodic
LookV3 **anchor match** (anchor_every=5, already in cfg) writes an absolute-Elo
reference row to the journal — guards against Elo inflation + silent regression.

**Minimal dashboard** (`catan_az/dashboard/`): one static `index.html` +
auto-refresh (every 5s) over a ~30-line FastAPI route serving the three JSON
files (journal.csv, status.json, ladder.json). At a glance:
- Champion name + Elo + **"Play the champion" link** (deep-link to the existing
  web app's lobby preselecting the AZ-champion difficulty tier).
- Elo-vs-iteration sparkline.
- Last-10-iterations table: verdict / cand-winrate / draw-rate / timeouts.
- Current run: stage, fresh progress bar, ETA, live disk/RAM/GPU.
- Stagnation / last-anchor flags.

Reuses the existing `catan_mcts/web` FastAPI patterns; launched with one command.

## 8. Data lifecycle (archive after full cycle)

After a cycle **publishes**, `archive.py`:
1. Recomputes the live window (games the *next* iteration needs).
2. Moves every run dir's raw parquet **not in the window** to
   `D:/catan_az_archive/<rules_id>/<iter>/` (HDD), leaving a small
   `ARCHIVED.txt` breadcrumb on fast disk.
3. **Keeps on fast disk:** all checkpoints, ladder, journal, status, and
   in-window self-play.

Never deletes — moves (per the no-delete rule). Archive is itself resumable
(idempotent move with breadcrumb).

**D:-drop caveat:** `/mnt/d/` is the same physical HDD that dropped mid-session.
Archive runs only *after* a cycle fully publishes (so an archive failure never
costs training progress), and the preflight HDD-free check confirms `/mnt/d/`
is mounted+writable before a cycle starts; a mid-archive D: drop leaves the
breadcrumb absent so the move re-runs cleanly next time.

## 9. The draw-rate trap (designed-in, flagged for tuning)

As champion and candidate converge, draw rate climbs toward the
`arena_max_draw_rate` (0.40) invalid threshold — a likely long-run plateau wall
for continual training. Mitigations built in now:
- **VP-margin tiebreak fallback:** extends today's `_vp_leader` helper — when
  top *public* VP ties, break by a *second* signal (total settlements+cities,
  then total resource count) before declaring a draw. A true tie on all signals
  is still a draw. (Strictly additive to the shipped VP-tiebreak.)
- The anchor match gives absolute progress signal even during relative
  plateaus.
- Threshold stays a config knob; if plateaus persist, revisit (promote-by-margin
  over more games) — out of scope for v1, noted as the first thing to tune.

## 10. Testing (TDD)

- **preflight:** mock low-disk / busy-GPU / stale-install / held-lock → correct
  hard-abort vs self-heal vs soft-degrade.
- **fresh-ratio:** window selection hits ≥0.70 fresh; same-`rules_id` only.
- **daily resume:** kill mid-self-play / mid-arena → resumes from manifest,
  regenerates only the deficit, ≤1 game lost.
- **archive:** only out-of-window games move; window intact; idempotent re-run.
- **VP-margin tiebreak:** tie on VP broken by margin signal; true tie → draw.
- **dashboard:** route returns well-formed JSON; champion link well-formed.
- **integration:** one micro daily cycle (tiny budget, scratch net, CPU) end to
  end, then a simulated kill+resume.

## 11. Out of scope (v1)

- Cron installation itself (built cron-ready; user wires the scheduler).
- Multi-GPU / Rust self-play (the throughput lever; separate work).
- Promote-by-margin (noted in §9 as first tuning if plateaus persist).
- Engine-fidelity changes (robber/trades) — separate specs; this design only
  makes the window `rules_id`-safe for them.

## 12. Config additions (AzConfig)

```
fresh_ratio: float = 0.70
rules_id: str = "v3-full"          # vp10+bonuses+current action space
min_fast_gb: float = 10.0
min_hdd_gb: float = 20.0
gpu_busy_mem_mb: float = 500.0     # foreign GPU usage above this => abort
gpu_busy_util_pct: float = 20.0
stagnation_holds: int = 5
archive_root: str = "/mnt/d/catan_az_archive"   # HDD
dashboard_port: int = 8099
```
