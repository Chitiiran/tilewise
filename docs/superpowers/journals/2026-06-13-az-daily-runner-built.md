# Faithful AZ Daily Runner — built & how to use it

**Date:** 2026-06-13
**Spec:** `docs/superpowers/specs/2026-06-13-faithful-az-daily-runner-design.md`
**Plan:** `docs/superpowers/plans/2026-06-13-faithful-az-daily-runner.md`
**Status:** built, 65 tests green (63 fast + micro daily integration).

## What it is

A resumable, resource-guarded **daily** AlphaZero trainer wrapping the
`catan_az` loop. Faithful AZ (fresh-ratio 0.70 window — fixes the iter-2
dilution), survives unpredictable interruption losing **≤1 game**, runs
self-play at low OS priority so your foreground work always wins the CPU, and
archives old games to the HDD after each cycle.

## Run it (daily)

From WSL with the mcts-study venv:

```bash
cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study
./scripts/run_az_day.sh                       # default loop root, run all day
MAX_ITERS=3 ./scripts/run_az_day.sh           # cap iterations for the day
```

Cron-ready: drop the same line in a scheduler. A PID-lock refuses a second
concurrent run; stale self-play procs from a dead run are reaped on start.

## Pause / resume (≤1-game loss)

```bash
touch /home/chitii/catan_data/runs/v3/az_loop/STOP   # graceful stop between games
rm   /home/chitii/catan_data/runs/v3/az_loop/STOP    # allow next run to proceed
```

A kill (Ctrl-C, sleep, WSL drop, power loss) loses at most the one in-flight
game: every completed game is flushed (parquet + done.txt), every stage has a
done-marker, and `daily_state.json` records the exact stage. The next
`run_az_day.sh` resumes from there — self-play regenerates only the deficit
toward the fresh-ratio target, never re-runs finished games.

## Watch it (dashboard)

```bash
python -c "import uvicorn; from catan_az.dashboard.server import create_dashboard; \
  uvicorn.run(create_dashboard(loop_root='/home/chitii/catan_data/runs/v3/az_loop', web_port=8000), host='127.0.0.1', port=8099)"
```

Open `http://localhost:8099` — auto-refreshes every 5s. Shows: champion +
Elo + **Play-the-champion link** (into the web app's az-champion tier), current
run stage, last-10-iterations table (verdict/winrate/draws), and a STAGNATION
flag if the champion hasn't improved for `stagnation_holds` (5) iterations.

## Resource model (hardware-grounded)

| Resource | Size | Binding? |
|---|---|---|
| CPU | 6c/12t Ryzen 5600H | no (7 procs leaves 5 threads) |
| RAM | 54 GB (WSL) | no |
| **VRAM** | **4 GB GTX 1650** | **YES — ~7 GPU procs (~535 MB each)** |

Preflight caps self-play procs to what free VRAM holds (degrade-soft, never
OOM); aborts hard on low disk (fast or HDD) or 0 VRAM. **No GPU-busy abort** —
you share the GPU fine (training is CPU-bound, ~25-56% GPU util). Workers run
at `nice 10`.

## Data lifecycle

After a cycle publishes, raw self-play parquet that's fallen out of the sliding
window is **moved** (never deleted) to `/mnt/d/catan_az_archive/<rules_id>/`.
Checkpoints, ladder, journal stay on fast disk. Idempotent (ARCHIVED.txt
breadcrumb).

## Failure model (the design's backbone)

| Category | Recurs? | Handling |
|---|---|---|
| Mechanical bugs | No — fixed once | impossible by construction (tests lock) |
| Environment/resource | Yes — physical | preflight + guards, survive ≤1-game loss |
| Algorithmic outcomes (HOLD, draws, stagnation) | results, not failures | surfaced to dashboard, never block |

## Config knobs (AzConfig, spec §12)

`fresh_ratio=0.70`, `rules_id="v3-full"`, `worker_procs_max=7`,
`per_proc_vram_mb=535`, `worker_nice=10`, `min_fast_gb=10`, `min_hdd_gb=20`,
`stagnation_holds=5`, `archive_root=/mnt/d/catan_az_archive`,
`dashboard_port=8099`. Override via a JSON config passed to `--config`.

## Deferred (noted in spec §11)

- Mixed GPU+CPU-inference worker pool (to raise the 7-proc cap toward 10).
- `--pause-if-busy` auto-throttle (low-priority workers cover v1).
- Promote-by-margin (first tuning if draw-rate plateaus persist).
- Engine-fidelity changes (robber/trades) — `rules_id` makes the window safe
  for them; each gets its own spec.

## Next

The runner is ready to drive iteration 3+ with a fresh-dominated window (the
iter-2 dilution fix). First real daily run will confirm fresh-ratio self-play
produces an improving candidate.
