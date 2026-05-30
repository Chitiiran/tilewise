# 2026-05-09 — Diagonal training crash, evidence-based postmortem + cleanup

## Context

Previous session (2026-05-09 00:56) launched a "diagonal" training run via
`scripts/launch_diagonal.sh` (PID 624 in WSL): h128_l4 fresh + h64_l3 resume
from epoch 10. The launcher used `setsid python ... &` and walked away.
Today (2026-05-09 morning) I re-opened the project to find PID 624 dead and
the orchestrator JSON stale. This journal documents the investigation,
the cleanup, and the hardened launch procedure.

## Timeline of events (UTC-relative epochs decoded to EDT)

| Timestamp | Event | Source of evidence |
|---|---|---|
| 2026-05-07 00:18 | h32_l2 starts (in-process trainer) | `dashboard/grid_pass100k.json` cells.h32_l2.started_at |
| 2026-05-07 22:03 | h32_l2 finishes 20 epochs | `training_h32_l2/checkpoint_epoch20.pt` mtime |
| 2026-05-07 22:04 | h64_l3 starts | dashboard JSON |
| 2026-05-08 08:39 | h64_l3 last update (epoch 10) — **process dies here** | dashboard JSON + `training_h64_l3/checkpoint_epoch10.pt` mtime |
| 2026-05-08 12:32:33 | `.claude.json` corrupted (zero bytes), Claude Code rebuilds from defaults | `~/.claude/backups/.claude.json.corrupted.1778257953551` (0 bytes) |
| 2026-05-09 00:56 | Diagonal re-launched, PID 624 | log line 1 + `project_pass100k_diagonal_run.md` memory |
| 2026-05-09 ~01:38 | Cache load completes (2536s) | log line 7 |
| 2026-05-09 01:39:53 | Last status JSON write — **process dies here** | `dashboard/grid_pass100k.json` mtime |

The h64_l3 death and the .claude.json corruption straddle the same
~4-hour window on 2026-05-08, strongly suggesting a single VSCode/WSL
crash event killed both.

## Evidence gathered

### What's on disk (good news)

Training output is intact for the cells that completed:

```
training_h32_l2/  → checkpoint_epoch01..epoch20.pt + checkpoint_best.pt + log/png
training_h64_l3/  → checkpoint_epoch01..epoch10.pt + checkpoint_best.pt + log
training_h128_l4/ → empty dir (last night's run never reached an epoch)
```

`grep -c "epoch" runs/v3/grid_pass100k_diagonal_2026-05-09T00-56.log` = **0**.
The diagonal launch loaded the cache and started rotation augmentation, but
died before the first batch.

### What's NOT on disk (the smoking gun)

- `dmesg` after WSL was restarted: empty. No internal Linux OOM kill.
- `/tmp/diagonal_pid` is gone: WSL `/tmp` was wiped. Only happens on
  WSL distro restart.
- `load_observability.log` is from 2026-05-06; nothing was running last
  night to capture the crash trajectory.

### Memory math (why "RAM issue" is plausible but not proven)

The 2026-05-06 observability log showed RSS climbing 0 → 27.4 GB over
~17 minutes during cache load. With 56 GB allocated to WSL and the cache
resident at ~22 GB sparse + transient peaks, it fits comfortably in
WSL — but the host machine sees vmmem balloon to 27+ GB and may apply
its own pressure decisions externally.

The diagonal died ~95 seconds after cache load completed — exactly when
training kicks in (first batch construction, optimizer state allocation,
first forward+backward pass). That's the largest *transient* spike of
the entire run.

## Hypothesis (ranked by confidence)

1. **WSL distro auto-shutdown.** Default WSL2 shuts down idle distros
   after a ~60s window of "no host-foreground client." When Claude/VSCode
   crashed yesterday, the parent-session orphaning meant nothing was
   keeping WSL alive once the launching shell exited. Even with `setsid`,
   the distro itself can be torn down by the host.

2. **Host-level vmmem kill.** Less likely but possible: Windows under
   memory pressure could SIGKILL the WSL VM. We'd see no Linux dmesg
   trace because the kill happens at the hypervisor layer.

3. **First-batch RAM transient.** PyG `Batch.from_data_list` + first
   forward pass is the biggest transient spike of the run. Combined with
   the still-resident chunk-load ghost objects, peak could exceed 30 GB
   transiently. Plausible but doesn't fit dmesg-empty.

(1) is the leading theory because of the `/tmp` wipe evidence — that
specifically signals WSL distro restart, not just a process kill.

## Actions taken this session

### Phase 1 — Cleanup

WSL home (`~`):
- `rm ~/cache_v2_d25_w0w1.pt` (3.4 GB) — superseded v2 cache
- `rm ~/cache_d15_subset.pt` (2.9 GB) — old subset cache
- `rm ~/cache_full.pt` (939 MB) — old monolithic cache
- **Total freed in WSL: 7.3 GB** (60→52 GB used, verified via `df -h`)

Project tree (`runs/v3/`):
- Deleted 4 `2026-05-02T*-e9_v3_data_gen` dirs (predecessors of 100k)
- Deleted 5 `2026-05-02T*-e10_v3_tournament` and `e10b_dual_gnn` dirs
- Deleted `2026-05-05T04-43-e9_v3_data_gen_100k` (partial run, superseded by w12)
- Deleted 9 `grid/2026-05-0[23]T*` and 5 `grid_full20/2026-05-0[34]T*`
  nested e10 tournament dirs
- Deleted matching log files (`e9_*.log`, `e10*.log`,
  `2026-05-05T04-43-e9_v3_data_gen_100k.log`)
- Deleted `grid_pass3_lastepoch.failed-load-bug.{bak,log.bak}`
- **Total freed in project: ~101 MB** (1014→913 MB)

What's preserved:
- `~/catan_cache/cache_100k*.pt` — active training cache (30 GB).
  /mnt/d/catan_v3_cache/ remains as backup copy (29 GB, untouched).
- `runs/v3/training/grid_pass100k/training_*/` — all training checkpoints
- `runs/v3/data_gen/2026-05-05T05-50-e9_v3_data_gen_100k_w12/` — source data
  behind cache_100k.pt (cache rebuild target)
- All `grid_pass3*` and `grid_full20/training_*` checkpoint dirs

### Phase 2 — Crash prevention

1. **`~/.wslconfig`** updated:
   ```
   [wsl2]
   memory=56GB
   vmIdleTimeout=-1   # NEW: never auto-shutdown
   swap=16GB           # NEW: explicit (was implicit default)
   ```
2. **`scripts/observe_memory.py`** (new): standalone background daemon
   that samples `/proc/<pid>/status` + `/proc/meminfo` every 5s while a
   target PID lives. Writes JSON-per-line. Exits cleanly when target dies.
3. **`scripts/launch_diagonal.sh`** rewritten with three guards:
   - Spawns `setsid sleep infinity` keepalive process (holds WSL alive
     past parent shell exit)
   - Spawns `observe_memory.py --pid <trainer>` alongside the trainer
   - Sets `oom_score_adj=-1000` on the trainer (kernel will pick other
     processes first if it ever has to choose)
4. Ran `wsl --shutdown` once to make `.wslconfig` changes effective.

### Phase 3 — This journal

For future-me reading the project tree.

## Re-launch checklist (next session)

```bash
# From PowerShell:
wsl -e bash -c '/mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study/scripts/launch_diagonal.sh'

# From the printed paths, watch:
wsl -e bash -c 'tail -F <log>'
wsl -e bash -c 'tail -F <obs_log>'
```

Cells in execution order: **h128_l4** (fresh, ~22h) then **h64_l3**
(resume from epoch10, ~11h). ETA ~33h wall-clock + 42min cache load.

## Open questions for next time

- If the trainer dies again, the `observ.log` should show RSS just
  before death. Compare against this run to confirm the RAM hypothesis.
- The observed RSS ceiling at training start under sparse cache + h128_l4
  + batch=256 is unknown — first time we'll have it instrumented.
- If memory pressure shows up, the next mitigation is `--batch-size 128`
  (halves the transient) at the cost of ~2× wall-clock.
