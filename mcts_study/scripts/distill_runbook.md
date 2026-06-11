# Distillation teacher data-gen RUNBOOK (2026-06-10 overnight)

## What is running

5 × `self_play_async` processes (AZ-style teacher self-play), all four seats =
GnnMcts@200 on `round0_Cell6.pt`, full Catan (`vp_target=10, bonuses=True`),
Dirichlet+temperature exploration (`--self-play`), GPU batched evaluator.

| proc | seed_base | num_games | n_concurrent | log |
|---|---|---|---|---|
| P1 | 21,000,000 | 2000 (ceiling) | 64 | `…/distill/teacher_p1.log` |
| P2 | 22,000,000 | 500 | 24 | `…/distill/teacher_p2.log` |
| P3 | 23,000,000 | 500 | 24 | `…/distill/teacher_p3.log` |
| P4 | 24,000,000 | 500 | 24 | `…/distill/teacher_p4.log` |
| P5 | 25,000,000 | 500 | 24 | `…/distill/teacher_p5.log` |

All under `/home/chitii/catan_data/runs/v3/distill/<ts>-self_play_async/`,
`--max-seconds 21600` (6 h whole-run cap), `--ram-budget-mb` set, per-game
parquet flush + `done.txt` (resumable).

**Why 5 processes (measured):** one asyncio process is GIL-bound ≈ 1 core
(8.3% user CPU on 12 vCPUs observed at 00:45) and used 535 MiB VRAM at 14%
GPU util. 5 processes ≈ 5 cores, ~2.7 GB VRAM on the 4 GB GTX 1650.
(6c/12t Ryzen 5600H — more processes would thrash cores and risk VRAM OOM.)

## Resume after crash / WSL restart

```bash
source ~/catan_mcts_venvs/mcts-study/bin/activate
cd /mnt/c/dojo/catan_bot/.claude/worktrees/az-bots/mcts_study
# Find each run dir (one per process):
ls /home/chitii/catan_data/runs/v3/distill/
# Re-launch each with --resume-dir <run_dir> and its ORIGINAL seed_base /
# num_games / n_concurrent from the table above, e.g. P2:
nohup setsid python -m catan_mcts.experiments.self_play_async \
  --out-root runs/v3/distill \
  --checkpoint runs/v3/rl_checkpoints/round0_Cell6.pt \
  --num-games 500 --n-sims 200 --n-concurrent 24 --max-batch 24 \
  --self-play --seed-base 22000000 --ram-budget-mb 8000 \
  --device cuda --max-seconds 21600 \
  --resume-dir <run_dir> > /home/chitii/catan_data/runs/v3/distill/teacher_p2.log 2>&1 &
```

WSL itself failed to start once tonight ("Failed to attach disk
D:/wsl-distros/Ubuntu/ext4.vhdx") — fixed by `wsl --shutdown` + retry.

## Next steps (Phase 3, after data accumulates)

1. Train student (warm-start Cell 6, distillation target):
```bash
python -m catan_gnn.train \
  --run-dirs runs/v3/distill/<all run dirs> \
  --out-dir runs/v3/training/distill_v1 \
  --hidden-dim 128 --num-layers 4 --epochs 10 \
  --init-from runs/v3/rl_checkpoints/round0_Cell6.pt \
  --policy-sharpen 2.0 --device cuda
```
2. Gate (e10g harness): student-argmax vs raw-Cell6-argmax vs LookV3,
   120 games, shared seeds, <5% timeouts, report 95% CI.
3. Promote to web `medium`/`hard` preset only if outside CI vs raw argmax.
