"""Batch-size sweep on a single cell to characterize speed-vs-quality tradeoff.

Plan:
  - Fix architecture (h32_l2 — smallest, fastest per-step)
  - Fix seed=0, fix data split, fix LR=1e-3
  - Vary batch_size in {256, 512, 1024, 2048, 4096}
  - Train each variant for 5 epochs (enough to characterize plateau)
  - Cache loaded ONCE in this process, shared across all 5 sweep runs

What we measure per variant:
  - epoch wall-clock time (steady-state, ignoring epoch 1 startup cost)
  - val_top1 trajectory (epochs 1..5)
  - val_top1 best
  - whether GPU OOM'd (record and skip the variant)

Output:
  - Per-variant training_log.json under runs/v3/grid_pass100k_sweep/training_h32_l2_b{bs}/
  - A summary file runs/v3/grid_pass100k_sweep/sweep_summary.json comparing all
  - Logs stream to runs/v3/grid_pass100k_sweep.log
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from catan_gnn.dataset import CachedDataset
from catan_gnn.train import train_main


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-path", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True,
                   help="Root for per-variant training out-dirs")
    p.add_argument("--batch-sizes", type=str, default="256,512,1024,2048,4096",
                   help="Comma-separated batch sizes to sweep")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Held constant across variants for clean comparison. "
                        "(Note: optimal LR scales with sqrt(batch_size); "
                        "we keep it fixed to isolate the batch-size effect.)")
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--rotate", action="store_true", default=True)
    p.add_argument("--no-rotate", dest="rotate", action="store_false")
    p.add_argument("--rotate-mode", type=str, default="random")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",") if b.strip()]

    # ============== Load cache ONCE ==============
    print(f"[sweep] loading cache: {args.cache_path}", flush=True)
    t0 = time.perf_counter()
    cache_ds = CachedDataset(source=None, cache_path=args.cache_path, verbose=True)
    load_secs = time.perf_counter() - t0
    print(f"[sweep] cache loaded in {load_secs:.1f}s ({len(cache_ds)} positions)", flush=True)

    # ============== Run each batch size ==============
    summary = {
        "config": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "rotate": args.rotate,
            "rotate_mode": args.rotate_mode,
            "cache_path": str(args.cache_path),
        },
        "variants": [],
    }

    for bs in batch_sizes:
        out_dir = args.out_root / f"training_h{args.hidden_dim}_l{args.num_layers}_b{bs}"
        print(f"\n[sweep] === batch_size={bs} ===", flush=True)
        print(f"[sweep] out_dir: {out_dir}", flush=True)

        variant = {
            "batch_size": bs,
            "out_dir": str(out_dir),
            "started_at": time.time(),
        }

        try:
            train_t0 = time.perf_counter()
            train_main(
                run_dirs=[],
                out_dir=out_dir,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                epochs=args.epochs,
                batch_size=bs,
                lr=args.lr,
                device=args.device,
                num_workers=args.num_workers,
                cache_dataset=cache_ds,
                rotate=args.rotate,
                rotate_mode=args.rotate_mode,
                early_stop_patience=0,
                seed=args.seed,
            )
            elapsed = time.perf_counter() - train_t0
            variant["elapsed_secs"] = elapsed
            variant["status"] = "ok"

            # Read training_log.json to get per-epoch stats
            log_path = out_dir / "training_log.json"
            if log_path.exists():
                log = json.loads(log_path.read_text())
                epochs = log.get("epochs", [])
                # Per-epoch durations (train+val) for each completed epoch
                per_epoch_secs = []
                val_top1_traj = []
                for ep in epochs:
                    train_secs = ep.get("train_secs", 0)
                    val_secs = ep.get("val_secs", 0)
                    per_epoch_secs.append(train_secs + val_secs)
                    val_top1_traj.append(ep.get("val_policy_top1_acc", 0))
                variant["per_epoch_secs"] = per_epoch_secs
                variant["val_top1_per_epoch"] = val_top1_traj
                variant["best_val_top1"] = max(val_top1_traj) if val_top1_traj else 0
                variant["best_val_top1_epoch"] = (
                    val_top1_traj.index(max(val_top1_traj)) + 1 if val_top1_traj else -1
                )
                # Steady-state epoch time = mean of epochs 2..N (skip warm-up)
                if len(per_epoch_secs) >= 2:
                    variant["steady_epoch_secs"] = sum(per_epoch_secs[1:]) / len(per_epoch_secs[1:])
                else:
                    variant["steady_epoch_secs"] = per_epoch_secs[0] if per_epoch_secs else 0
        except torch.cuda.OutOfMemoryError as oom:
            variant["status"] = "cuda_oom"
            variant["error"] = str(oom)
            print(f"[sweep] CUDA OOM at batch_size={bs} — skipping", flush=True)
            traceback.print_exc()
        except Exception as e:
            variant["status"] = "error"
            variant["error"] = str(e)
            print(f"[sweep] FAILED at batch_size={bs}: {e}", flush=True)
            traceback.print_exc()

        variant["finished_at"] = time.time()
        summary["variants"].append(variant)

        # Cleanup state between variants — critical to avoid creep
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Write summary after each variant so we have partial results if we crash
        summary_path = args.out_root / "sweep_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

    # ============== Final report ==============
    print("\n" + "=" * 70)
    print("BATCH-SIZE SWEEP RESULTS")
    print("=" * 70)
    print(f"{'batch_size':>10}  {'status':>10}  {'best_val_top1':>14}  "
          f"{'best_ep':>8}  {'epoch_secs':>11}  {'speedup':>8}")
    baseline_epoch_secs = None
    for v in summary["variants"]:
        bs = v["batch_size"]
        status = v.get("status", "?")
        best = v.get("best_val_top1", 0)
        best_ep = v.get("best_val_top1_epoch", -1)
        epoch_secs = v.get("steady_epoch_secs", 0)
        if baseline_epoch_secs is None and status == "ok":
            baseline_epoch_secs = epoch_secs
            speedup = "1.0x"
        elif baseline_epoch_secs and epoch_secs:
            speedup = f"{baseline_epoch_secs/epoch_secs:.2f}x"
        else:
            speedup = "-"
        print(f"{bs:>10d}  {status:>10s}  {best:>13.4f}  {best_ep:>8d}  "
              f"{epoch_secs:>10.1f}s  {speedup:>8s}")

    print(f"\nFull summary: {args.out_root / 'sweep_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
