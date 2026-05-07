"""In-process grid trainer: load cache once, train all cells back-to-back.

Subprocess-per-cell (the existing grid_orchestrator.py model) does one cache
load per cell. For the 100k corpus that's ~30 min × 9 cells = ~4.5h of
redundant loads. This driver loads the CachedDataset ONCE, then loops over
the cells calling train_main(cache_dataset=ds, ...) — saves ~4h.

Phase 0 of the pass-100k roadmap v3.

Status JSON has the same shape as the existing grid_orchestrator output
so the dashboard's "Pass 100k" tab reads it unchanged.

Usage:
    python scripts/train_grid_inproc.py \\
        --cache-path /home/chitii/catan_cache/cache_100k.pt \\
        --out-root runs/v3/grid_pass100k \\
        --status-file runs/v3/dashboard/grid_pass100k.json \\
        --epochs 20 --early-stop-patience 0 --batch-size 256 \\
        --device auto --rotate --rotate-mode random \\
        --cells "h32_l2,h64_l3,h128_l4,h32_l3,h32_l4,h64_l2,h64_l4,h128_l2,h128_l3"
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


DEFAULT_GRID = [
    {"hidden_dim": 32,  "num_layers": 2, "label": "h32_l2"},
    {"hidden_dim": 32,  "num_layers": 3, "label": "h32_l3"},
    {"hidden_dim": 32,  "num_layers": 4, "label": "h32_l4"},
    {"hidden_dim": 64,  "num_layers": 2, "label": "h64_l2"},
    {"hidden_dim": 64,  "num_layers": 3, "label": "h64_l3"},
    {"hidden_dim": 64,  "num_layers": 4, "label": "h64_l4"},
    {"hidden_dim": 128, "num_layers": 2, "label": "h128_l2"},
    {"hidden_dim": 128, "num_layers": 3, "label": "h128_l3"},
    {"hidden_dim": 128, "num_layers": 4, "label": "h128_l4"},
]


def _atomic_write_json(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(blob, indent=2))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _write_orchestrator_state(status_file: Path, **kwargs) -> None:
    blob = _read_json(status_file)
    orch = blob.setdefault("orchestrator", {})
    orch.update(kwargs)
    orch["updated_at"] = time.time()
    blob["updated_at"] = time.time()
    _atomic_write_json(status_file, blob)


def _write_cell_state(status_file: Path, label: str, **kwargs) -> None:
    blob = _read_json(status_file)
    blob.setdefault("cells", {})
    cell = blob["cells"].setdefault(label, {})
    cell.update(kwargs)
    cell["updated_at"] = time.time()
    blob["updated_at"] = time.time()
    _atomic_write_json(status_file, blob)


def _resolve_cells(cells_arg: str) -> list[dict]:
    by_label = {c["label"]: c for c in DEFAULT_GRID}
    if cells_arg == "all":
        return list(DEFAULT_GRID)
    wanted = [c.strip() for c in cells_arg.split(",")]
    out = []
    for w in wanted:
        if w in by_label:
            out.append(by_label[w])
        else:
            print(f"[inproc] WARNING: unknown cell '{w}' skipped", flush=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-path", type=Path, required=True,
                   help="Path to the pre-built CachedDataset (chunked manifest or monolithic)")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Root for per-cell training out-dirs (training_<label>/)")
    p.add_argument("--status-file", type=Path, required=True,
                   help="Dashboard JSON file (single source of truth)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--early-stop-patience", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--rotate", action="store_true", default=True)
    p.add_argument("--no-rotate", dest="rotate", action="store_false")
    p.add_argument("--rotate-mode", type=str, default="random")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (default 0 = main proc)")
    p.add_argument("--cells", type=str, default="all",
                   help="Comma-separated cell labels in execution order, or 'all'")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.status_file.parent.mkdir(parents=True, exist_ok=True)

    cells = _resolve_cells(args.cells)
    if not cells:
        print(f"[inproc] no cells matched '{args.cells}'", flush=True)
        return 1

    # Initial orchestrator state.
    _write_orchestrator_state(
        args.status_file,
        status="loading_cache",
        cells_total=len(cells),
        cells_done=0,
        started_at=time.time(),
        config={
            "cache_path": str(args.cache_path),
            "epochs": args.epochs,
            "early_stop_patience": args.early_stop_patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "rotate": args.rotate,
            "rotate_mode": args.rotate_mode,
            "cells_order": [c["label"] for c in cells],
            "driver": "train_grid_inproc.py",
        },
    )

    # ============== Load the cache ONCE ==============
    print(f"[inproc] loading cache: {args.cache_path}", flush=True)
    t_load = time.perf_counter()
    cache_ds = CachedDataset(source=None, cache_path=args.cache_path, verbose=True)
    load_secs = time.perf_counter() - t_load
    print(f"[inproc] cache loaded in {load_secs:.1f}s ({len(cache_ds)} positions)", flush=True)
    _write_orchestrator_state(args.status_file, status="running",
                               cache_load_secs=load_secs)

    # ============== Loop over cells ==============
    cells_done = 0
    for cell_idx, cell in enumerate(cells):
        label = cell["label"]
        hidden_dim = cell["hidden_dim"]
        num_layers = cell["num_layers"]
        cell_out = args.out_root / f"training_{label}"
        cell_out.mkdir(parents=True, exist_ok=True)
        print(f"\n[inproc] === cell {cell_idx + 1}/{len(cells)}: {label} "
              f"(h={hidden_dim}, l={num_layers}) ===", flush=True)

        _write_cell_state(
            args.status_file, label,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            cell_idx=cell_idx,
            state="training_starting",
            started_at=time.time(),
        )

        try:
            t0 = time.perf_counter()
            train_main(
                run_dirs=[],  # not used when cache_dataset is provided
                out_dir=cell_out,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                num_workers=args.num_workers,
                cache_dataset=cache_ds,  # ← shared, NO reload
                rotate=args.rotate,
                rotate_mode=args.rotate_mode,
                early_stop_patience=args.early_stop_patience,
                status_file=args.status_file,
                status_label=label,
                seed=args.seed,
            )
            elapsed = time.perf_counter() - t0
            print(f"[inproc] {label} done in {elapsed:.1f}s "
                  f"({elapsed/60:.1f} min)", flush=True)
            _write_cell_state(args.status_file, label,
                              state="training_done",
                              training_secs=elapsed)
        except Exception as e:
            print(f"[inproc] {label} FAILED: {e}", flush=True)
            traceback.print_exc()
            _write_cell_state(args.status_file, label,
                              state="training_failed",
                              error=str(e))
            # Keep going to the next cell — one failure shouldn't kill the
            # whole batch since we've already paid the cache load cost.

        # Free GPU + CPU state between cells. Critical: torch caches and
        # Python references can leak across iterations and trigger OOM
        # halfway through the grid.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cells_done += 1
        _write_orchestrator_state(args.status_file, cells_done=cells_done)

    _write_orchestrator_state(args.status_file, status="finished",
                               finished_at=time.time())
    print(f"\n[inproc] all {cells_done}/{len(cells)} cells complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
