"""Grid experiment orchestrator: train a 3x3 (hidden, layers) grid and tournament each cell.

Per cell:
  1. Train GNN with given hidden_dim/num_layers, early stopping on val_top1.
  2. Run e10b tournament (PureGnn-cell vs PureGnn-d500 vs LookaheadV3 vs Random).
  3. Update the dashboard JSON.

The dashboard JSON is the single source of truth for the live HTML dashboard;
the training script writes per-epoch entries directly via --status-file, and
this orchestrator writes the tournament block when the tournament finishes.

Cells run sequentially (one cell trains and tournaments before the next starts)
so the GPU is free for whichever stage needs it.

Idempotent: cells already containing a tournament block in the status file
are skipped on rerun. Lets you Ctrl+C and resume cleanly.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# 3x3 grid plus the "current best" baseline cell that's already been trained
# (10k-ep10). The baseline cell isn't trained; we just record it and run a
# tournament against it for context.
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


def _write_cell_state(status_file: Path, label: str, **kwargs) -> None:
    blob = _read_json(status_file)
    blob.setdefault("cells", {})
    cell = blob["cells"].setdefault(label, {})
    cell.update(kwargs)
    cell["updated_at"] = time.time()
    blob["updated_at"] = time.time()
    _atomic_write_json(status_file, blob)


def _write_orchestrator_state(status_file: Path, **kwargs) -> None:
    blob = _read_json(status_file)
    orch = blob.setdefault("orchestrator", {})
    orch.update(kwargs)
    orch["updated_at"] = time.time()
    blob["updated_at"] = time.time()
    _atomic_write_json(status_file, blob)


def _has_tournament(status_file: Path, label: str) -> bool:
    blob = _read_json(status_file)
    return bool(blob.get("cells", {}).get(label, {}).get("tournament"))


def _run_training(
    *,
    label: str,
    hidden_dim: int,
    num_layers: int,
    run_dirs: list[Path],
    out_dir: Path,
    cache_path: Path,
    status_file: Path,
    epochs: int,
    early_stop_patience: int,
    batch_size: int,
    device: str,
    rotate: bool,
    rotate_mode: str,
) -> int:
    cmd = [
        sys.executable, "-m", "catan_gnn.train",
        "--run-dirs", *[str(p) for p in run_dirs],
        "--out-dir", str(out_dir),
        "--hidden-dim", str(hidden_dim),
        "--num-layers", str(num_layers),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--device", device,
        "--cache-path", str(cache_path),
        "--early-stop-patience", str(early_stop_patience),
        "--status-file", str(status_file),
        "--status-label", label,
    ]
    if rotate:
        cmd.extend(["--rotate", "--rotate-mode", rotate_mode])
    print(f"[orchestrator] training cell {label} (h{hidden_dim} l{num_layers})", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def _run_tournament(
    *,
    label: str,
    out_root: Path,
    checkpoint_a: Path,
    checkpoint_b: Path,
    label_a: str,
    label_b: str,
    hidden_dim_a: int,
    num_layers_a: int,
    num_games_per_seating: int,
    base_sims_v3: int,
    lookahead_depth: int,
    seed_base: int,
    workers: int,
    device: str,
) -> int:
    # Tournament needs both checkpoints to share the same model arch loader.
    # Our e10b script uses the --hidden-dim/--num-layers args for BOTH slots,
    # so checkpoint B (d500) must match those args. d500 is h32 l2 — so we
    # only run the tournament with --hidden-dim 32 --num-layers 2 if the cell
    # being tested is also h32 l2. For other architectures, we'd need a
    # variant of e10b that loads each checkpoint with its own arch.
    #
    # Workaround: load the cell-A checkpoint with its native arch (hidden_dim_a,
    # num_layers_a). For cell-B (d500), we use a separate arch (32, 2). The
    # current e10b script doesn't support split-arch — see grid_orchestrator
    # docs. For now, ALL tournament arch loads use the cell's arch; this means
    # B's d500 checkpoint will fail to load if the cell is not h32 l2.
    #
    # Since this is a known limitation we work around by skipping d500 in
    # cells that don't share its arch — replaced with another LookaheadV3
    # bot for the second slot. e10b doesn't currently support that either,
    # so we use a different tournament: e10 (which requires only one
    # checkpoint, the cell's, and runs GnnMcts/PureGnn from it vs LookaheadV3
    # vs Random).
    cmd = [
        sys.executable, "-m", "catan_mcts", "run", "e10",
        "--out-root", str(out_root),
        "--checkpoint", str(checkpoint_a),
        "--num-games-per-seating", str(num_games_per_seating),
        "--sims", "100",
        "--lookahead-depth", str(lookahead_depth),
        "--base-sims-v3", str(base_sims_v3),
        "--hidden-dim", str(hidden_dim_a),
        "--num-layers", str(num_layers_a),
        "--seed-base", str(seed_base),
        "--workers", str(workers),
        "--device", device,
    ]
    print(f"[orchestrator] tournament for cell {label}", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def _aggregate_tournament(run_dir: Path, label: str) -> dict:
    """Read e10's parquet output and compute win rates."""
    import pyarrow.parquet as pq
    import pandas as pd
    from collections import Counter

    cfg_files = list(run_dir.glob("**/config.json"))
    if not cfg_files:
        return {"error": "no config.json found"}
    cfg = json.loads(cfg_files[0].read_text())
    seating_base = cfg.get("seating", ["GnnMcts", "PureGnn", "LookaheadMctsV3", "Random"])
    seed_base = cfg.get("seed_base", 0)

    games_files = list(run_dir.glob("worker*/games.*.parquet"))
    if not games_files:
        return {"error": "no games shards"}
    games = pd.concat([pq.read_table(f).to_pandas() for f in games_files], ignore_index=True)

    def role_at_slot(rot, slot):
        return seating_base[(slot + rot) % 4]

    wins = Counter()
    vps = {r: [] for r in seating_base}
    for _, r in games.iterrows():
        rot = (int(r.seed) - seed_base) // 10_000
        if r.winner >= 0:
            wins[role_at_slot(rot, int(r.winner))] += 1
        for s in range(4):
            vps[role_at_slot(rot, s)].append(int(r.final_vp[s]))

    n = len(games)
    summary = {
        "label": label,
        "n_games": n,
        "wins_by_role": dict(wins),
        "win_rate_by_role": {r: wins.get(r, 0) / max(n, 1) for r in seating_base},
        "mean_vp_by_role": {r: (sum(vs) / max(len(vs), 1)) for r, vs in vps.items()},
        "run_dir": str(run_dir),
    }
    return summary


def _find_latest_run(out_root: Path, prefix: str) -> Path | None:
    candidates = sorted(out_root.glob(f"*-{prefix}"))
    return candidates[-1] if candidates else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Path to the e9 data-gen run dir (parquet shards live in worker*/)")
    p.add_argument("--cache-path", type=Path, required=True,
                   help="Path to the CachedDataset .pt to reuse across cells")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Root for per-cell training out-dirs and tournaments")
    p.add_argument("--status-file", type=Path, required=True,
                   help="Dashboard JSON file (single source of truth)")
    p.add_argument("--epochs", type=int, default=30, help="Max epochs per cell")
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--rotate", action="store_true", default=True)
    p.add_argument("--no-rotate", dest="rotate", action="store_false")
    p.add_argument("--rotate-mode", type=str, default="random")
    # Tournament settings
    p.add_argument("--num-games-per-seating", type=int, default=12,
                   help="48 total games / cell (12 per rotation x 4 rotations)")
    p.add_argument("--base-sims-v3", type=int, default=200)
    p.add_argument("--lookahead-depth", type=int, default=10)
    p.add_argument("--tournament-workers", type=int, default=4)
    p.add_argument("--seed-base", type=int, default=18_000_000)
    # Cell selection
    p.add_argument("--cells", type=str, default="all",
                   help="Comma-separated cell labels to run, or 'all' (default).")
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.status_file.parent.mkdir(parents=True, exist_ok=True)

    if args.cells == "all":
        grid = DEFAULT_GRID
    else:
        wanted = set(c.strip() for c in args.cells.split(","))
        grid = [c for c in DEFAULT_GRID if c["label"] in wanted]
    if not grid:
        print(f"[orchestrator] no cells matched '{args.cells}'", flush=True)
        return 1

    # Initialize orchestrator state.
    _write_orchestrator_state(
        args.status_file,
        status="running",
        cells_total=len(grid),
        cells_done=0,
        started_at=time.time(),
        config={
            "data_dir": str(args.data_dir),
            "epochs": args.epochs,
            "early_stop_patience": args.early_stop_patience,
            "batch_size": args.batch_size,
            "rotate": args.rotate,
            "rotate_mode": args.rotate_mode,
            "num_games_per_seating": args.num_games_per_seating,
            "base_sims_v3": args.base_sims_v3,
            "lookahead_depth": args.lookahead_depth,
        },
    )

    for cell_idx, cell in enumerate(grid):
        label = cell["label"]
        # Resume support: if this cell already has a tournament block, skip.
        if _has_tournament(args.status_file, label):
            print(f"[orchestrator] cell {label} already has tournament — skipping", flush=True)
            continue

        train_out = args.out_root / f"training_{label}"

        # Mark cell starting.
        _write_cell_state(
            args.status_file, label,
            hidden_dim=cell["hidden_dim"],
            num_layers=cell["num_layers"],
            cell_idx=cell_idx,
            state="training_starting",
            started_at=time.time(),
        )

        rc = _run_training(
            label=label,
            hidden_dim=cell["hidden_dim"],
            num_layers=cell["num_layers"],
            run_dirs=[args.data_dir],
            out_dir=train_out,
            cache_path=args.cache_path,
            status_file=args.status_file,
            epochs=args.epochs,
            early_stop_patience=args.early_stop_patience,
            batch_size=args.batch_size,
            device=args.device,
            rotate=args.rotate,
            rotate_mode=args.rotate_mode,
        )
        if rc != 0:
            _write_cell_state(args.status_file, label, state="training_failed", rc=rc)
            print(f"[orchestrator] cell {label} training failed (rc={rc})", flush=True)
            continue
        _write_cell_state(args.status_file, label, state="training_done")

        # Tournament — uses the cell's checkpoint_best.pt
        ckpt = train_out / "checkpoint_best.pt"
        if not ckpt.exists():
            _write_cell_state(args.status_file, label, state="no_checkpoint")
            continue

        # The e10 tournament uses ONE checkpoint (the cell's own) and pits its
        # GnnMcts + PureGnn against LookaheadMctsV3 + Random. d500 comparison
        # has to be handled separately or via the dashboard's diff display.
        rc = _run_tournament(
            label=label,
            out_root=args.out_root,
            checkpoint_a=ckpt,
            checkpoint_b=Path(),  # ignored
            label_a=label,
            label_b="d500",
            hidden_dim_a=cell["hidden_dim"],
            num_layers_a=cell["num_layers"],
            num_games_per_seating=args.num_games_per_seating,
            base_sims_v3=args.base_sims_v3,
            lookahead_depth=args.lookahead_depth,
            seed_base=args.seed_base + cell_idx * 100_000,
            workers=args.tournament_workers,
            device=args.device,
        )
        if rc != 0:
            _write_cell_state(args.status_file, label, state="tournament_failed", rc=rc)
            continue

        # Aggregate the tournament parquet.
        tournament_dir = _find_latest_run(args.out_root, "e10_v3_tournament")
        if tournament_dir:
            summary = _aggregate_tournament(tournament_dir, label)
            _write_cell_state(args.status_file, label, state="done", tournament=summary,
                              finished_at=time.time())
        else:
            _write_cell_state(args.status_file, label, state="tournament_no_dir")

        # Bump orchestrator counter.
        blob = _read_json(args.status_file)
        cells_done = sum(
            1 for c in blob.get("cells", {}).values() if c.get("state") == "done"
        )
        _write_orchestrator_state(args.status_file, cells_done=cells_done)

    _write_orchestrator_state(args.status_file, status="finished", finished_at=time.time())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
