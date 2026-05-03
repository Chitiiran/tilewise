"""Pass-3 final tournament: 120 games per cell on a shared fresh seed range.

Reads each cell's pass-2 checkpoint_best.pt and runs e10 against
LookaheadMctsV3 + Random. All 9 cells use the SAME seed range
(--seed-base 19000000) so they're directly comparable.

Output:
  runs/v3/grid_pass3/<label>/                  (per-cell e10 output)
  runs/v3/dashboard/grid_pass3.json            (live status for dashboard)

Sequential by default — runs cells one at a time so the GPU is dedicated
to each. ~30-60 min per cell, ~5-9 hours total.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from collections import Counter
from pathlib import Path


CELLS = [
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
        try: return json.loads(path.read_text())
        except: pass
    return {}


def _write_state(status_file: Path, **orchestrator_updates):
    blob = _read_json(status_file)
    o = blob.setdefault("orchestrator", {})
    o.update(orchestrator_updates)
    o["updated_at"] = time.time()
    blob["updated_at"] = time.time()
    _atomic_write_json(status_file, blob)


def _write_cell(status_file: Path, label: str, **updates):
    blob = _read_json(status_file)
    cells = blob.setdefault("cells", {})
    cell = cells.setdefault(label, {})
    cell.update(updates)
    cell["updated_at"] = time.time()
    blob["updated_at"] = time.time()
    _atomic_write_json(status_file, blob)


def _aggregate(out_dir: Path, label: str) -> dict | None:
    """Read e10's parquet output and compute win rates."""
    try:
        import pyarrow.parquet as pq
        import pandas as pd
    except ImportError:
        return None
    cfg_files = list(out_dir.glob("**/config.json"))
    if not cfg_files: return None
    try: cfg = json.loads(cfg_files[0].read_text())
    except: return None
    seating_base = cfg.get("seating", ["GnnMcts", "PureGnn", "LookaheadMctsV3", "Random"])
    games_files = list(out_dir.glob("worker*/games.*.parquet"))
    if not games_files: return None
    games = pd.concat([pq.read_table(f).to_pandas() for f in games_files], ignore_index=True)
    seed_base = int(games.seed.min()) if len(games) else 0

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
    return {
        "n_games": n,
        "wins_by_role": dict(wins),
        "win_rate_by_role": {r: wins.get(r, 0) / max(n, 1) for r in seating_base},
        "mean_vp_by_role": {r: (sum(vs) / max(len(vs), 1)) for r, vs in vps.items()},
        "run_dir": str(out_dir),
    }


def _find_latest_run(parent: Path, prefix: str) -> Path | None:
    candidates = sorted(parent.glob(f"*-{prefix}"))
    return candidates[-1] if candidates else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints-root", type=Path, required=True,
                   help="Pass-2 grid out-root (each cell has training_<label>/checkpoint_best.pt)")
    p.add_argument("--out-root", type=Path, required=True,
                   help="Where to write per-cell tournament output")
    p.add_argument("--status-file", type=Path, required=True,
                   help="Dashboard JSON to update live")
    p.add_argument("--num-games-per-seating", type=int, default=30,
                   help="120 games per cell at default (30 per rotation × 4 rotations)")
    p.add_argument("--lookahead-depth", type=int, default=10)
    p.add_argument("--base-sims-v3", type=int, default=200)
    p.add_argument("--seed-base", type=int, default=19_000_000,
                   help="Fresh seed range — must be disjoint from data-gen and earlier tournaments")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--cells", type=str, default="all",
                   help="Comma-separated cell labels, or 'all'")
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.status_file.parent.mkdir(parents=True, exist_ok=True)

    if args.cells == "all":
        grid = CELLS
    else:
        wanted = set(c.strip() for c in args.cells.split(","))
        grid = [c for c in CELLS if c["label"] in wanted]

    _write_state(
        args.status_file,
        status="running", cells_total=len(grid), cells_done=0,
        started_at=time.time(),
        config={
            "checkpoints_root": str(args.checkpoints_root),
            "num_games_per_seating": args.num_games_per_seating,
            "seed_base": args.seed_base,
            "lookahead_depth": args.lookahead_depth,
            "base_sims_v3": args.base_sims_v3,
        },
    )

    for cell_idx, cell in enumerate(grid):
        label = cell["label"]
        cell_out = args.out_root / label
        cell_out.mkdir(parents=True, exist_ok=True)

        ckpt = args.checkpoints_root / f"training_{label}" / "checkpoint_best.pt"
        if not ckpt.exists():
            print(f"[pass3] {label}: no checkpoint at {ckpt} — skipping", flush=True)
            _write_cell(args.status_file, label, state="no_checkpoint",
                        hidden_dim=cell["hidden_dim"], num_layers=cell["num_layers"])
            continue

        # Skip if this cell already has a tournament block.
        existing = _read_json(args.status_file).get("cells", {}).get(label, {})
        if existing.get("tournament"):
            print(f"[pass3] {label}: already has tournament — skipping", flush=True)
            continue

        _write_cell(
            args.status_file, label,
            hidden_dim=cell["hidden_dim"], num_layers=cell["num_layers"],
            state="tournament_running",
            tournament_started_at=time.time(),
        )

        cmd = [
            sys.executable, "-m", "catan_mcts", "run", "e10",
            "--out-root", str(cell_out),
            "--checkpoint", str(ckpt),
            "--num-games-per-seating", str(args.num_games_per_seating),
            "--sims", "100",
            "--lookahead-depth", str(args.lookahead_depth),
            "--base-sims-v3", str(args.base_sims_v3),
            "--hidden-dim", str(cell["hidden_dim"]),
            "--num-layers", str(cell["num_layers"]),
            "--seed-base", str(args.seed_base),  # SAME for all cells
            "--workers", str(args.workers),
            "--device", args.device,
        ]
        print(f"[pass3] cell {cell_idx+1}/{len(grid)}: {label}", flush=True)
        print(f"  cmd: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            _write_cell(args.status_file, label, state="failed", rc=rc)
            continue

        # Find the e10 tournament dir and aggregate.
        e10_dir = _find_latest_run(cell_out, "e10_v3_tournament")
        if not e10_dir:
            _write_cell(args.status_file, label, state="no_dir")
            continue
        summary = _aggregate(e10_dir, label)
        if summary:
            _write_cell(args.status_file, label, state="done",
                        tournament=summary, finished_at=time.time())
        else:
            _write_cell(args.status_file, label, state="aggregate_failed")

        cells_done = sum(
            1 for c in _read_json(args.status_file).get("cells", {}).values()
            if c.get("state") == "done"
        )
        _write_state(args.status_file, cells_done=cells_done)

    _write_state(args.status_file, status="finished", finished_at=time.time())
    print("[pass3] complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
