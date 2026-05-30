"""Background sync: read each cell's training_log.json and update the dashboard JSON.

Solves the "old train.py binary doesn't write status" problem during the running
grid experiment. Reads each cell's training_log.json (which the train script
ALWAYS writes), and patches the dashboard JSON's `cells[label].training` block.

Also infers tournament phase by scanning the orchestrator log + per-cell
e10 tournament directories, and writes a synthetic `tournament_progress`
block so the dashboard can show "rot=2/4 · 8/12 games done" while a
tournament is mid-flight (the old orchestrator binary doesn't do this).

Run this in the background alongside the orchestrator. Idempotent.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, time
from pathlib import Path


def _atomic_write_json(path: Path, blob: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(blob, indent=2))
    os.replace(tmp, path)


def _per_game(epoch_log: dict) -> tuple:
    return (
        epoch_log.get("val_top1_per_game_min", 0.0),
        epoch_log.get("val_top1_per_game_p25", 0.0),
        epoch_log.get("val_top1_per_game_median", 0.0),
        epoch_log.get("val_top1_per_game_p75", 0.0),
        epoch_log.get("val_top1_per_game_max", 0.0),
    )


_TOURNEY_RE = re.compile(r"--checkpoint\s+\S*/training_(\w+)/")


def _reaggregate_tournament(tdir: Path) -> dict | None:
    """Re-read parquet shards and compute correct win rates. Mirrors the
    fixed _aggregate_tournament in grid_orchestrator.py."""
    try:
        import pyarrow.parquet as pq
        import pandas as pd
        from collections import Counter
    except ImportError:
        return None
    cfg_files = list(tdir.glob("worker*/config.json"))
    if not cfg_files:
        return None
    try:
        cfg = json.loads(cfg_files[0].read_text())
    except Exception:
        return None
    seating_base = cfg.get("seating", ["GnnMcts", "PureGnn", "LookaheadMctsV3", "Random"])
    games_files = list(tdir.glob("worker*/games.*.parquet"))
    if not games_files:
        return None
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
        "run_dir": str(tdir),
    }


def _running_tournament_cell() -> str | None:
    """Inspect ps for a running e10 tournament; return its cell label or None."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "args"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if "catan_mcts" not in line or "run e10" not in line:
            continue
        m = _TOURNEY_RE.search(line)
        if m:
            return m.group(1)
    return None


def _latest_tournament_dir(out_root: Path, cell_label: str) -> Path | None:
    """Find the most recent e10 tournament directory containing this cell's
    checkpoint. Heuristic: scan e10_v3_tournament dirs and read their
    config.json's `checkpoint` to match the cell."""
    candidates = sorted(
        out_root.glob("*-e10_v3_tournament*"),
        key=lambda p: p.name, reverse=True,
    )
    for d in candidates:
        cfg_files = list(d.glob("worker*/config.json"))
        if not cfg_files:
            continue
        try:
            cfg = json.loads(cfg_files[0].read_text())
        except Exception:
            continue
        if cell_label in str(cfg.get("checkpoint", "")):
            return d
    return None


def _tournament_progress(tdir: Path, num_games: int = 48) -> dict:
    """Count completed games (parquet shards) and games-per-rotation."""
    files = list(tdir.glob("worker*/games.*.parquet"))
    completed = len(files)
    return {
        "completed_games": completed,
        "expected_games": num_games,
        "pct": completed / num_games if num_games else 0,
    }


def sync(out_root: Path, status_file: Path) -> int:
    """One pass — return number of cells updated."""
    if not status_file.exists():
        return 0
    try:
        blob = json.loads(status_file.read_text())
    except Exception:
        return 0
    cells = blob.get("cells", {})
    updates = 0
    # First pass: re-aggregate any existing tournament blocks. Fixes the
    # off-by-seed_base bug where rot was computed wrong against seed_base=0.
    for label, cell in cells.items():
        if not cell.get("tournament"):
            continue
        tdir = _latest_tournament_dir(out_root, label)
        if not tdir:
            continue
        new_summary = _reaggregate_tournament(tdir)
        if not new_summary:
            continue
        old = cell.get("tournament", {})
        # Always re-aggregate once with the corrected seed_base derivation.
        # Old aggregations might be wrong (seed_base=0 default) — this is a
        # one-shot fix per cell. After the v2 sentinel is set we skip.
        if old.get("_reaggregated_v2") is True:
            continue
        new_summary["_reaggregated_v2"] = True
        cell["tournament"] = new_summary
        cell["updated_at"] = time.time()
        updates += 1

    for label, cell in cells.items():
        log_path = out_root / f"training_{label}" / "training_log.json"
        if not log_path.exists():
            continue
        try:
            log = json.loads(log_path.read_text())
        except Exception:
            continue
        epochs = log.get("epochs", [])
        if not epochs:
            continue
        latest = epochs[-1]
        # Compute best across all epochs.
        best_top1 = 0.0
        best_epoch = 0
        for e in epochs:
            t = e.get("val_policy_top1_acc", 0.0)
            if t > best_top1:
                best_top1 = t
                best_epoch = e.get("epoch", 0)
        epochs_since_best = max(0, latest.get("epoch", 0) - best_epoch)
        pg = _per_game(latest)
        existing = cell.get("training", {})
        # Skip if nothing changed (compare epoch + val_top1 + best).
        if (
            existing.get("epoch") == latest.get("epoch")
            and existing.get("val_top1") == latest.get("val_policy_top1_acc")
            and existing.get("best_top1") == best_top1
        ):
            continue
        # Determine state. If training is still alive (latest epoch < epochs_total
        # and we haven't seen an "early_stopped" line), call it 'training'.
        # If epoch == epochs_total, training completed.
        epochs_total = log.get("epochs_total")
        if epochs_total is None:
            cfg_path = out_root / f"training_{label}" / "config.json"
            if cfg_path.exists():
                try:
                    epochs_total = json.loads(cfg_path.read_text()).get("epochs", 30)
                except Exception:
                    epochs_total = 30
            else:
                epochs_total = 30
        cell["training"] = {
            "state": existing.get("state", "training"),
            "epoch": latest.get("epoch", 0),
            "epochs_total": epochs_total,
            "train_loss": latest.get("train_loss_total", 0.0),
            "val_loss": latest.get("val_loss_total", 0.0),
            "val_top1": latest.get("val_policy_top1_acc", 0.0),
            "per_game_min": pg[0], "per_game_p25": pg[1], "per_game_median": pg[2],
            "per_game_p75": pg[3], "per_game_max": pg[4],
            "best_top1": best_top1, "best_top1_epoch": best_epoch,
            "epochs_since_best": epochs_since_best,
            "early_stop_patience": cell.get("training", {}).get("early_stop_patience", 3),
            "train_secs": latest.get("train_secs", 0.0),
            "val_secs": latest.get("val_secs", 0.0),
            "updated_at": time.time(),
            "synced_by": "dashboard_sync.py",
        }
        cell["updated_at"] = time.time()
        updates += 1
    # Detect a running tournament and update progress.
    running_cell = _running_tournament_cell()
    if running_cell and running_cell in cells:
        cell = cells[running_cell]
        # Set tournament_started_at if missing (best-effort: now if we don't know).
        if not cell.get("tournament_started_at"):
            cell["tournament_started_at"] = time.time()
        # Mark cell state as tournament_running so dashboard shows purple.
        if cell.get("state") in ("training_done", "tournament_running") or (
            cell.get("training", {}).get("state") == "early_stopped"
            and not cell.get("tournament")
        ):
            cell["state"] = "tournament_running"
        # Find the tournament dir and count parquet shards.
        tdir = _latest_tournament_dir(out_root, running_cell)
        if tdir:
            cfg_files = list(tdir.glob("worker*/config.json"))
            num_games = 48
            if cfg_files:
                try:
                    cfg = json.loads(cfg_files[0].read_text())
                    num_games = cfg.get("num_games_per_seating", 12) * 4
                except Exception:
                    pass
            progress = _tournament_progress(tdir, num_games=num_games)
            cell["tournament_progress"] = {
                "completed": progress["completed_games"],
                "total": progress["expected_games"],
                "pct": progress["pct"],
                "tournament_dir": str(tdir),
                "updated_at": time.time(),
            }
            cell["updated_at"] = time.time()
            updates += 1
    if updates:
        blob["updated_at"] = time.time()
        _atomic_write_json(status_file, blob)
    return updates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--status-file", type=Path, required=True)
    p.add_argument("--interval", type=int, default=20, help="seconds between syncs")
    args = p.parse_args()
    while True:
        try:
            n = sync(args.out_root, args.status_file)
            if n > 0:
                print(f"[sync] updated {n} cell(s)", flush=True)
        except Exception as e:
            print(f"[sync] error: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
