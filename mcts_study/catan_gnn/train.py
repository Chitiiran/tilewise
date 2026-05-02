"""GNN-v0 training loop. Reads parquets, trains, writes artifacts.

Per-epoch artifacts (wishlist §4b.1, .4, .8 — added 2026-04-30):
- checkpoint_epoch{NN}.pt — saved after each epoch
- checkpoint_best.pt — copy of the epoch with the best val_top1
- checkpoint.pt — alias for the latest epoch's checkpoint (backward compat)
- progress.png — live training plot regenerated after each epoch
- training_log.json — accumulating per-epoch stats

Resume (wishlist §4b.2): --resume <path> loads model+optimizer state and
continues from the recorded epoch. The cache file is reused as-is.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch

from .dataset import CachedDataset, CatanReplayDataset, RotatedDataset
from .gnn_model import GnnModel


@dataclass
class EpochStats:
    epoch: int
    train_loss_total: float
    train_loss_value: float
    train_loss_policy: float
    val_loss_total: float
    val_loss_value: float
    val_loss_policy: float
    val_value_mae: float
    val_policy_top1_acc: float
    # Per-game val_top1 distribution (wishlist §4b.5). Aggregate val_top1 is
    # noisy because effective n is the # of distinct val games, not positions.
    # These fields show the spread across val games at this epoch.
    val_top1_per_game_min: float = 0.0
    val_top1_per_game_p25: float = 0.0
    val_top1_per_game_median: float = 0.0
    val_top1_per_game_p75: float = 0.0
    val_top1_per_game_max: float = 0.0
    val_n_games: int = 0


def _split_by_seed(ds: CatanReplayDataset, val_frac: float, seed: int):
    rng = random.Random(seed)
    # Pull a per-position seed list from whichever dataset type we got.
    # CatanReplayDataset exposes ._index (pandas frame); CachedDataset exposes .seeds (list).
    if hasattr(ds, "_index"):
        per_pos_seeds = [int(s) for s in ds._index["seed"]]
    elif hasattr(ds, "seeds"):
        per_pos_seeds = list(ds.seeds)
    else:
        raise RuntimeError(
            f"_split_by_seed: dataset {type(ds).__name__} has neither ._index nor .seeds"
        )
    if not per_pos_seeds:
        raise RuntimeError("_split_by_seed: dataset has no per-position seeds")

    seeds = sorted(set(per_pos_seeds))
    rng.shuffle(seeds)
    n_val = max(1, int(len(seeds) * val_frac))
    val_seeds = set(seeds[:n_val])
    train_idx, val_idx = [], []
    for i, s in enumerate(per_pos_seeds):
        if s in val_seeds:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return Subset(ds, train_idx), Subset(ds, val_idx)


def _collate(batch):
    """Custom collate: stack HeteroData via Batch.from_data_list, stack targets."""
    datas, values, policies, legals = zip(*batch)
    return (
        Batch.from_data_list(list(datas)),
        torch.stack(list(values)),
        torch.stack(list(policies)),
        torch.stack(list(legals)),
    )


def _masked_policy_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy where illegal logits are sent to -inf before softmax.

    Note on the masked_fill(0) below: log_softmax over -inf logits returns -inf
    at those positions, and 0 * -inf = NaN in IEEE 754. The dataset guarantees
    target == 0 at illegal positions (visit counts are zero there), so we
    explicitly zero the log-probs there before the multiply.
    """
    masked = logits.masked_fill(~mask, float("-inf"))
    log_probs = F.log_softmax(masked, dim=1)
    log_probs = log_probs.masked_fill(~mask, 0.0)
    # CE = -sum(target * log_probs) per sample, then mean.
    return -(target * log_probs).sum(dim=1).mean()


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent
        ).decode().strip()
        return sha
    except Exception:
        return "unknown"


def _resolve_device(device: str) -> torch.device:
    """device='auto' -> cuda if available else cpu. Otherwise pass-through."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _write_progress_plot(out_path: Path, log: dict, *, epochs_total: int,
                          best_top1: float, best_top1_epoch: int) -> None:
    """Generate a live progress plot from accumulated per-epoch stats.
    Wishlist §4b.8."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = [e["epoch"] for e in log["epochs"]]
    if not eps:
        return
    tl = [e["train_loss_total"] for e in log["epochs"]]
    vl = [e["val_loss_total"] for e in log["epochs"]]
    vt = [e["val_policy_top1_acc"] for e in log["epochs"]]
    pg_min = [e.get("val_top1_per_game_min", 0.0) for e in log["epochs"]]
    pg_max = [e.get("val_top1_per_game_max", 0.0) for e in log["epochs"]]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1]})
    fig.suptitle(
        f"GNN training — epoch {eps[-1]}/{epochs_total}",
        fontsize=12, fontweight="bold",
    )
    ax = axes[0]
    ax.plot(eps, tl, "o-", color="#1f77b4", label="train_loss", linewidth=2)
    ax.plot(eps, vl, "s-", color="#d62728", label="val_loss", linewidth=2)
    ax.set_ylabel("loss")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title(f"final train_loss={tl[-1]:.3f}, val_loss={vl[-1]:.3f}",
                 fontsize=10, color="#444")

    ax = axes[1]
    ax.fill_between(eps, pg_min, pg_max, color="#2ca02c", alpha=0.15,
                    label="per-game min..max")
    ax.plot(eps, vt, "^-", color="#2ca02c", label="val_top1 (mean over positions)", linewidth=2)
    if best_top1_epoch > 0:
        ax.scatter([best_top1_epoch], [best_top1], color="orange", s=140, zorder=5,
                   edgecolor="black", linewidth=1.5,
                   label=f"best: ep{best_top1_epoch} ({best_top1:.3f})")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("val top-1 acc")
    ax.set_xlabel("epoch")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def train_main(
    *,
    run_dirs: list[Path],
    out_dir: Path,
    hidden_dim: int = 32,
    num_layers: int = 2,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    w_value: float = 1.0,
    w_policy: float = 1.0,
    val_frac: float = 0.1,
    seed: int = 0,
    device: str = "auto",
    max_train_samples: int | None = None,
    num_workers: int = 0,
    cache_path: Path | None = None,
    resume_from: Path | None = None,
    init_from: Path | None = None,
    rotate: bool = False,
    rotate_mode: str = "fixed",
    rotate_k: int = 1,
) -> Path:
    """
    max_train_samples: if set, subsample the training set to this many positions
        per epoch. Useful for fast smoke runs when the dataset replay is CPU-bound
        and full epochs would take too long. None = use full train set.
    num_workers: DataLoader worker processes for parallel engine replay. 0 (default)
        uses the main process. Higher values can speed up CPU-bound replay but PyG
        HeteroData multiprocessing has known sharp edges; use with care.
    cache_path: if set, build (or load) a CachedDataset to disk. The first call
        with a given run_dirs replays every position once into RAM and persists
        to cache_path. Subsequent calls load from disk and skip replay entirely.
        This is the GPU-utilization fix: replay-from-scratch is CPU-bound and
        starves the GPU; the cache puts all positions in RAM-resident tensors.
    """
    import time as _time
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    t_load_start = _time.perf_counter()
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            full_ds = CachedDataset(source=None, cache_path=cache_path)
        else:
            source = CatanReplayDataset([Path(p) for p in run_dirs])
            full_ds = CachedDataset(source=source, cache_path=cache_path)
    else:
        full_ds = CatanReplayDataset([Path(p) for p in run_dirs])
    if len(full_ds) == 0:
        raise RuntimeError(f"Empty dataset under {run_dirs}")
    load_secs = _time.perf_counter() - t_load_start
    print(f"[timing] dataset load: {load_secs:.1f}s for {len(full_ds)} positions", flush=True)
    if rotate:
        full_ds = RotatedDataset(full_ds, mode=rotate_mode, k=rotate_k, seed=seed)
        if rotate_mode == "random":
            print(f"[rotate] random hex symmetry per item (k uniform in 0..5)", flush=True)
        else:
            print(f"[rotate] fixed {rotate_k}×60° hex rotation per item", flush=True)
    train_ds, val_ds = _split_by_seed(full_ds, val_frac=val_frac, seed=seed)
    if max_train_samples is not None and len(train_ds) > max_train_samples:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(train_ds), size=max_train_samples, replace=False)
        train_ds = Subset(train_ds, sorted(keep.tolist()))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate,
        num_workers=num_workers,
    )

    dev = _resolve_device(device)
    model = GnnModel(hidden_dim=hidden_dim, num_layers=num_layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Per-game val seed list (wishlist §4b.5). val_ds is a Subset of
    # full_ds; full_ds.seeds[i] gives the seed for position i. The Subset
    # carries indices via .indices.
    if hasattr(full_ds, "seeds"):
        all_seeds = list(full_ds.seeds)
        val_position_seeds = [int(all_seeds[i]) for i in val_ds.indices]
    else:
        # Fallback: CatanReplayDataset uses ._index["seed"].
        per_pos = [int(s) for s in full_ds._index["seed"]]
        val_position_seeds = [per_pos[i] for i in val_ds.indices]

    # Resume support (wishlist §4b.2): if resume_from is set, load state_dict
    # + optimizer state and figure out which epoch to start from.
    start_epoch = 1
    log: dict = {"epochs": []}
    best_top1 = -1.0
    best_top1_epoch = -1
    if resume_from is not None:
        resume_path = Path(resume_from)
        ck = torch.load(resume_path, map_location=dev)
        if "model_state" in ck:
            # Full resume bundle (model + optimizer + epoch + log).
            model.load_state_dict(ck["model_state"])
            if "opt_state" in ck:
                opt.load_state_dict(ck["opt_state"])
            start_epoch = int(ck.get("next_epoch", 1))
            log = ck.get("log", log)
            best_top1 = float(ck.get("best_top1", -1.0))
            best_top1_epoch = int(ck.get("best_top1_epoch", -1))
            print(f"[resume] loaded {resume_path} → resuming from epoch {start_epoch}", flush=True)
        else:
            # Older checkpoint (model state_dict only). Load weights, restart
            # optimizer, count starting epoch from current epochs already done.
            model.load_state_dict(ck)
            print(f"[resume] loaded weights from {resume_path} (no optimizer state)", flush=True)

    # init_from: load only model weights, leave optimizer/log fresh, start_epoch=1.
    # Used for fine-tuning from a prior checkpoint without inheriting its
    # epoch counter or training history.
    if init_from is not None and resume_from is None:
        init_path = Path(init_from)
        ck = torch.load(init_path, map_location=dev)
        if "model_state" in ck:
            model.load_state_dict(ck["model_state"])
        else:
            model.load_state_dict(ck)
        print(f"[init_from] loaded model weights from {init_path} (fresh optimizer + log)", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        # Train.
        model.train()
        tot, tv, tp, n = 0.0, 0.0, 0.0, 0
        t_train_start = _time.perf_counter()
        for batch, value_t, policy_t, legal in train_loader:
            batch = batch.to(dev)
            value_t = value_t.to(dev)
            policy_t = policy_t.to(dev)
            legal = legal.to(dev)
            opt.zero_grad()
            v_pred, p_logits = model(batch)
            lv = F.mse_loss(v_pred, value_t)
            lp = _masked_policy_loss(p_logits, policy_t, legal)
            loss = w_value * lv + w_policy * lp
            loss.backward()
            opt.step()
            tot += loss.item(); tv += lv.item(); tp += lp.item(); n += 1
        train_secs = _time.perf_counter() - t_train_start

        # Val.
        model.eval()
        vt, vv, vp_, vmae, top1, n_pos = 0.0, 0.0, 0.0, 0.0, 0, 0
        # Per-game tracking (wishlist §4b.5): hit/total counts per game seed.
        per_game_hits: dict[int, int] = defaultdict(int)
        per_game_total: dict[int, int] = defaultdict(int)
        val_pos_offset = 0  # tracks our position into val_position_seeds
        t_val_start = _time.perf_counter()
        with torch.no_grad():
            for batch, value_t, policy_t, legal in val_loader:
                batch = batch.to(dev)
                value_t = value_t.to(dev)
                policy_t = policy_t.to(dev)
                legal = legal.to(dev)
                v_pred, p_logits = model(batch)
                lv = F.mse_loss(v_pred, value_t)
                lp = _masked_policy_loss(p_logits, policy_t, legal)
                vt += float(w_value * lv + w_policy * lp); vv += float(lv); vp_ += float(lp)
                vmae += float((v_pred - value_t).abs().mean())
                # Top-1: did argmax(masked logits) hit argmax(target)?
                masked = p_logits.masked_fill(~legal, float("-inf"))
                pred = masked.argmax(dim=1)
                gt = policy_t.argmax(dim=1)
                hits = (pred == gt).cpu().numpy()
                top1 += int(hits.sum())
                n_pos += value_t.shape[0]
                # Attribute hits/totals per source game seed. val_loader is
                # shuffle=False so position order is stable.
                bsz = value_t.shape[0]
                for j in range(bsz):
                    s = val_position_seeds[val_pos_offset + j]
                    per_game_total[s] += 1
                    if hits[j]:
                        per_game_hits[s] += 1
                val_pos_offset += bsz
        val_secs = _time.perf_counter() - t_val_start

        # Per-game val_top1 distribution.
        per_game_acc = sorted(
            per_game_hits[s] / per_game_total[s] for s in per_game_total
        )
        n_games = len(per_game_acc)
        if n_games > 0:
            quart = lambda q: per_game_acc[max(0, min(n_games - 1, int(round(q * (n_games - 1)))))]
            pg_min = per_game_acc[0]
            pg_p25 = quart(0.25)
            pg_p50 = quart(0.50)
            pg_p75 = quart(0.75)
            pg_max = per_game_acc[-1]
        else:
            pg_min = pg_p25 = pg_p50 = pg_p75 = pg_max = 0.0

        stats = EpochStats(
            epoch=epoch,
            train_loss_total=tot / max(1, n), train_loss_value=tv / max(1, n),
            train_loss_policy=tp / max(1, n),
            val_loss_total=vt / max(1, len(val_loader)),
            val_loss_value=vv / max(1, len(val_loader)),
            val_loss_policy=vp_ / max(1, len(val_loader)),
            val_value_mae=vmae / max(1, len(val_loader)),
            val_policy_top1_acc=top1 / max(1, n_pos),
            val_top1_per_game_min=pg_min,
            val_top1_per_game_p25=pg_p25,
            val_top1_per_game_median=pg_p50,
            val_top1_per_game_p75=pg_p75,
            val_top1_per_game_max=pg_max,
            val_n_games=n_games,
        )
        log["epochs"].append(asdict(stats))
        print(f"[epoch {epoch}/{epochs}] train_loss={stats.train_loss_total:.3f} "
              f"val_loss={stats.val_loss_total:.3f} val_top1={stats.val_policy_top1_acc:.3f} "
              f"per_game[{pg_min:.2f}|{pg_p25:.2f}|{pg_p50:.2f}|{pg_p75:.2f}|{pg_max:.2f}] "
              f"[timing] train={train_secs:.1f}s ({n} batches, "
              f"{train_secs * 1000 / max(1, n):.0f}ms/batch) val={val_secs:.1f}s",
              flush=True)

        # === Post-epoch artifacts (wishlist §4b.1, .4, .8) ===
        # Save state_dict on CPU without mutating the live model.
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        # Per-epoch checkpoint as a full bundle (model + optimizer + epoch +
        # log + best_so_far) so resume can pick up cleanly.
        bundle = {
            "model_state": cpu_state,
            "opt_state": opt.state_dict(),
            "next_epoch": epoch + 1,
            "log": log,
            "best_top1": best_top1,
            "best_top1_epoch": best_top1_epoch,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        }
        torch.save(bundle, out_dir / f"checkpoint_epoch{epoch:02d}.pt")
        # Plain weights-only checkpoint for downstream loaders (e7 tournament,
        # bench-2, etc.) that expect torch.save(model.state_dict()).
        torch.save(cpu_state, out_dir / "checkpoint.pt")
        # Best-checkpoint tracking.
        if stats.val_policy_top1_acc > best_top1:
            best_top1 = stats.val_policy_top1_acc
            best_top1_epoch = epoch
            torch.save(cpu_state, out_dir / "checkpoint_best.pt")
            print(f"  ↳ new best_top1={best_top1:.3f} at epoch {epoch} (saved checkpoint_best.pt)",
                  flush=True)
        # Append-only training log (so a kill mid-run still has all stats).
        (out_dir / "training_log.json").write_text(json.dumps(log, indent=2))
        # Live progress plot.
        try:
            _write_progress_plot(out_dir / "progress.png", log,
                                 epochs_total=epochs,
                                 best_top1=best_top1,
                                 best_top1_epoch=best_top1_epoch)
        except Exception as e:
            # Plot is non-critical; don't crash training.
            print(f"  (progress.png write failed: {e})", flush=True)

    # Per-epoch artifacts already wrote checkpoint.pt + training_log.json
    # after the final epoch. Just write the run config below.
    config = {
        "run_dirs": [str(p) for p in run_dirs],
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "w_value": w_value,
        "w_policy": w_policy,
        "val_frac": val_frac,
        "seed": seed,
        "device_resolved": str(dev),
        "dataset_size": len(full_ds),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "git_sha": _git_sha(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    return out_dir


def cli_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w-value", type=float, default=1.0)
    p.add_argument("--w-policy", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Subsample training set to this many positions per epoch (smoke runs).")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader worker processes for parallel engine replay (0 = main process only).")
    p.add_argument("--cache-path", type=Path, default=None,
                   help="Path to a CachedDataset .pt file. If exists, loads from disk; "
                        "otherwise builds the cache from run-dirs and saves there. "
                        "Removes the per-epoch engine-replay bottleneck (GPU starvation fix).")
    p.add_argument("--resume", type=Path, default=None, dest="resume_from",
                   help="Path to a checkpoint_epochNN.pt bundle to resume from. "
                        "Loads model + optimizer state and continues from the saved next_epoch. "
                        "Use this with --epochs > saved_epoch to extend training.")
    p.add_argument("--init-from", type=Path, default=None, dest="init_from",
                   help="Path to a checkpoint to load MODEL WEIGHTS from. Unlike --resume, "
                        "this starts a fresh training run (fresh optimizer, fresh log, "
                        "epoch counter starts at 1). Use for fine-tuning a prior model "
                        "on new data or augmented data.")
    p.add_argument("--rotate", action="store_true",
                   help="Wrap the dataset in RotatedDataset. Same dataset size; each "
                        "position is rotated before training.")
    p.add_argument("--rotate-mode", type=str, default="fixed", choices=["fixed", "random"],
                   help="fixed: same k×60° rotation every time (use --rotate-k). "
                        "random: pick k uniformly from 0..5 per sample (full 6-fold "
                        "hex symmetry augmentation).")
    p.add_argument("--rotate-k", type=int, default=1,
                   help="If --rotate-mode=fixed, apply this many 60° rotations (0..5).")
    args = p.parse_args()
    train_main(
        run_dirs=args.run_dirs, out_dir=args.out_dir,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        w_value=args.w_value, w_policy=args.w_policy, seed=args.seed,
        device=args.device, max_train_samples=args.max_train_samples,
        num_workers=args.num_workers, cache_path=args.cache_path,
        resume_from=args.resume_from,
        init_from=args.init_from,
        rotate=args.rotate,
        rotate_mode=args.rotate_mode,
        rotate_k=args.rotate_k,
    )


if __name__ == "__main__":
    cli_main()
