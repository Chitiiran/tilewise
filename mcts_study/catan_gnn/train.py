"""GNN-v0 training loop. Reads parquets, trains, writes artifacts."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch

from .dataset import CachedDataset, CatanReplayDataset
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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

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

    log = {"epochs": []}
    for epoch in range(1, epochs + 1):
        # Train.
        model.train()
        tot, tv, tp, n = 0.0, 0.0, 0.0, 0
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

        # Val.
        model.eval()
        vt, vv, vp_, vmae, top1, n_pos = 0.0, 0.0, 0.0, 0.0, 0, 0
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
                top1 += int((pred == gt).sum())
                n_pos += value_t.shape[0]

        stats = EpochStats(
            epoch=epoch,
            train_loss_total=tot / max(1, n), train_loss_value=tv / max(1, n),
            train_loss_policy=tp / max(1, n),
            val_loss_total=vt / max(1, len(val_loader)),
            val_loss_value=vv / max(1, len(val_loader)),
            val_loss_policy=vp_ / max(1, len(val_loader)),
            val_value_mae=vmae / max(1, len(val_loader)),
            val_policy_top1_acc=top1 / max(1, n_pos),
        )
        log["epochs"].append(asdict(stats))
        print(f"[epoch {epoch}/{epochs}] train_loss={stats.train_loss_total:.3f} "
              f"val_loss={stats.val_loss_total:.3f} val_top1={stats.val_policy_top1_acc:.3f}",
              flush=True)

    # Move state_dict tensors to CPU for portable serialization without
    # mutating the model in place — caller may want to keep using it on GPU.
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state, out_dir / "checkpoint.pt")
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2))
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
    args = p.parse_args()
    train_main(
        run_dirs=args.run_dirs, out_dir=args.out_dir,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        w_value=args.w_value, w_policy=args.w_policy, seed=args.seed,
        device=args.device, max_train_samples=args.max_train_samples,
        num_workers=args.num_workers, cache_path=args.cache_path,
    )


if __name__ == "__main__":
    cli_main()
