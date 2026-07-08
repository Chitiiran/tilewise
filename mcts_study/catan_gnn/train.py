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
import os
import random
import re
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
    val_value_mse: float
    val_value_sign_acc: float
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


def class_balanced_target(target: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Cand 7: rebalance per-sample policy target so each action_class
    contributes the same gradient share regardless of how many action_ids
    it occupies.

    For each sample s and each legal action a:
        adjusted[s, a] = target[s, a] / class_count[s, class_of(a)]
    where class_count[s, c] = number of legal action_ids in class c for sample s.
    Then renormalize so each sample's adjusted target sums to 1.

    Samples whose target is all-zero (degenerate) stay all-zero — the
    masked policy loss handles zero targets without NaN.

    Cited rationale: ProposeTrade spans 20 action_ids, BuildRoad spans 72.
    Without rebalancing, the model's gradient share on roads/trades is
    proportional to their action-id count rather than their decision
    importance.
    """
    from .action_classes import ACTION_CLASS_ID, NUM_ACTION_CLASSES

    cls_id = ACTION_CLASS_ID.to(target.device)            # (280,)
    legal_f = legal.to(target.dtype)
    # Per-sample per-class legal count via scatter_add.
    # class_count[s, c] = sum_{a: ACTION_CLASS_ID[a]==c} legal[s, a]
    batch_size = target.shape[0]
    class_count = torch.zeros(
        batch_size, NUM_ACTION_CLASSES, device=target.device, dtype=target.dtype,
    )
    class_count.scatter_add_(1, cls_id.unsqueeze(0).expand(batch_size, -1), legal_f)
    # Per-action divisor: class_count[s, class_of(a)]. Clamp to avoid /0
    # (those actions are illegal anyway and have target 0, so the result
    # at those positions is 0/clamped = 0).
    per_action_count = class_count.gather(1, cls_id.unsqueeze(0).expand(batch_size, -1))
    divisor = per_action_count.clamp(min=1.0)
    adjusted = target / divisor
    # Renormalize per sample. Samples with all-zero target stay zero.
    row_sum = adjusted.sum(dim=-1, keepdim=True)
    safe_sum = row_sum.clamp(min=1e-12)
    adjusted = torch.where(row_sum > 0, adjusted / safe_sum, adjusted)
    return adjusted


def sharpen_policy_target(target: torch.Tensor, p: float) -> torch.Tensor:
    """Distillation target: raise the (normalized) visit distribution to the
    p-th power and renormalize per row. (v/s)^p ∝ v^p, so this equals
    sharpening raw visit counts.

    Why: the 2026-06-01 deploy-valley diagnosis — Catan states have several
    near-equal-value moves, visit targets are honestly soft, and argmax
    deployment discards the value signal that separates them. Sharpening the
    GnnMcts@200 teacher's visits concentrates probability on the searcher's
    DECISION so the student's argmax inherits it (recommended target form,
    puregnn-deploy-investigation-FINAL §4).

    p=1 is the identity; exact ties stay tied; all-zero rows stay zero (the
    masked policy loss handles them without NaN). Detached — a target
    transform, not a loss term.
    """
    if p == 1.0:
        return target
    with torch.no_grad():
        powered = target.pow(p)
        row_sum = powered.sum(dim=-1, keepdim=True)
        safe = row_sum.clamp(min=1e-12)
        return torch.where(row_sum > 0, powered / safe, powered)


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


def _vp_prior_loss(logits: torch.Tensor, vp_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cand 8 auxiliary loss: KL(policy || vp_prior) over legal actions.

    Same form as _masked_policy_loss (cross-entropy against a target
    distribution), but the target is the action-class VP prior from
    catan_gnn.action_classes.build_vp_prior_target. Cited plan:
        loss += lambda_vp * KL(softmax(logits over legal) || vp_target)

    Since KL(p || q) = -sum(q * log(p)) + sum(q * log(q)), and the second
    term has no gradient w.r.t. logits, we can use the cross-entropy
    form -sum(vp_target * log_softmax(logits)) which is equivalent for
    gradient purposes.
    """
    masked = logits.masked_fill(~mask, float("-inf"))
    log_probs = F.log_softmax(masked, dim=1)
    log_probs = log_probs.masked_fill(~mask, 0.0)
    return -(vp_target * log_probs).sum(dim=1).mean()


def _parse_worktree_sha(cwd: Path) -> str | None:
    """Textual fallback for `git rev-parse HEAD` when the git binary itself
    can't resolve the repo (BUG B, shake-out journal 2026-06-19 §6): a git
    WORKTREE's `.git` file is `gitdir: <main>/.git/worktrees/<name>`. On a
    Windows worktree checked out under WSL, that path is a Windows path
    (`C:/...`) which is meaningless as a WSL cwd, so `git rev-parse` fails
    with "fatal: not a git repository" and the run silently loses its SHA.

    Parses the layout by hand instead of shelling out:
      <worktree>/.git            -> "gitdir: <main>/.git/worktrees/<name>"
      <main>/.git/worktrees/<name>/HEAD -> "ref: refs/heads/<branch>"
      <main>/.git/refs/heads/<branch>   -> "<40-hex sha>\n"
    (falls back to <main>/.git/packed-refs if the loose ref file is absent).

    Returns the 40-hex SHA, or None if any step of the chain is unavailable
    (caller is responsible for the final "unknown" sentinel — never raises).
    """
    try:
        # mirror git's own behaviour: walk UP from cwd looking for `.git`
        # (real `git rev-parse` does this automatically; our textual parser
        # must too, or it only works when cwd IS the worktree root).
        dot_git = None
        for candidate in (Path(cwd), *Path(cwd).resolve().parents):
            probe = candidate / ".git"
            if probe.is_file() or probe.is_dir():
                dot_git = probe
                break
        if dot_git is None or not dot_git.is_file():
            return None
        text = dot_git.read_text().strip()
        if not text.startswith("gitdir:"):
            return None
        gitdir = text.split(":", 1)[1].strip()
        # Windows path (worktree checked out on Windows, read from WSL) ->
        # translate C:/... or C:\... to /mnt/c/... so the path resolves here.
        m = re.match(r"^([A-Za-z]):[/\\](.*)$", gitdir)
        if m and not Path(gitdir).exists():
            drive, rest = m.group(1).lower(), m.group(2).replace("\\", "/")
            gitdir = f"/mnt/{drive}/{rest}"
        gitdir_path = Path(gitdir)
        head_file = gitdir_path / "HEAD"
        if not head_file.is_file():
            return None
        head_text = head_file.read_text().strip()
        if head_text.startswith("ref:"):
            ref = head_text.split(":", 1)[1].strip()
            # gitdir for a worktree is <main>/.git/worktrees/<name>; refs for
            # the main repo live at <main>/.git/refs/... (two levels up).
            main_git_dir = gitdir_path.parent.parent
            ref_file = main_git_dir / ref
            if ref_file.is_file():
                sha = ref_file.read_text().strip()
                if re.fullmatch(r"[0-9a-fA-F]{40}", sha):
                    return sha
            # loose ref not found (or already packed) -> check packed-refs.
            packed = main_git_dir / "packed-refs"
            if packed.is_file():
                for line in packed.read_text().splitlines():
                    if line.endswith(f" {ref}"):
                        sha = line.split(" ", 1)[0].strip()
                        if re.fullmatch(r"[0-9a-fA-F]{40}", sha):
                            return sha
            return None
        # detached HEAD: HEAD file itself holds the sha.
        if re.fullmatch(r"[0-9a-fA-F]{40}", head_text):
            return head_text
        return None
    except OSError:
        return None


def _git_sha(cwd: Path | None = None) -> str:
    """Commit SHA for reproducibility metadata (ENGINEERING-PRINCIPLES §1/§7).

    Fallback chain (BUG B, shake-out journal 2026-06-19 §6):
      1. AZ_GIT_SHA env var, if set and non-empty (the launch path sets this
         once, driver-side, so subprocess workers never need to re-derive it).
      2. plain `git rev-parse HEAD` (works Windows-side or in a normal clone).
      3. textual worktree parse (works when git itself can't resolve a
         Windows-path `gitdir:` from a WSL cwd).
      4. "unknown" + a logged warning — NEVER a silent empty string.
    """
    env_sha = os.environ.get("AZ_GIT_SHA", "").strip()
    if env_sha:
        return env_sha

    cwd = Path(cwd) if cwd is not None else Path(__file__).parent
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd
        ).decode().strip()
    except Exception:
        pass

    sha = _parse_worktree_sha(cwd)
    if sha:
        return sha

    print(f"[git_sha] WARNING: could not resolve a commit SHA from {cwd} "
          "(git failed and no worktree files were readable); stamping "
          "'unknown'", flush=True)
    return "unknown"


def _resolve_device(device: str) -> torch.device:
    """device='auto' -> cuda if available else cpu. Otherwise pass-through."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _write_training_status(
    *,
    status_file: Path,
    label: str,
    epoch: int,
    epochs_total: int,
    train_loss: float,
    val_loss: float,
    val_top1: float,
    per_game: tuple,
    best_top1: float,
    best_top1_epoch: int,
    epochs_since_best: int,
    early_stop_patience: int,
    state: str,
    train_secs: float,
    val_secs: float,
) -> None:
    """Write a per-cell training status block into a shared dashboard JSON.

    Atomic via temp-file + rename so the dashboard never reads a partial write.
    Layout: status_file is the path to a JSON file shared across all cells.
    Each cell writes/updates `cells[label]` with its current state.
    """
    import json as _json
    import os as _os
    import time as _time2

    status_file = Path(status_file)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    # Read existing JSON (or start fresh).
    if status_file.exists():
        try:
            blob = _json.loads(status_file.read_text())
        except Exception:
            blob = {}
    else:
        blob = {}
    blob.setdefault("cells", {})
    cell = blob["cells"].setdefault(label, {})
    cell["training"] = {
        "state": state,
        "epoch": epoch,
        "epochs_total": epochs_total,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_top1": float(val_top1),
        "per_game_min": float(per_game[0]),
        "per_game_p25": float(per_game[1]),
        "per_game_median": float(per_game[2]),
        "per_game_p75": float(per_game[3]),
        "per_game_max": float(per_game[4]),
        "best_top1": float(best_top1),
        "best_top1_epoch": int(best_top1_epoch),
        "epochs_since_best": int(epochs_since_best),
        "early_stop_patience": int(early_stop_patience),
        "train_secs": float(train_secs),
        "val_secs": float(val_secs),
        "updated_at": _time2.time(),
    }
    blob["updated_at"] = _time2.time()
    tmp = status_file.with_suffix(status_file.suffix + ".tmp")
    tmp.write_text(_json.dumps(blob, indent=2))
    _os.replace(tmp, status_file)


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
    lambda_vp: float = 0.0,
    vp_compare_rule: bool = False,
    lambda_settle: float = 0.0,
    lambda_road: float = 0.0,
    class_balanced_policy: bool = False,
    policy_sharpen: float = 1.0,
    val_frac: float = 0.1,
    seed: int = 0,
    device: str = "auto",
    max_train_samples: int | None = None,
    num_workers: int = 0,
    cache_path: Path | None = None,
    cache_dataset=None,
    resume_from: Path | None = None,
    init_from: Path | None = None,
    rotate: bool = False,
    rotate_mode: str = "fixed",
    rotate_k: int = 1,
    early_stop_patience: int = 0,
    status_file: Path | None = None,
    status_label: str = "",
    pause_dir: Path | None = None,
    mid_tournament_every: int = 0,
    mid_tournament_games_per_seating: int = 30,
    mid_tournament_drop_threshold: int = 3,
    mid_tournament_sims: int = 100,
    mid_tournament_lookahead_depth: int = 10,
    mid_tournament_base_sims_v3: int = 200,
    mid_tournament_seed_base: int = 19_000_000,  # Phase 1 standard (pass-3 baseline range)
    mid_tournament_max_seconds: float = 600.0,
    mid_tournament_workers: int = 10,
    mid_tournament_device: str = "cuda",
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
    cache_dataset: pre-loaded CachedDataset instance. If provided, skips ALL
        loading (cache_path AND run_dirs are ignored for dataset construction).
        Used by scripts/train_grid_inproc.py to share one loaded cache across
        multiple cells in a single Python process — saves the ~30-min cache
        deserialization cost per cell.
    lambda_road: Cand 11 (pure-pip road prior). Weight on the auxiliary KL
        pulling the model's softmax-over-legal-roads toward a distribution
        proportional to pip(far endpoint of the road) when that endpoint
        is settlement-legal. Gate A: fires only when NO legal settlement
        action exists in the sample. Layer 1: independent softmax over
        road slice (no global logit inflation). Default 0 (off). Cell 5
        first run uses 0.05.
    mid_tournament_every: if > 0, run a PureGnn-vs-LookaheadV3 tournament
        every N epochs (after the per-epoch checkpoint save). Used as the
        early-stopping signal for the loss-augmentation roadmap, since
        val_top1 is by-design pulled away from MCTS visits by Cands 8/10
        and no longer correlates with tournament winrate. Default 0 (off).
    mid_tournament_games_per_seating: per-rotation game count. 30 × 4
        rotations = 120 games per tournament (the plan's standard).
    mid_tournament_drop_threshold: stop iff PureGnn wins drop by this many
        games vs the previous mid-training tournament. Default 3 (~2.5pp
        at 120 games). First tournament always passes (no prior).
    mid_tournament_sims / lookahead_depth / base_sims_v3 / seed_base /
    max_seconds: pass-through to e10_v3_tournament.main.
    """
    import time as _time
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    t_load_start = _time.perf_counter()
    if cache_dataset is not None:
        full_ds = cache_dataset
    elif cache_path is not None:
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
    # Per-epoch-seeded shuffle generator: epoch K's data order is seed+K,
    # INDEPENDENT of how K is reached (in-process vs resumed-from-checkpoint).
    # This makes pause/resume produce a run IDENTICAL to a never-paused run —
    # the replayability contract for training (re-seeded at each epoch top).
    _shuffle_gen = torch.Generator()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate,
        num_workers=num_workers, generator=_shuffle_gen,
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

    # Early-stopping counter: number of consecutive epochs with no improvement
    # over best_top1. Resets to 0 whenever we set a new best.
    epochs_since_best = 0

    # Mid-training tournament history: list of PureGnn win counts (int)
    # from past mid-training tournaments, oldest first. Used by the
    # drop-based early-stop decision rule.
    mid_tournament_history: list[int] = []
    log.setdefault("mid_tournaments", [])

    # Initial dashboard write so the dashboard knows training has begun
    # the cache load + first epoch can be 10-20 min on big datasets, and
    # without this write the dashboard sits on `training_starting` for
    # that whole window.
    if status_file is not None:
        try:
            _write_training_status(
                status_file=status_file,
                label=status_label,
                epoch=0,
                epochs_total=epochs,
                train_loss=0.0,
                val_loss=0.0,
                val_top1=0.0,
                per_game=(0.0, 0.0, 0.0, 0.0, 0.0),
                best_top1=0.0,
                best_top1_epoch=0,
                epochs_since_best=0,
                early_stop_patience=early_stop_patience,
                state="warming_up",
                train_secs=0.0,
                val_secs=0.0,
            )
        except Exception:
            pass

    # Per-batch observability cadence: log every N batches and update the
    # dashboard so long epochs aren't silent. Per memory
    # feedback_training_observability.md (2026-05-25 lesson from Cell 5).
    batches_total = len(train_loader)
    progress_every = max(1, batches_total // 50)  # ~50 lines/epoch

    for epoch in range(start_epoch, epochs + 1):
        # Deterministic per-epoch shuffle: order depends only on (seed, epoch),
        # so a resumed run reproduces the same epoch orders as a straight run.
        _shuffle_gen.manual_seed(seed * 1_000_003 + epoch)
        # Train.
        model.train()
        tot, tv, tp, n = 0.0, 0.0, 0.0, 0
        t_train_start = _time.perf_counter()
        last_progress_time = _time.perf_counter()
        # Track Cand 10 swap rate per epoch.
        vp_swap_total = 0
        vp_swap_samples = 0
        for batch, value_t, policy_t, legal in train_loader:
            batch = batch.to(dev)
            value_t = value_t.to(dev)
            policy_t = policy_t.to(dev)
            legal = legal.to(dev)
            opt.zero_grad()
            v_pred, p_logits = model(batch)
            # Cand 10: 1-step VP comparison target swap. Modifies policy_t
            # in place for samples where the model's argmax is a strictly
            # higher-VP action than the teacher's. Cited: in v3 with
            # bonuses=False, vp-delta is fully determined by action_class.
            if vp_compare_rule:
                from .vp_compare import vp_compare_swap_target
                # Detach: this is just a target rewrite, not a loss term;
                # gradient should flow only through the CE on the new target.
                with torch.no_grad():
                    policy_t, swap_count = vp_compare_swap_target(
                        p_logits.detach(), policy_t, legal,
                    )
                vp_swap_total += swap_count
                vp_swap_samples += policy_t.shape[0]
            # Cand 7: action-class-balanced policy target. Rewrite target so
            # each action_class gets equal gradient share regardless of how
            # many action_ids it occupies. Applied AFTER Cand 10's swap so
            # the rebalance operates on the final target.
            if class_balanced_policy:
                policy_t = class_balanced_target(policy_t, legal)
            # Distillation: sharpen the (final) target toward the teacher's
            # decision. Applied last so it sharpens whatever target the
            # earlier transforms produced (all off in distill runs anyway).
            if policy_sharpen != 1.0:
                policy_t = sharpen_policy_target(policy_t, policy_sharpen)
            lv = F.mse_loss(v_pred, value_t)
            lp = _masked_policy_loss(p_logits, policy_t, legal)
            loss = w_value * lv + w_policy * lp
            # Cand 8: action-class VP prior KL term. Off by default
            # (lambda_vp=0). Enabled per Phase 1 cell config.
            if lambda_vp > 0.0:
                from .action_classes import build_vp_prior_target
                vp_target = build_vp_prior_target(legal)
                lvp = _vp_prior_loss(p_logits, vp_target, legal)
                loss = loss + lambda_vp * lvp
            # Cand 1: pure-pip settlement-vertex prior. Off by default
            # (lambda_settle=0). Fires whenever any settlement action
            # is legal in a sample. Per chat 2026-05-12: pure pip
            # (no resource VP weighting); robber ignored (Option A);
            # no phase gating.
            if lambda_settle > 0.0:
                from .settlement_vertex_prior import settlement_prior_loss
                # Reshape PyG-concat hex features [B*19, 8] -> [B, 19, 8].
                hex_features = batch["hex"].x.view(-1, 19, 8)
                lsettle = settlement_prior_loss(p_logits, legal, hex_features)
                loss = loss + lambda_settle * lsettle
            # Cand 11: pure-pip road prior. Off by default (lambda_road=0).
            # Gate A fires when no legal settlement exists. Layer 1 KL is
            # over the road slice only (no global logit inflation).
            if lambda_road > 0.0:
                from .road_pip_prior import road_pip_prior_loss
                # Reshape PyG-concat features [B*N, F] -> [B, N, F].
                hex_feat_b = batch["hex"].x.view(-1, 19, 8)
                vert_feat_b = batch["vertex"].x.view(-1, 54, 13)
                edge_feat_b = batch["edge"].x.view(-1, 72, 6)
                lroad = road_pip_prior_loss(
                    p_logits=p_logits,
                    legal_mask=legal,
                    edge_features=edge_feat_b,
                    vertex_features=vert_feat_b,
                    hex_features=hex_feat_b,
                )
                loss = loss + lambda_road * lroad
            loss.backward()
            opt.step()
            tot += loss.item(); tv += lv.item(); tp += lp.item(); n += 1
            # Per-batch progress (cadence: ~50 lines/epoch).
            if n % progress_every == 0 or n == batches_total:
                now = _time.perf_counter()
                elapsed = now - t_train_start
                ms_per_batch = elapsed / n * 1000
                pct = 100 * n / batches_total
                eta_s = (batches_total - n) * elapsed / n
                print(
                    f"[ep{epoch:>2} batch {n:>5}/{batches_total} "
                    f"({pct:5.1f}%)] "
                    f"loss={loss.item():.3f} "
                    f"{ms_per_batch:6.1f} ms/batch "
                    f"eta {eta_s/60:5.1f}min",
                    flush=True,
                )
                last_progress_time = now
                # Streaming per-batch metrics for the live dashboard loss curve
                # (feedback_training_observability: per-batch, not per-epoch).
                # grad-norm is read before the next zero_grad clears grads.
                try:
                    import json as _json
                    gnorm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            gnorm += float(p.grad.detach().norm().item()) ** 2
                    gnorm = gnorm ** 0.5
                    tp_path = out_dir / "train_progress.jsonl"
                    with tp_path.open("a", encoding="utf-8") as _f:
                        _f.write(_json.dumps({
                            "epoch": epoch, "batch": n, "batches_total": batches_total,
                            "loss": float(loss.item()), "loss_value": float(lv.item()),
                            "loss_policy": float(lp.item()), "grad_norm": gnorm,
                            "ms_per_batch": ms_per_batch, "ts": now,
                        }) + "\n")
                except Exception:
                    pass
                # Mid-epoch dashboard update so the dashboard isn't silent
                # for hours between epoch boundaries.
                if status_file is not None:
                    try:
                        _write_training_status(
                            status_file=Path(status_file),
                            label=status_label or out_dir.name,
                            epoch=epoch,
                            epochs_total=epochs,
                            train_loss=tot / max(n, 1),
                            val_loss=0.0,
                            val_top1=0.0,
                            per_game=(0.0, 0.0, 0.0, 0.0, 0.0),
                            best_top1=best_top1,
                            best_top1_epoch=best_top1_epoch,
                            epochs_since_best=epochs_since_best,
                            early_stop_patience=early_stop_patience,
                            state=f"training_batch_{n}_of_{batches_total}",
                            train_secs=elapsed,
                            val_secs=0.0,
                        )
                    except Exception:
                        pass
        train_secs = _time.perf_counter() - t_train_start

        # Val.
        model.eval()
        vt, vv, vp_, vmae, top1, n_pos = 0.0, 0.0, 0.0, 0.0, 0, 0
        # Value-head metrics (Task 7, observability minimum — the value head,
        # the project's documented root-cause villain for miscalibration, had
        # NO validation metric before this; only the policy head (val_top1)
        # was measured). val_value_mse mirrors val_value_mae's per-batch-mean
        # accumulation (consistent with the existing MAE convention here,
        # even though it under-weights the last partial batch slightly).
        # val_value_sign_acc is computed over NONZERO targets only: value_t is
        # all-zero for winner==-1 (no-decisive-winner) games (dataset.py:159),
        # and sign(0) vs sign(pred) is not a meaningful calibration question —
        # counting those positions would dilute the metric with undefined-target
        # noise instead of measuring "does the value head pick the right side".
        vmse = 0.0
        sign_hits = 0
        sign_total = 0
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
                # Cand 7: rebalance for the LOSS only. top1 uses the
                # ORIGINAL teacher target so cross-cell val_top1 numbers
                # remain comparable (val_top1 is "argmax of teacher" — that
                # answer shouldn't depend on the loss reweighting).
                policy_t_for_loss = (
                    class_balanced_target(policy_t, legal)
                    if class_balanced_policy
                    else policy_t
                )
                # Distillation: keep val loss on the same target the train
                # loss optimizes. top1 below stays on the ORIGINAL teacher
                # target (sharpening preserves argmax, so it's identical
                # either way — but original keeps cross-run comparability).
                if policy_sharpen != 1.0:
                    policy_t_for_loss = sharpen_policy_target(
                        policy_t_for_loss, policy_sharpen)
                lv = F.mse_loss(v_pred, value_t)
                lp = _masked_policy_loss(p_logits, policy_t_for_loss, legal)
                vt += float(w_value * lv + w_policy * lp); vv += float(lv); vp_ += float(lp)
                vmae += float((v_pred - value_t).abs().mean())
                vmse += float(((v_pred - value_t) ** 2).mean())
                nonzero_mask = value_t != 0.0
                if nonzero_mask.any():
                    sign_hits += int((torch.sign(v_pred[nonzero_mask])
                                      == torch.sign(value_t[nonzero_mask])).sum())
                    sign_total += int(nonzero_mask.sum())
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
            val_value_mse=vmse / max(1, len(val_loader)),
            val_value_sign_acc=sign_hits / max(1, sign_total),
            val_policy_top1_acc=top1 / max(1, n_pos),
            val_top1_per_game_min=pg_min,
            val_top1_per_game_p25=pg_p25,
            val_top1_per_game_median=pg_p50,
            val_top1_per_game_p75=pg_p75,
            val_top1_per_game_max=pg_max,
            val_n_games=n_games,
        )
        log["epochs"].append(asdict(stats))
        vp_swap_str = ""
        if vp_compare_rule and vp_swap_samples > 0:
            vp_swap_rate = vp_swap_total / vp_swap_samples
            vp_swap_str = f" vp_swap={vp_swap_total}/{vp_swap_samples} ({100*vp_swap_rate:.2f}%)"
        print(f"[epoch {epoch}/{epochs}] train_loss={stats.train_loss_total:.3f} "
              f"val_loss={stats.val_loss_total:.3f} val_top1={stats.val_policy_top1_acc:.3f} "
              f"val_value_mse={stats.val_value_mse:.4f} "
              f"val_value_sign_acc={stats.val_value_sign_acc:.3f} "
              f"per_game[{pg_min:.2f}|{pg_p25:.2f}|{pg_p50:.2f}|{pg_p75:.2f}|{pg_max:.2f}]"
              f"{vp_swap_str} "
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
        # We track two notions of "best":
        #   - best_top1 (strict): saved checkpoint, used by downstream code.
        #   - For early-stop "improvement" we require a small margin (0.005)
        #     over the best so noise-spikes from a single early epoch don't
        #     wedge the patience counter. Without the margin we observed
        #     val_top1 oscillating ±0.01 between epochs (true noise) and
        #     patience=3 firing too early — see `feedback_early_stop_margin`.
        EARLY_STOP_MARGIN = 0.005
        if stats.val_policy_top1_acc > best_top1:
            best_top1 = stats.val_policy_top1_acc
            best_top1_epoch = epoch
            torch.save(cpu_state, out_dir / "checkpoint_best.pt")
            print(f"  ↳ new best_top1={best_top1:.3f} at epoch {epoch} (saved checkpoint_best.pt)",
                  flush=True)
        # Early-stop counter: increment unless this epoch's val_top1 is
        # within EARLY_STOP_MARGIN of the running best. This treats epochs
        # that nearly match the high as "still trending toward improvement"
        # rather than as a regression.
        if stats.val_policy_top1_acc >= best_top1 - EARLY_STOP_MARGIN:
            epochs_since_best = 0
        else:
            epochs_since_best += 1
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

        # Live dashboard status write — atomic via tmp + rename.
        if status_file is not None:
            try:
                _write_training_status(
                    status_file=status_file,
                    label=status_label,
                    epoch=epoch,
                    epochs_total=epochs,
                    train_loss=stats.train_loss_total,
                    val_loss=stats.val_loss_total,
                    val_top1=stats.val_policy_top1_acc,
                    per_game=(pg_min, pg_p25, pg_p50, pg_p75, pg_max),
                    best_top1=best_top1,
                    best_top1_epoch=best_top1_epoch,
                    epochs_since_best=epochs_since_best,
                    early_stop_patience=early_stop_patience,
                    state="training",
                    train_secs=train_secs,
                    val_secs=val_secs,
                )
            except Exception as e:
                print(f"  (status_file write failed: {e})", flush=True)

        # === Mid-training tournament hook (loss-augmentation roadmap) ===
        # Runs after the per-epoch checkpoint save; uses the just-saved
        # checkpoint_best.pt (= the running highest val_top1 so far). On a
        # >= drop_threshold-game drop in PureGnn winrate vs the previous
        # mid-training tournament, request stop.
        mid_tournament_stop = False
        if mid_tournament_every > 0 and (epoch % mid_tournament_every == 0):
            from .mid_training_tournament import (
                run_mid_training_tournament, should_stop_for_drop,
            )
            mt_out_root = out_dir / "mid_tournaments"
            mt_out_root.mkdir(parents=True, exist_ok=True)
            try:
                # Mid-tournament tests the CURRENT epoch's weights — not the
                # "best by val_top1" weights. val_top1 stops tracking
                # tournament strength once we're in the overfitting regime
                # (cited v3.6: winrate doubled with flat val_top1). The
                # whole point of the mid-tournament is to measure the
                # current model's real strength, so we point at the
                # current epoch's bundle (model_state is unwrapped by
                # _load_model in e10_v3_tournament).
                epoch_ckpt_path = out_dir / f"checkpoint_epoch{epoch:02d}.pt"
                result = run_mid_training_tournament(
                    epoch=epoch,
                    checkpoint_path=epoch_ckpt_path,
                    out_root=mt_out_root,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    num_games_per_seating=mid_tournament_games_per_seating,
                    sims=mid_tournament_sims,
                    lookahead_depth=mid_tournament_lookahead_depth,
                    base_sims_v3=mid_tournament_base_sims_v3,
                    seed_base=mid_tournament_seed_base,
                    max_seconds=mid_tournament_max_seconds,
                    device=mid_tournament_device,
                    workers=mid_tournament_workers,
                )
                s = result.summary
                print(f"  ↳ mid-tournament @ epoch {epoch}: "
                      f"PureGnn={s['pure_gnn_wins']}/{s['total_games']} "
                      f"({100 * s['pure_gnn_winrate']:.1f}%); "
                      f"GnnMcts={s['gnn_mcts_wins']}, "
                      f"LookaheadV3={s['lookahead_v3_wins']}, "
                      f"Random={s['random_wins']}, "
                      f"draws={s['no_winner_games']}; "
                      f"{result.elapsed_seconds:.0f}s",
                      flush=True)
                log["mid_tournaments"].append({
                    "epoch": epoch,
                    "pure_gnn_wins": s["pure_gnn_wins"],
                    "pure_gnn_winrate": s["pure_gnn_winrate"],
                    "gnn_mcts_wins": s["gnn_mcts_wins"],
                    "lookahead_v3_wins": s["lookahead_v3_wins"],
                    "random_wins": s["random_wins"],
                    "no_winner_games": s["no_winner_games"],
                    "total_games": s["total_games"],
                    "run_dir": str(result.run_dir),
                    "elapsed_seconds": result.elapsed_seconds,
                })
                (out_dir / "training_log.json").write_text(json.dumps(log, indent=2))
                if should_stop_for_drop(
                    history=mid_tournament_history,
                    current_wins=s["pure_gnn_wins"],
                    total_games=s["total_games"],
                    drop_threshold=mid_tournament_drop_threshold,
                ):
                    prev = mid_tournament_history[-1] if mid_tournament_history else 0
                    print(f"  ↳ mid-tournament early-stop: "
                          f"PureGnn wins dropped {prev} → {s['pure_gnn_wins']} "
                          f"(>= {mid_tournament_drop_threshold}-game threshold); "
                          f"stopping at epoch {epoch}",
                          flush=True)
                    mid_tournament_stop = True
                mid_tournament_history.append(s["pure_gnn_wins"])
            except Exception as e:
                # Tournament infra failure should not kill training — log and
                # continue. The next mid-tournament will retry.
                print(f"  ↳ mid-tournament @ epoch {epoch} FAILED: {e}", flush=True)
                import traceback
                traceback.print_exc()

        if mid_tournament_stop:
            break

        # PAUSE (pausability): a PAUSE sentinel in pause_dir (or a parent) stops
        # cleanly at the epoch boundary. The per-epoch checkpoint.pt (next_epoch=
        # epoch+1) was JUST written above, so resume_from=checkpoint.pt continues
        # from the next epoch — reproducible (epoch-granular; an epoch is minutes).
        if pause_dir is not None:
            _pd = Path(pause_dir)
            if ((_pd / "PAUSE").exists() or (_pd.parent / "PAUSE").exists()
                    or (_pd.parent.parent / "PAUSE").exists()):
                print(f"  ↳ PAUSED at epoch {epoch}; checkpoint.pt holds "
                      f"next_epoch={epoch + 1}. Resume with resume_from=checkpoint.pt",
                      flush=True)
                (out_dir / "PAUSED").write_text(str(epoch + 1))
                break

        # Early-stop check: if patience set and we've not improved for N epochs, stop.
        if early_stop_patience > 0 and epochs_since_best >= early_stop_patience:
            print(f"  ↳ early-stop: val_top1 not improved for {epochs_since_best} epochs "
                  f"(best={best_top1:.3f} at epoch {best_top1_epoch}); stopping at epoch {epoch}",
                  flush=True)
            if status_file is not None:
                try:
                    _write_training_status(
                        status_file=status_file,
                        label=status_label,
                        epoch=epoch,
                        epochs_total=epochs,
                        train_loss=stats.train_loss_total,
                        val_loss=stats.val_loss_total,
                        val_top1=stats.val_policy_top1_acc,
                        per_game=(pg_min, pg_p25, pg_p50, pg_p75, pg_max),
                        best_top1=best_top1,
                        best_top1_epoch=best_top1_epoch,
                        epochs_since_best=epochs_since_best,
                        early_stop_patience=early_stop_patience,
                        state="early_stopped",
                        train_secs=train_secs,
                        val_secs=val_secs,
                    )
                except Exception:
                    pass
            break

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
    p.add_argument("--policy-sharpen", type=float, default=1.0,
                   help="Distillation: raise visit-count policy targets to this "
                        "power and renormalize (2.0 = visits^2, the recommended "
                        "teacher-decision target). 1.0 = off.")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Stop training when val_top1 hasn't beaten best for N epochs in a row. "
                        "0 disables early stopping (default; preserves prior --epochs behavior).")
    p.add_argument("--status-file", type=Path, default=None,
                   help="If set, write a per-epoch dashboard JSON entry to this file. Used by "
                        "the grid-experiment orchestrator (grid_dashboard.py).")
    p.add_argument("--status-label", type=str, default="",
                   help="Cell label to use when writing status (e.g. 'h64_l3'). Required with --status-file.")
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
        policy_sharpen=args.policy_sharpen,
        early_stop_patience=args.early_stop_patience,
        status_file=args.status_file,
        status_label=args.status_label,
    )


if __name__ == "__main__":
    cli_main()
