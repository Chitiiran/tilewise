"""GPU per-batch timing for Cell 6 (Cand 11 + Cand 8 + Cand 10) vs vanilla
and Cand 11 alone. Adapted from scratch_road_pip_timing.py.

Catches the kind of CUDA-sync regression that bit Cell 5 v1 (40x slower
than vanilla due to .item() calls). Vanilla on GPU is ~60-70 ms/batch
at B=256; Cell 5 v2 production showed +7%. Cell 6 stacks two more loss
terms — we want to verify per-batch stays under ~2x vanilla before
committing 19h.

Usage:
    cd mcts_study
    python scratch_cell6_timing.py
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from catan_gnn.dataset import CatanReplayDataset
from catan_gnn.gnn_model import GnnModel
from catan_gnn.train import _collate, _masked_policy_loss, _vp_prior_loss
from catan_gnn.road_pip_prior import (
    ROAD_ACTION_OFFSET,
    NUM_EDGES,
    road_pip_prior_loss,
)
from catan_gnn.action_classes import build_vp_prior_target
from catan_gnn.vp_compare import vp_compare_swap_target


def generate_e1_fixture(out_root: Path) -> Path:
    from catan_mcts.experiments.e1_winrate_vs_random import main as e1_main
    return e1_main(
        out_root=out_root,
        num_games=20, sims_per_move_grid=[2],
        seed_base=12345, max_seconds=300.0,
    )


def time_train_step(*, lambda_road: float, lambda_vp: float,
                    vp_compare_rule: bool,
                    n_warmup=3, n_measure=10,
                    batch_size: int = 256, e1_dir: Path,
                    device: str = "cuda"):
    """Time forward + losses + backward for n_measure iterations."""
    ds = CatanReplayDataset([e1_dir])
    if len(ds) < batch_size:
        from torch.utils.data import ConcatDataset
        n_repeat = (batch_size + len(ds) - 1) // len(ds) + 1
        ds = ConcatDataset([ds] * n_repeat)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=_collate, drop_last=True)

    dev = torch.device(device)
    model = GnnModel(hidden_dim=128, num_layers=4).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    it = iter(loader)
    batches = []
    for _ in range(n_warmup + n_measure):
        try:
            batches.append(next(it))
        except StopIteration:
            it = iter(loader)
            batches.append(next(it))

    def one_step(b):
        batch, value_t, policy_t, legal = b
        batch = batch.to(dev)
        value_t = value_t.to(dev)
        policy_t = policy_t.to(dev)
        legal = legal.to(dev).bool()
        opt.zero_grad()
        v_pred, p_logits = model(batch)
        # Cand 10: VP-compare target swap (matches train.py)
        if vp_compare_rule:
            with torch.no_grad():
                policy_t, _ = vp_compare_swap_target(p_logits.detach(), policy_t, legal)
        lv = F.mse_loss(v_pred, value_t)
        lp = _masked_policy_loss(p_logits, policy_t, legal)
        loss = lv + lp
        # Cand 8: VP prior KL
        if lambda_vp > 0:
            vp_target = build_vp_prior_target(legal)
            lvp = _vp_prior_loss(p_logits, vp_target, legal)
            loss = loss + lambda_vp * lvp
        # Cand 11: road-pip prior
        if lambda_road > 0:
            hex_feat_b = batch["hex"].x.view(-1, 19, 8)
            vert_feat_b = batch["vertex"].x.view(-1, 54, 13)
            edge_feat_b = batch["edge"].x.view(-1, 72, 6)
            lroad = road_pip_prior_loss(
                p_logits=p_logits, legal_mask=legal,
                edge_features=edge_feat_b, vertex_features=vert_feat_b,
                hex_features=hex_feat_b,
            )
            loss = loss + lambda_road * lroad
        loss.backward()
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return float(loss.item())

    # Warmup
    for i in range(n_warmup):
        one_step(batches[i])

    # Measure
    t0 = time.perf_counter()
    for i in range(n_warmup, n_warmup + n_measure):
        one_step(batches[i])
    elapsed = time.perf_counter() - t0

    return elapsed / n_measure * 1000  # ms/batch


def main():
    print("=== Cell 6 per-batch timing (GPU, B=256, real h128_l4 GNN) ===\n")
    e1_root = Path("/tmp/cell6_timing_e1_root")
    e1_dirs = list(e1_root.glob("**/games.*.parquet"))
    if e1_dirs:
        # Find the run dir (parent of worker dir)
        e1_dir = next(p.parent.parent for p in e1_dirs)
    else:
        e1_dir = generate_e1_fixture(e1_root)
    print(f"e1 fixture: {e1_dir}\n")

    configs = [
        ("vanilla            ", dict(lambda_road=0.0, lambda_vp=0.0, vp_compare_rule=False)),
        ("Cand 11 alone      ", dict(lambda_road=0.05, lambda_vp=0.0, vp_compare_rule=False)),
        ("Cand 8 alone       ", dict(lambda_road=0.0, lambda_vp=0.10, vp_compare_rule=False)),
        ("Cand 8+10          ", dict(lambda_road=0.0, lambda_vp=0.10, vp_compare_rule=True)),
        ("Cell 6 (all stack) ", dict(lambda_road=0.05, lambda_vp=0.10, vp_compare_rule=True)),
    ]

    results = {}
    for label, cfg in configs:
        print(f"Running {label.strip()}...")
        ms = time_train_step(e1_dir=e1_dir, **cfg)
        results[label] = ms
        print(f"  {label.strip()}: {ms:.1f} ms/batch")

    print("\n=== Summary ===")
    ms_vanilla = results["vanilla            "]
    print(f"  {'config':<22s} {'ms/batch':>10s} {'ratio':>8s}")
    for label, ms in results.items():
        ratio = ms / ms_vanilla
        print(f"  {label:<22s} {ms:>9.1f}  {ratio:>7.2f}x")

    print(f"\n=== Projection to 100k-cache epoch (12,627 batches @ B=256) ===")
    for label, ms in results.items():
        epoch_min = ms * 12627 / 1000 / 60
        print(f"  {label:<22s} {epoch_min:>7.1f} min/epoch")

    print(f"\n=== Cell 6 verdict ===")
    cell6_ms = results["Cell 6 (all stack) "]
    cell6_ratio = cell6_ms / ms_vanilla
    if cell6_ratio > 5.0:
        print(f"  REGRESSION: Cell 6 is {cell6_ratio:.1f}x vanilla. KILL the running training.")
    elif cell6_ratio > 2.0:
        print(f"  WARN: Cell 6 is {cell6_ratio:.1f}x vanilla. Tolerable but slow.")
    else:
        print(f"  OK: Cell 6 is {cell6_ratio:.2f}x vanilla. Safe to continue.")


if __name__ == "__main__":
    main()
