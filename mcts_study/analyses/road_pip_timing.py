"""End-to-end per-batch timing for Cand 11.

Measures, at production batch size B=256, three regimes on a real GNN
forward+backward step:
  - vanilla (lambda_road = 0)
  - cand 11 (lambda_road = 0.05) using current loop impl
  - reports the ratio

Uses the e1 fixture (tiny replay dataset) as the data source so the
cache load is fast (~10s). Builds a real PyG batch via the standard
DataLoader path so the timing is representative of training-step cost.

Usage:
    cd mcts_study
    python scratch_road_pip_timing.py
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from catan_gnn.dataset import CatanReplayDataset
from catan_gnn.gnn_model import GnnModel
from catan_gnn.train import _collate, _masked_policy_loss
from catan_gnn.road_pip_prior import (
    ROAD_ACTION_OFFSET,
    NUM_EDGES,
    road_pip_prior_loss,
)


def generate_e1_fixture(out_root: Path) -> Path:
    from catan_mcts.experiments.e1_winrate_vs_random import main as e1_main
    return e1_main(
        out_root=out_root,
        num_games=20, sims_per_move_grid=[2],
        seed_base=12345, max_seconds=300.0,
    )


def time_train_step(*, lambda_road: float, n_warmup=3, n_measure=10,
                     batch_size: int = 256, e1_dir: Path,
                     device: str = "cuda"):
    """Time forward+road_loss+backward for n_measure iterations.

    Returns mean ms/batch over the measured iterations (warmup excluded).
    """
    ds = CatanReplayDataset([e1_dir])
    if len(ds) < batch_size:
        # Repeat samples to reach batch_size — we want to time at production
        # batch size, even on a small fixture.
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
    # Pre-load batches so dataset iteration doesn't pollute timing.
    for _ in range(n_warmup + n_measure):
        try:
            batches.append(next(it))
        except StopIteration:
            # Cycle the iterator.
            it = iter(loader)
            batches.append(next(it))

    def one_step(b):
        batch, value_t, policy_t, legal = b
        batch = batch.to(dev)
        value_t = value_t.to(dev)
        policy_t = policy_t.to(dev)
        legal = legal.to(dev)
        opt.zero_grad()
        v_pred, p_logits = model(batch)
        lv = F.mse_loss(v_pred, value_t)
        lp = _masked_policy_loss(p_logits, policy_t, legal.bool())
        loss = lv + lp
        if lambda_road > 0:
            hex_feat_b = batch["hex"].x.view(-1, 19, 8)
            vert_feat_b = batch["vertex"].x.view(-1, 54, 13)
            edge_feat_b = batch["edge"].x.view(-1, 72, 6)
            lroad = road_pip_prior_loss(
                p_logits=p_logits,
                legal_mask=legal.bool(),
                edge_features=edge_feat_b,
                vertex_features=vert_feat_b,
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
    print("=== Cand 11 end-to-end timing (with real GNN + DataLoader) ===\n")

    e1_root = Path("/tmp/cand11_timing_e1_root")
    if not (e1_root / "e1_winrate_vs_random").exists():
        e1_dir = generate_e1_fixture(e1_root)
    else:
        # Pick whatever was generated.
        e1_dir = list((e1_root).glob("*e1*"))[0]
    print(f"e1 fixture: {e1_dir}\n")

    # Vanilla
    print("Running vanilla (lambda_road = 0)...")
    ms_vanilla = time_train_step(lambda_road=0.0, e1_dir=e1_dir)
    print(f"  vanilla:  {ms_vanilla:.1f} ms/batch")

    # Cand 11 enabled
    print("Running Cand 11 (lambda_road = 0.05)...")
    ms_cand11 = time_train_step(lambda_road=0.05, e1_dir=e1_dir)
    print(f"  Cand 11:  {ms_cand11:.1f} ms/batch")

    ratio = ms_cand11 / ms_vanilla
    overhead = ms_cand11 - ms_vanilla
    print(f"\n=== Results ===")
    print(f"  vanilla:  {ms_vanilla:7.1f} ms/batch")
    print(f"  Cand 11:  {ms_cand11:7.1f} ms/batch")
    print(f"  overhead: {overhead:7.1f} ms/batch ({100*(ratio-1):+.0f}%)")
    print(f"  ratio:    {ratio:7.2f}x slower")

    # Project to a full 100k-cache epoch (3.22M samples / B=256 = ~12,627 batches)
    batches_per_epoch = 12627
    vanilla_epoch_min = ms_vanilla * batches_per_epoch / 1000 / 60
    cand11_epoch_min = ms_cand11 * batches_per_epoch / 1000 / 60
    print(f"\n=== Projection to 100k-cache epoch (~12,627 batches @ B=256) ===")
    print(f"  vanilla:  {vanilla_epoch_min:6.1f} min/epoch")
    print(f"  Cand 11:  {cand11_epoch_min:6.1f} min/epoch")
    print(f"  Cell 1 reference (vanilla, measured): ~63 min/epoch")


if __name__ == "__main__":
    main()
