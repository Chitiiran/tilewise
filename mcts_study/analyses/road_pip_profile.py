"""cProfile-based root-cause analysis for Cand 11 perf regression.

Builds a realistic batch (B=256) from a single sample replicated, then runs
road_pip_prior_loss N times under cProfile. Prints top-30 functions by
cumulative time. This is evidence, not speculation.

We also compare with a vanilla forward+backward step (no road loss) at the
same batch size for ratio.

Usage:
    cd mcts_study
    python scratch_road_pip_profile.py
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time

import torch

from catan_gnn.road_pip_prior import (
    ROAD_ACTION_OFFSET,
    NUM_EDGES,
    road_pip_prior_loss,
)


B = 256


def build_batch(b: int = B):
    """Build a synthetic batch matching the training-time tensor shapes."""
    torch.manual_seed(0)
    p_logits = torch.randn(b, 280, requires_grad=True)

    # Build a legal mask that mostly has NO settlement legal (so Gate A part 1
    # fires) and has 5-10 legal roads (typical mid-game).
    legal = torch.zeros(b, 280, dtype=torch.bool)
    legal[:, 204] = True  # EndTurn always legal
    legal[:, 205] = True  # RollDice
    # 7 legal roads per sample, distributed.
    for i in range(b):
        for k in range(7):
            edge_id = (i * 7 + k) % NUM_EDGES
            legal[i, ROAD_ACTION_OFFSET + edge_id] = True

    # No legal settlements (Gate A part 1 fires).
    # Vertex/edge features matching observation.rs layout.
    vertex_features = torch.zeros(b, 54, 13)
    vertex_features[:, :, 0] = 1.0  # all empty

    edge_features = torch.zeros(b, 72, 6)
    edge_features[:, :, 0] = 1.0  # all empty
    # Give viewer 3 roads per sample so frontier is non-trivial.
    for i in range(b):
        for k in range(3):
            e = (i * 3 + k) % NUM_EDGES
            edge_features[i, e, 0] = 0.0
            edge_features[i, e, 1] = 1.0  # has road
            edge_features[i, e, 2] = 1.0  # viewer owns

    hex_features = torch.zeros(b, 19, 8)
    hex_features[:, :, 0] = 1.0  # wood resource
    # Set dice number = 6 (max pip): hex_features[h, 5] = (6 - 7) / 5 = -0.2
    hex_features[:, :, 5] = -0.2

    return p_logits, legal, edge_features, vertex_features, hex_features


def run_road_loss(n_iters: int):
    """Run road_pip_prior_loss n_iters times. Profile target."""
    p_logits, legal, ef, vf, hf = build_batch()
    for _ in range(n_iters):
        loss = road_pip_prior_loss(
            p_logits=p_logits, legal_mask=legal,
            edge_features=ef, vertex_features=vf, hex_features=hf,
        )
        # Force the loss to materialize (cProfile would skip lazy work otherwise).
        _ = float(loss.item())


def measure_wall_clock():
    """Plain wall-clock for N iterations, no profiler overhead."""
    p_logits, legal, ef, vf, hf = build_batch()
    n_warmup = 3
    n_measure = 10

    # Warmup
    for _ in range(n_warmup):
        loss = road_pip_prior_loss(
            p_logits=p_logits, legal_mask=legal,
            edge_features=ef, vertex_features=vf, hex_features=hf,
        )
        _ = float(loss.item())

    t0 = time.perf_counter()
    for _ in range(n_measure):
        loss = road_pip_prior_loss(
            p_logits=p_logits, legal_mask=legal,
            edge_features=ef, vertex_features=vf, hex_features=hf,
        )
        _ = float(loss.item())
    elapsed = time.perf_counter() - t0
    per_call_ms = elapsed / n_measure * 1000
    print(f"\n=== wall-clock: road_pip_prior_loss only (B={B}) ===")
    print(f"  {n_measure} iters in {elapsed:.3f}s -> {per_call_ms:.1f} ms/call")
    return per_call_ms


def main():
    print(f"=== Cand 11 profile, batch size B={B} ===\n")

    # 1) Wall-clock measurement.
    ms_per_call = measure_wall_clock()

    # 2) cProfile breakdown.
    print(f"\n=== cProfile of road_pip_prior_loss (10 iters at B={B}) ===")
    pr = cProfile.Profile()
    pr.enable()
    run_road_loss(10)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())

    # 3) Per-batch-component cost estimate.
    # If road_pip_prior_loss is X ms and vanilla per-batch is Y ms,
    # the overhead ratio matters more than absolute X.
    print(f"\n=== Interpretation ===")
    print(f"road_pip_prior_loss alone: {ms_per_call:.1f} ms/call at B={B}")
    print(f"For 12,627 batches/epoch (100k cache / B=256), that's")
    print(f"  {ms_per_call * 12627 / 1000:.0f}s = {ms_per_call * 12627 / 60000:.1f} min/epoch")
    print(f"  added ON TOP of whatever the model forward+backward costs.")


if __name__ == "__main__":
    main()
