"""Pre-launch diagnostic for Cand 11 (road-pip prior).

Walks 1000 random samples from the 100k cache and reports:
  - Fraction of samples where Gate A's first condition fires (no legal settle).
  - Of those, fraction where any legal road has nonzero score (gate fully fires).
  - Distribution of |L_R| (number of legal roads) on firing samples.
  - Mean entropy of prior vs mean entropy of MCTS visits restricted to roads.

If prior entropy is much lower than visits entropy, the prior is sharper and
lambda_road=0.05 may be too aggressive. If they're comparable, the chosen
lambda is reasonable.

Usage:
    cd mcts_study
    python scratch_road_pip_calibration.py --cache-path ~/catan_cache/cache_100k.pt
"""
import argparse
import math
from pathlib import Path

import torch

from catan_gnn.dataset import CachedDataset
from catan_gnn.road_pip_prior import (
    ROAD_ACTION_OFFSET,
    compute_road_scores,
    NUM_EDGES,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-path", type=Path, required=True)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = CachedDataset(source=None, cache_path=args.cache_path)
    print(f"cache loaded: {len(ds)} positions")
    rng = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=rng)[: args.n_samples].tolist()

    n_no_settle = 0
    n_road_nonzero = 0
    n_road_zero_after_no_settle = 0
    road_count_hist: dict[int, int] = {}
    prior_entropies: list[float] = []
    visit_entropies: list[float] = []

    for i, k in enumerate(idx):
        data, value_t, policy_t, legal = ds[k]
        legal = legal.bool()
        # Reshape per-sample [N, F] (no batch dim from __getitem__).
        hex_f = data["hex"].x        # [19, 8]
        vert_f = data["vertex"].x    # [54, 13]
        edge_f = data["edge"].x      # [72, 6]

        legal_settle_any = legal[0:54].any().item()
        legal_road_mask = legal[ROAD_ACTION_OFFSET:ROAD_ACTION_OFFSET + NUM_EDGES]
        n_legal_roads = int(legal_road_mask.sum().item())

        if legal_settle_any:
            continue
        n_no_settle += 1
        road_count_hist[n_legal_roads] = road_count_hist.get(n_legal_roads, 0) + 1

        scores = compute_road_scores(
            edge_features=edge_f, vertex_features=vert_f,
            hex_features=hex_f, legal_road_mask=legal_road_mask,
        )
        total = float(scores.sum().item())
        if total <= 0:
            n_road_zero_after_no_settle += 1
            continue
        n_road_nonzero += 1

        prior = (scores / total).clamp(min=1e-12)
        prior_nonzero = prior[prior > 0]
        H_prior = -(prior_nonzero * prior_nonzero.log()).sum().item()
        prior_entropies.append(H_prior)

        # MCTS visit entropy restricted to legal roads.
        road_visits = policy_t[ROAD_ACTION_OFFSET:ROAD_ACTION_OFFSET + NUM_EDGES]
        road_visits = road_visits * legal_road_mask.to(road_visits.dtype)
        s = float(road_visits.sum().item())
        if s > 0:
            rv = (road_visits / s).clamp(min=1e-12)
            rv_nz = rv[rv > 0]
            H_visits = -(rv_nz * rv_nz.log()).sum().item()
            visit_entropies.append(H_visits)

    N = args.n_samples
    print()
    print(f"=== Cand 11 calibration on {N} random cache samples ===")
    print(f"Samples with NO legal settlement (Gate A part 1): {n_no_settle} ({100*n_no_settle/N:.1f}%)")
    print(f"  Of those, with at least one nonzero road score (Gate A fully fires): "
          f"{n_road_nonzero} ({100*n_road_nonzero/max(n_no_settle,1):.1f}%)")
    print(f"  All-zero road scores (gate part 3 blocks): "
          f"{n_road_zero_after_no_settle} ({100*n_road_zero_after_no_settle/max(n_no_settle,1):.1f}%)")
    overall_fire_rate = 100 * n_road_nonzero / N
    print(f"  OVERALL gate-firing rate: {n_road_nonzero}/{N} = {overall_fire_rate:.1f}%")
    print()
    print(f"|L_R| histogram on Gate-A-part-1 samples:")
    for k in sorted(road_count_hist):
        print(f"  |L_R| = {k:2d}: {road_count_hist[k]:4d} samples")
    print()
    if prior_entropies:
        mean_H_prior = sum(prior_entropies) / len(prior_entropies)
        print(f"Mean prior entropy (firing samples): {mean_H_prior:.3f}")
    if visit_entropies:
        mean_H_visits = sum(visit_entropies) / len(visit_entropies)
        print(f"Mean MCTS-visits entropy over legal roads (firing samples): {mean_H_visits:.3f}")
        if prior_entropies:
            print(f"Ratio prior/visits: {mean_H_prior/max(mean_H_visits,1e-6):.3f}")
            print(f"  (1.0 = comparable sharpness; <0.5 = prior much sharper, "
                  f"consider lower lambda_road)")
    print()
    print("If overall gate-firing rate < 5%, the loss term is rarely active;")
    print("the experiment may not produce a measurable signal.")
    print("If > 60%, the prior dominates a large fraction of samples;")
    print("consider lower lambda_road (e.g. 0.025).")


if __name__ == "__main__":
    main()
