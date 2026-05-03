"""Print torchinfo summaries for the 9 grid architectures.

Usage:
  python scripts/torchinfo_summary.py [h32_l2 ...]   # specific cells, or all
"""
from __future__ import annotations
import sys, io
from contextlib import redirect_stdout

import torch
from torch_geometric.data import Batch
from torchinfo import summary

from catan_bot import _engine
from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg


CELLS = [
    (32, 2), (32, 3), (32, 4),
    (64, 2), (64, 3), (64, 4),
    (128, 2), (128, 3), (128, 4),
]


def make_batch():
    e = _engine.Engine(42)
    obs = e.observation()
    data = state_to_pyg(obs)
    return Batch.from_data_list([data])


def summarize(hidden_dim: int, num_layers: int):
    m = GnnModel(hidden_dim=hidden_dim, num_layers=num_layers)
    m.eval()
    batch = make_batch()
    label = f"h{hidden_dim}_l{num_layers}"
    print(f"\n{'='*78}\n=== {label}  (hidden_dim={hidden_dim}, num_layers={num_layers}) ===\n{'='*78}")
    try:
        info = summary(
            m,
            input_data=[batch],
            depth=4,
            col_names=("input_size", "output_size", "num_params"),
            row_settings=("var_names",),
            verbose=0,
        )
        print(info)
    except Exception as ex:
        # torchinfo can choke on PyG-style irregular inputs; fall back to module tree.
        print(f"(torchinfo failed: {ex})")
        print()
        print("Manual layer breakdown:")
        total = 0
        for name, p in m.named_parameters():
            n = p.numel()
            total += n
            print(f"  {name:60s} shape={tuple(p.shape)}  params={n}")
        print(f"  ---")
        print(f"  TOTAL: {total} params")


def main():
    args = sys.argv[1:]
    if not args:
        cells = CELLS
    else:
        wanted = set(args)
        cells = [(h, l) for h, l in CELLS if f"h{h}_l{l}" in wanted]
    for h, l in cells:
        summarize(h, l)


if __name__ == "__main__":
    main()
