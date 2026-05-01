"""Convert engine observation dict to PyG HeteroData.

Pure function. No state. Adjacency tables come from `adjacency.py`.

The observation dict (from `engine.observation()`) carries:
- hex_features:    np.ndarray [19, 8]  float32
- vertex_features: np.ndarray [54, 7]  float32
- edge_features:   np.ndarray [72, 6]  float32
- scalars:         np.ndarray [N_SCALARS] float32
- legal_mask:      np.ndarray [ACTION_SPACE_SIZE] uint8 (engine returns 0/1 bytes)
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .adjacency import HEX_VERTEX_EDGE_INDEX, VERTEX_EDGE_EDGE_INDEX, NUM_HEXES, NUM_EDGES


# Pre-build edge_index tensors once (shared across all calls; they never change).
# These are READ-ONLY at runtime -- do NOT mutate in place. PyG's standard ops
# (Batch.from_data_list, HeteroConv, .to(device)) all return new tensors, so
# sharing is safe. If you ever introduce custom in-place edge_index ops,
# .clone() before mutating.
_H2V_EI = torch.from_numpy(HEX_VERTEX_EDGE_INDEX[:, :NUM_HEXES * 6].copy())   # [2, 114] hex -> vertex
_V2H_EI = torch.from_numpy(HEX_VERTEX_EDGE_INDEX[:, NUM_HEXES * 6:].copy())   # [2, 114] vertex -> hex
_V2E_EI = torch.from_numpy(VERTEX_EDGE_EDGE_INDEX[:, :NUM_EDGES * 2].copy())  # [2, 144] vertex -> edge
_E2V_EI = torch.from_numpy(VERTEX_EDGE_EDGE_INDEX[:, NUM_EDGES * 2:].copy())  # [2, 144] edge -> vertex


def state_to_pyg(obs: dict) -> HeteroData:
    data = HeteroData()
    data["hex"].x = torch.from_numpy(np.ascontiguousarray(obs["hex_features"], dtype=np.float32))
    data["vertex"].x = torch.from_numpy(np.ascontiguousarray(obs["vertex_features"], dtype=np.float32))
    data["edge"].x = torch.from_numpy(np.ascontiguousarray(obs["edge_features"], dtype=np.float32))
    data["hex", "to", "vertex"].edge_index = _H2V_EI
    data["vertex", "to", "hex"].edge_index = _V2H_EI
    data["vertex", "to", "edge"].edge_index = _V2E_EI
    data["edge", "to", "vertex"].edge_index = _E2V_EI
    # Shape [1, N_SCALARS] (not [N_SCALARS]) so PyG's Batch collation deterministically
    # produces [B, N_SCALARS] regardless of PyG version. Otherwise some versions stack
    # to [B, N_SCALARS] and others concat to [B*N_SCALARS] requiring a view().
    data.scalars = torch.from_numpy(np.ascontiguousarray(obs["scalars"], dtype=np.float32)).unsqueeze(0)
    data.legal_mask = torch.from_numpy(np.ascontiguousarray(obs["legal_mask"], dtype=np.uint8)).bool()
    return data
