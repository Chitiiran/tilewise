"""GnnModel: heterogeneous PyG body + value head + policy head.

Spec §2-3 architecture:
  per-type linear input projection (hex 8->32, vertex 7->32, edge 6->32)
  -> N x HeteroConv(SAGEConv per edge type, hidden_dim)
  -> per-type mean-pool over nodes (3 x [B, hidden_dim])
  -> concat with scalars (22) -> final linear -> embedding [128]
  -> ValueHead(128 -> 64 -> 4, tanh)
  -> PolicyHead(128 -> 206) -- caller masks illegals + softmaxes

The policy head outputs raw logits over the full 206 action space. Masking
and softmax happen outside this module (in GnnEvaluator and the dataset
loss). This keeps the model itself loss-agnostic and easy to test.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import scatter

from catan_mcts import ACTION_SPACE_SIZE


F_HEX, F_VERT, F_EDGE = 8, 7, 6
N_SCALARS = 22
EMBED_DIM = 128


class GnnBody(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.proj_hex = nn.Linear(F_HEX, hidden_dim)
        self.proj_vertex = nn.Linear(F_VERT, hidden_dim)
        self.proj_edge = nn.Linear(F_EDGE, hidden_dim)
        # Heterogeneous message passing: one SAGEConv per edge type.
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HeteroConv({
                ("hex", "to", "vertex"): SAGEConv(hidden_dim, hidden_dim),
                ("vertex", "to", "hex"): SAGEConv(hidden_dim, hidden_dim),
                ("vertex", "to", "edge"): SAGEConv(hidden_dim, hidden_dim),
                ("edge", "to", "vertex"): SAGEConv(hidden_dim, hidden_dim),
            }, aggr="mean"))
        # Final projection: pooled hex (hidden) + pooled vertex (hidden) +
        # pooled edge (hidden) + scalars (22) -> EMBED_DIM.
        self.final = nn.Linear(3 * hidden_dim + N_SCALARS, EMBED_DIM)

    def forward(self, batch: HeteroData) -> torch.Tensor:
        x_dict = {
            "hex": F.relu(self.proj_hex(batch["hex"].x)),
            "vertex": F.relu(self.proj_vertex(batch["vertex"].x)),
            "edge": F.relu(self.proj_edge(batch["edge"].x)),
        }
        for conv in self.convs:
            x_dict = conv(x_dict, batch.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        # Per-type mean pool, batched.
        # PyG's Batch concatenates node lists; batch[k].batch maps each node to
        # its graph index in the batch.
        pooled = []
        for k in ("hex", "vertex", "edge"):
            idx = batch[k].batch
            pooled.append(scatter(x_dict[k], idx, dim=0, reduce="mean"))
        # state_to_pyg now stores data.scalars as [1, 22], so collation produces
        # [B, 22] deterministically. Reshape kept as no-op safety against
        # PyG-version layout drift.
        scalars = batch.scalars.view(-1, N_SCALARS)
        emb = torch.cat([*pooled, scalars], dim=1)
        return self.final(emb)


class ValueHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Tanh(),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


class PolicyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(EMBED_DIM, ACTION_SPACE_SIZE)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)  # raw logits


class GnnModel(nn.Module):
    def __init__(self, hidden_dim: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        self.body = GnnBody(hidden_dim=hidden_dim, num_layers=num_layers)
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()

    def forward(self, batch: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.body(batch)
        return self.value_head(emb), self.policy_head(emb)
