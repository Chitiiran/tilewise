"""GNN-only player: argmax of policy head, no search.

Lightweight bot that consumes a trained GnnModel and picks the legal
action with the highest policy logit. Useful as a baseline:
  - PureGNN(v1) vs Random tells us if the trained policy beats random
    (without any MCTS to fix mistakes).
  - PureGNN(v1) vs MCTS+GNN(v1) tells us if MCTS adds value at sims=N
    given the same evaluator.

Cost: ~20ms per move (one forward pass), regardless of game depth or
position complexity. Much cheaper than MCTS, useful for tournament
games where slow players gate progress.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg


class PureGnnBot:
    """Plays the legal action that the GNN policy head ranks highest.

    Ties broken deterministically (lowest action ID wins). No exploration,
    no temperature, no search. Pure greedy on the model's prior.
    """

    def __init__(self, model: GnnModel, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def step(self, state) -> int:
        legal = state.legal_actions()
        if not legal:
            raise RuntimeError("PureGnnBot: no legal actions in non-terminal state")
        if len(legal) == 1:
            return int(legal[0])
        obs = state._engine.observation()
        data = state_to_pyg(obs).to(self.device)
        batch = Batch.from_data_list([data])
        self.model.eval()
        _, logits = self.model(batch)
        logits_np = logits.squeeze(0).cpu().numpy()
        # Argmax over legal actions only.
        legal_arr = np.asarray(legal, dtype=np.int64)
        idx = int(np.argmax(logits_np[legal_arr]))
        return int(legal_arr[idx])
