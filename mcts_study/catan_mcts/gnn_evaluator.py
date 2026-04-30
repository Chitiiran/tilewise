"""OpenSpiel Evaluator backed by a trained GnnModel.

Mirrors the API of catan_mcts.evaluator.LookaheadVpEvaluator. evaluate() and
prior() share a single forward pass via a tiny per-state cache keyed on the
engine's action_history (deterministic state ID). MCTSBot calls them
back-to-back on the same leaf, so the cache hit rate is essentially 100%.
"""
from __future__ import annotations

import numpy as np
import torch
from open_spiel.python.algorithms import mcts as os_mcts
from torch_geometric.data import Batch

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


class GnnEvaluator(os_mcts.Evaluator):
    def __init__(self, model: GnnModel, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device
        # Cache the most recent forward result so MCTSBot's back-to-back
        # evaluate(state) + prior(state) calls share one forward pass.
        # Key by id(state) — Python object identity. MCTSBot keeps the same
        # State object alive for those paired calls, so id() is constant.
        # Using tuple(state._engine.action_history()) was O(N) per call where
        # N grows to 5000+ in deep games; that O(N) hash dominated MCTS+GNN
        # wall-clock at sims=100 (each move was ~6-10 sec; theoretical was
        # ~700ms). id() is O(1).
        self._cache_id: int | None = None
        self._cache_value: np.ndarray | None = None
        self._cache_policy: np.ndarray | None = None

    @torch.no_grad()
    def _forward(self, state):
        # NOTE: each call is one forward pass at batch=1. At sims=400 with ~150
        # MCTS-decided moves per game, that's ~60k sequential forward passes per
        # game. Per-call host->device transfer dominates compute at this size.
        # A BatchedGnnEvaluator that collects leaf evaluations across MCTS sims
        # and runs them as one batched forward pass is the obvious v1 follow-up.
        # See spec §5 "out of scope for v0".
        sid = id(state)
        if sid == self._cache_id:
            return self._cache_value, self._cache_policy
        obs = state._engine.observation()
        data = state_to_pyg(obs).to(self.device)
        batch = Batch.from_data_list([data])
        self.model.eval()
        v, logits = self.model(batch)
        v_np = v.squeeze(0).cpu().numpy().astype(np.float32)
        l_np = logits.squeeze(0).cpu().numpy().astype(np.float32)
        self._cache_id = sid
        self._cache_value = v_np
        self._cache_policy = l_np
        return v_np, l_np

    def evaluate(self, state):
        v, _ = self._forward(state)
        return v

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        _, logits = self._forward(state)
        legal = state.legal_actions(state.current_player())
        if not legal:
            return []
        legal_logits = logits[np.asarray(legal, dtype=np.int64)]
        probs = _softmax(legal_logits)
        return [(int(a), float(p)) for a, p in zip(legal, probs)]
