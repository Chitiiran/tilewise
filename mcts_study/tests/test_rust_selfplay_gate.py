"""Phase 6 END-TO-END GATE (spec §5, NON-NEGOTIABLE): the same seeds through
Python AND Rust self-play produce IDENTICAL game records.

Compares length, winner, action_history, per-move (current_player, move_index,
legal_mask, visit_counts, action_taken, root_value). Covers greedy (self_play=
False) and exploratory (self_play=True) paths. Uses a small fixed net exported
to .ts for Rust and run eagerly (sync) for Python — proven bit-exact (Phase 0).

Default 12 seeds for CI speed; the full 100-seed gate runs via
scripts/rust_selfplay_gate_100.sh before the loop trusts Rust self-play.
"""
import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg
from catan_gnn.export_torchscript import export
from catan_mcts.adapter import CatanGame
from catan_mcts.async_mcts import play_one_async_game

catan_mcts_rs = pytest.importorskip("catan_mcts_rs")

HIDDEN, LAYERS = 32, 2
N_SIMS = 12
# Default 12 seeds for CI speed; the full 100-seed gate sets the env var.
SEEDS = list(range(int(os.environ.get("RUST_SELFPLAY_GATE_SEEDS", "12"))))


def _softmax(x):
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


class SyncEval:
    def __init__(self, model):
        self.model = model.eval()

    async def eval_leaf(self, state):
        if state.is_terminal():
            return np.asarray(state.returns(), dtype=np.float32), None
        obs = state._engine.observation()
        with torch.no_grad():
            v, logits = self.model(Batch.from_data_list([state_to_pyg(obs)]))
        value = v.squeeze(0).numpy().astype(np.float32)
        logits = logits.squeeze(0).numpy().astype(np.float32)
        legal = state.legal_actions()
        la = np.asarray(legal, dtype=np.int64)
        probs = _softmax(logits[la])
        return value, [(int(a), float(p)) for a, p in zip(legal, probs)]


@pytest.fixture(scope="module")
def net(tmp_path_factory):
    torch.manual_seed(12345)
    model = GnnModel(hidden_dim=HIDDEN, num_layers=LAYERS).eval()
    d = tmp_path_factory.mktemp("net")
    ckpt = d / "n.pt"
    torch.save({"model_state": model.state_dict()}, ckpt)
    ts = d / "n.ts"
    export(checkpoint=ckpt, out_ts=ts, hidden_dim=HIDDEN, num_layers=LAYERS)
    return model, str(ts)


def _py_game(model, seed, self_play):
    return asyncio.run(play_one_async_game(
        game=CatanGame(), seed=seed, evaluator=SyncEval(model), n_sims=N_SIMS,
        rng=np.random.default_rng(seed), self_play=self_play))


def _assert_identical(py, rs):
    assert rs["length_in_moves"] == py.length_in_moves
    assert rs["winner"] == py.winner
    assert list(rs["action_history"]) == list(py.action_history)
    assert len(rs["moves"]) == len(py.moves)
    for mr, mp in zip(rs["moves"], py.moves):
        assert mr["current_player"] == mp.current_player
        assert mr["move_index"] == mp.move_index
        assert mr["action_taken"] == mp.action_taken
        assert list(mr["visit_counts"]) == list(mp.visit_counts)
        assert list(mr["legal_mask"]) == [int(x) for x in mp.legal_mask]
        assert np.float64(mr["root_value"]).tobytes() == np.float64(mp.root_value).tobytes()


@pytest.mark.parametrize("seed", SEEDS)
def test_greedy_selfplay_record_parity(net, seed):
    model, ts = net
    py = _py_game(model, seed, self_play=False)
    rs = catan_mcts_rs.debug_selfplay_game(ts, seed, N_SIMS, False)
    _assert_identical(py, rs)


@pytest.mark.parametrize("seed", SEEDS)
def test_exploratory_selfplay_record_parity(net, seed):
    model, ts = net
    py = _py_game(model, seed, self_play=True)
    rs = catan_mcts_rs.debug_selfplay_game(ts, seed, N_SIMS, True)
    _assert_identical(py, rs)
