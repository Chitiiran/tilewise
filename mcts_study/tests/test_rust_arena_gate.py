"""Phase 7 END-TO-END GATE: Rust arena per-game (winner_seat, timed_out,
vp_margin) == Python arena._play_arena_game, and the aggregate winrate is
identical.

Two small distinct nets -> two .ts. The DUAL-RNG path is exercised: the
game-level chance fast-path uses random.Random(seed) (MtRng on the Rust side),
the per-seat MCTS uses np.random.default_rng(seed+11/+13). Arena is greedy.
"""
import asyncio
import os

import numpy as np
import pytest
import random
import torch
from torch_geometric.data import Batch

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg
from catan_gnn.export_torchscript import export
from catan_mcts.adapter import CatanGame
from catan_mcts.async_mcts import AsyncMcts
from catan_az.arena import _play_arena_game, seating_for_rotation, seed_plan, ArenaResult

catan_mcts_rs = pytest.importorskip("catan_mcts_rs")

HIDDEN, LAYERS = 32, 2
SIMS = 12
GAMES = int(os.environ.get("RUST_ARENA_GATE_GAMES", "8"))  # multiple of 4
SEED_BASE = 30_000_000


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
def nets(tmp_path_factory):
    d = tmp_path_factory.mktemp("nets")
    out = {}
    for label, seed in (("cand", 111), ("champ", 222)):
        torch.manual_seed(seed)
        m = GnnModel(hidden_dim=HIDDEN, num_layers=LAYERS).eval()
        ck = d / f"{label}.pt"
        torch.save({"model_state": m.state_dict()}, ck)
        ts = d / f"{label}.ts"
        export(checkpoint=ck, out_ts=ts, hidden_dim=HIDDEN, num_layers=LAYERS)
        out[label] = (m, str(ts))
    return out


def _py_arena_game(cand_model, champ_model, seed, rot):
    seating = seating_for_rotation(rot)
    mcts_c = AsyncMcts(evaluator=SyncEval(cand_model), c=1.4,
                       rng=np.random.default_rng(seed + 11))
    mcts_x = AsyncMcts(evaluator=SyncEval(champ_model), c=1.4,
                       rng=np.random.default_rng(seed + 13))
    return asyncio.run(_play_arena_game(
        game=CatanGame(), seed=seed, seating=seating,
        mcts_cand=mcts_c, mcts_champ=mcts_x, sims=SIMS, max_seconds=None))


def test_arena_per_game_and_winrate_parity(nets):
    cand_model, cand_ts = nets["cand"]
    champ_model, champ_ts = nets["champ"]
    plan = seed_plan(seed_base=SEED_BASE, games=GAMES)

    py_res = ArenaResult()
    rs_res = ArenaResult()
    for rot, seed in plan:
        py_winner, py_timeout, py_margin = _py_arena_game(cand_model, champ_model, seed, rot)
        rs_winner, rs_timeout, rs_margin = catan_mcts_rs.debug_arena_game(
            cand_ts, champ_ts, seed, rot, SIMS)
        assert rs_winner == py_winner, f"winner seed={seed} rot={rot}"
        assert rs_timeout == py_timeout, f"timeout seed={seed} rot={rot}"
        assert rs_margin == py_margin, f"vp_margin seed={seed} rot={rot}"

        # aggregate both ways using the SAME role logic as run_arena
        for res, winner, timeout in ((py_res, py_winner, py_timeout),
                                     (rs_res, rs_winner, rs_timeout)):
            seating = seating_for_rotation(rot)
            role = seating[winner] if winner >= 0 else None
            if timeout:
                res.timeouts += 1
            if role == "cand":
                res.wins_cand += 1
            elif role == "champ":
                res.wins_champ += 1
            else:
                res.draws += 1

    assert rs_res.wins_cand == py_res.wins_cand
    assert rs_res.wins_champ == py_res.wins_champ
    assert rs_res.draws == py_res.draws
    assert rs_res.winrate_cand == py_res.winrate_cand
