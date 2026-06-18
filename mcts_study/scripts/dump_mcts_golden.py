"""Dump Python MCTS golden (visit_counts, best_action, root_value) for the Rust
parity test. Uses a small fixed-seed net exported to .ts (so Rust loads the
identical graph) and a SYNCHRONOUS evaluator wrapping the eager model.

For each case (seed, prefix_len, n_sims, eps, alpha) we:
  - build CatanState(seed), replay prefix_len random moves to reach a decision
    node (skipping chance/single-legal), recording the engine action history
  - run AsyncMcts(evaluator, c=1.4, rng=default_rng(rng_seed), dirichlet...)
    .search(state, n_sims) under asyncio
  - dump visit_counts[280], best_action (argmax), last_root_value, and the
    engine history needed for Rust to reconstruct the exact same state.

Writes tests/data/mcts_golden.json and the net .ts next to it.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg
from catan_gnn.export_torchscript import export
from catan_mcts.adapter import CatanGame
from catan_mcts.async_mcts import AsyncMcts
from catan_mcts import ACTION_SPACE_SIZE

OUT = Path(__file__).resolve().parents[1] / "tests" / "data"
OUT.mkdir(parents=True, exist_ok=True)
HIDDEN, LAYERS = 32, 2


def _softmax(x):
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


class SyncEval:
    """Synchronous (no-batch) evaluator matching BatchedGnnEvaluator.eval_leaf
    output: (value[4] ego-relative, priors[(action,prob)] | None)."""

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
        legal_arr = np.asarray(legal, dtype=np.int64)
        probs = _softmax(logits[legal_arr])
        priors = [(int(a), float(p)) for a, p in zip(legal, probs)]
        return value, priors


def reach_decision(seed, prefix_len):
    """Replay prefix_len *decision* moves (auto-resolving chance/single-legal),
    return (state, history-as-(is_chance,id)-list) at a multi-legal node."""
    st = CatanGame().new_initial_state(seed=seed)
    rng = np.random.default_rng(seed ^ 0xABCD)
    taken = 0
    while not st.is_terminal() and taken < prefix_len:
        if st.is_chance_node():
            outs = st.chance_outcomes()
            r = float(rng.random())
            cum, chosen = 0.0, outs[-1][0]
            for v, p in outs:
                cum += p
                if r <= cum:
                    chosen = v
                    break
            st.apply_action(int(chosen))
            continue
        legal = st.legal_actions()
        if len(legal) == 1:
            st.apply_action(int(legal[0]))
            continue
        a = int(legal[int(rng.integers(len(legal)))])
        st.apply_action(a)
        taken += 1
    # Encode the engine history as (is_chance, id) for Rust replay.
    CHANCE_FLAG = 0x80000000
    entries = []
    for h in st._engine.action_history():
        h = int(h)
        if h & CHANCE_FLAG:
            entries.append((True, h & ~CHANCE_FLAG))
        else:
            entries.append((False, h))
    return st, entries


async def run_case(model_ts_eager, seed, prefix_len, n_sims, eps, alpha, rng_seed):
    st, entries = reach_decision(seed, prefix_len)
    if st.is_terminal():
        return None
    ev = SyncEval(model_ts_eager)
    mcts = AsyncMcts(evaluator=ev, c=1.4, rng=np.random.default_rng(rng_seed),
                     dirichlet_alpha=alpha, dirichlet_eps=eps)
    vc = await mcts.search(st, n_sims=n_sims)
    return {
        "seed": seed, "prefix_len": prefix_len, "n_sims": n_sims,
        "eps": eps, "alpha": alpha, "rng_seed": rng_seed,
        "entries": entries,
        "visit_counts": [int(x) for x in vc],
        "best_action": int(np.argmax(vc)),
        "root_value": float(mcts.last_root_value),
    }


def main():
    torch.manual_seed(12345)
    model = GnnModel(hidden_dim=HIDDEN, num_layers=LAYERS).eval()
    ckpt = OUT / "mcts_net.pt"
    torch.save({"model_state": model.state_dict()}, ckpt)
    ts = OUT / "mcts_net.ts"
    export(checkpoint=ckpt, out_ts=ts, hidden_dim=HIDDEN, num_layers=LAYERS)

    cases = []
    # eps=0 (arena/greedy) AND eps>0 (self-play) cases, small n_sims.
    specs = [
        (101, 6, 8, 0.0, 0.8, 1011),
        (202, 10, 16, 0.0, 0.8, 2022),
        (303, 8, 8, 0.25, 0.8, 3033),
        (404, 12, 16, 0.25, 0.8, 4044),
        (505, 4, 32, 0.0, 0.8, 5055),
    ]
    for (seed, plen, sims, eps, alpha, rng_seed) in specs:
        c = asyncio.run(run_case(model, seed, plen, sims, eps, alpha, rng_seed))
        if c is not None:
            c["net_ts"] = str(ts)
            cases.append(c)
    (OUT / "mcts_golden.json").write_text(json.dumps(cases, indent=1))
    print(f"wrote {len(cases)} cases -> {OUT/'mcts_golden.json'}")


if __name__ == "__main__":
    main()
