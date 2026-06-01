"""D3: are sims=160 policy targets too blurry? Re-search states at higher sims.

Replays a sample of self-play states from the 509-game corpus to the recorded
decision point, then runs MCTS at sims=160 vs sims=800 on the SAME state, and
compares the visit-count distributions' sharpness (peak share, entropy, argmax
agreement). If higher sims concentrate the visits, the sims=160 targets were
blurry -> the fix for PureGnn is sharper (more-sim) targets.

Usage: python -m analyses.target_sharpness_vs_sims <net_ckpt> <n_states>
"""
from __future__ import annotations
import sys, glob, random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import asyncio

from catan_bot import _engine
from catan_gnn.gnn_model import GnnModel
from catan_mcts.adapter import CatanGame
from catan_mcts.batched_evaluator import BatchedGnnEvaluator
from catan_mcts.async_mcts import AsyncMcts

CHANCE_BIT = 0x80000000


def load_model(ckpt, device):
    m = GnnModel(hidden_dim=128, num_layers=4)
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    st = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    m.load_state_dict(st); return m.to(device).eval()


def sharpness(visits):
    v = np.asarray(visits, dtype=float)
    s = v.sum()
    if s <= 0: return None
    p = v[v > 0] / s
    return float((v / s).max()), float(-(p * np.log(p)).sum())


async def main():
    ckpt = sys.argv[1]
    n_states = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt, device)
    game = CatanGame(vp_target=10, bonuses=True)

    # sample (seed, move_index) decision points from the corpus moves
    mf = glob.glob("/home/chitii/catan_data/runs/v3/az9h_corpus/*/moves*.parquet")
    md = pd.concat([pd.read_parquet(f) for f in mf], ignore_index=True)
    gf = glob.glob("/home/chitii/catan_data/runs/v3/az9h_corpus/*/games*.parquet")
    gd = pd.concat([pd.read_parquet(f) for f in gf], ignore_index=True)
    hist_by_seed = {int(s): list(h) for s, h in zip(gd["seed"], gd["action_history"])}

    rng = random.Random(0)
    sample = md.sample(min(n_states * 3, len(md)), random_state=1).to_dict("records")

    ev = BatchedGnnEvaluator(model=model, device=device, max_batch=8, window_ms=5)
    ev.start()
    results = []  # (peak160, ent160, peak800, ent800, argmax_agree)
    try:
        done = 0
        for row in sample:
            if done >= n_states:
                break
            seed = int(row["seed"]); midx = int(row["move_index"]); cp = int(row["current_player"])
            hist = hist_by_seed.get(seed)
            if hist is None:
                continue
            # replay to this player's midx-th multi-legal decision
            eng = _engine.Engine.with_rules(seed, 10, True)
            seen = 0; stopped = False
            for a in hist:
                ai = int(a)
                if ai & CHANCE_BIT:
                    eng.apply_chance_outcome(ai & ~CHANCE_BIT); continue
                if int(eng.current_player()) == cp and len(eng.legal_actions()) > 1:
                    if seen == midx:
                        stopped = True; break
                    seen += 1
                eng.step(ai)
            if not stopped:
                continue
            # build a CatanState wrapping this engine to search
            state = game.new_initial_state(seed=seed)
            state._engine = eng.clone()
            mcts = AsyncMcts(evaluator=ev, c=1.4, rng=np.random.default_rng(done))
            v160 = await mcts.search(state.clone() if hasattr(state, "clone") else state, n_sims=160)
            # fresh state clone for 800 (search mutates via clones internally but reuse engine)
            state2 = game.new_initial_state(seed=seed); state2._engine = eng.clone()
            mcts2 = AsyncMcts(evaluator=ev, c=1.4, rng=np.random.default_rng(done))
            v800 = await mcts2.search(state2, n_sims=800)
            s160 = sharpness(v160); s800 = sharpness(v800)
            if s160 is None or s800 is None:
                continue
            am160 = int(np.argmax(v160)); am800 = int(np.argmax(v800))
            results.append((s160[0], s160[1], s800[0], s800[1], int(am160 == am800)))
            done += 1
            print(f"  state {done}/{n_states}: peak160={s160[0]:.2f} peak800={s800[0]:.2f} agree={int(am160==am800)}", flush=True)
    finally:
        await ev.stop()

    r = np.array(results)
    print(f"states analyzed: {len(r)}")
    print(f"sims=160:  mean peak-share={r[:,0].mean():.3f}  mean entropy={r[:,1].mean():.3f}  frac flat(peak<0.3)={(r[:,0]<0.3).mean():.2f}")
    print(f"sims=800:  mean peak-share={r[:,2].mean():.3f}  mean entropy={r[:,3].mean():.3f}  frac flat(peak<0.3)={(r[:,2]<0.3).mean():.2f}")
    print(f"argmax(160) == argmax(800):  {r[:,4].mean():.2f}  (low = 160 picks a different move than deeper search)")
    print()
    if r[:,2].mean() - r[:,0].mean() > 0.08:
        print("=> sims=800 targets are NOTABLY SHARPER. sims=160 was too shallow; the")
        print("   policy targets were blurry. FIX: generate self-play at higher sims.")
    else:
        print("=> higher sims did NOT sharpen targets much. The blur is intrinsic")
        print("   (positions genuinely have several near-equal moves) -> capacity/argmax issue, not sims.")


if __name__ == "__main__":
    asyncio.run(main())
