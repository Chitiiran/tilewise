"""Where does GnnMcts diverge from PureGnn? (the search-correction diagnostic)

For each game in the Gate-2 tournament, replay the GnnMcts seat's decisions.
At each such decision, compare:
  - what was ACTUALLY played (GnnMcts's MCTS choice, from action_history), vs
  - what PureGnn (argmax of the SAME net's policy head) WOULD have played.

Where they differ = a state where search overrode the raw policy. Categorize
those divergences by action type to find the policy's systematic blind spots.

Usage:
  python -m analyses.gnnmcts_vs_puregnn_divergence <gate2_run_dir> <net_ckpt> <seed_base>

The GnnMcts seat = slot 1 in e10e_async seating [PureGnnA, GnnMctsB, PureGnnC,
LookaheadMctsV3]; its role rotates per game.
"""
from __future__ import annotations

import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from catan_bot import _engine
from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg
from torch_geometric.data import Batch

CHANCE_BIT = 0x80000000
SEATING = ["PureGnnA", "GnnMctsB", "PureGnnC", "LookaheadMctsV3"]
GNNMCTS_SLOT = 1  # GnnMctsB

# Action categories (cited observation.rs ranges)
def categorize(a: int) -> str:
    if 0 <= a < 54: return "settle"
    if 54 <= a < 108: return "city"
    if 108 <= a < 180: return "road"
    if 180 <= a < 199: return "robber"
    if 199 <= a < 204: return "discard"
    if a == 204: return "endturn"
    if a == 205: return "roll"
    if 206 <= a < 226: return "trade_bank"
    if a == 226: return "buy_dev"
    if a == 227: return "play_knight"
    if a == 228: return "play_roadbuilding"
    if 229 <= a < 234: return "play_mono"
    if 234 <= a < 259: return "play_yop"
    if a == 259: return "play_vp"
    if 260 <= a < 280: return "propose_trade"
    return "other"


def load_model(ckpt, device="cpu"):
    m = GnnModel(hidden_dim=128, num_layers=4)
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    st = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    m.load_state_dict(st)
    return m.to(device).eval()


@torch.no_grad()
def puregnn_argmax(model, eng, device="cpu"):
    """What would PureGnn (argmax over legal) play in this engine state?"""
    legal = [int(a) for a in eng.legal_actions()]
    if len(legal) <= 1:
        return legal[0] if legal else None
    obs = eng.observation()
    data = state_to_pyg(obs).to(device)
    _, logits = model(Batch.from_data_list([data]))
    lg = logits.squeeze(0).cpu().numpy()
    la = np.asarray(legal, dtype=np.int64)
    return int(la[int(np.argmax(lg[la]))])


def main():
    run_dir = Path(sys.argv[1])
    ckpt = sys.argv[2]
    seed_base = int(sys.argv[3])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt, device)

    parqs = list(run_dir.rglob("games*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parqs], ignore_index=True)
    print(f"games={len(df)}  net={Path(ckpt).name}  device={device}\n")

    total_decisions = 0
    divergences = 0
    # Of the GnnMcts decisions: category of the ACTUAL (MCTS) move when it
    # DIVERGED from PureGnn, and what PureGnn wanted instead.
    div_mcts_cat = Counter()       # what MCTS chose (that PureGnn missed)
    div_puregnn_cat = Counter()    # what PureGnn wrongly wanted
    agree_cat = Counter()          # categories where they agreed
    fails = 0

    for _, row in df.iterrows():
        seed = int(row["seed"])
        rot = (seed - seed_base) // 10000
        if not (0 <= rot < 4):
            rot = 0
        # which absolute seat is the GnnMcts seat this game?
        seating = SEATING[rot:] + SEATING[:rot]
        gnnmcts_seat = seating.index("GnnMctsB")
        hist = [int(x) for x in row["action_history"]]
        eng = _engine.Engine.with_rules(seed, 10, True)
        ok = True
        for a in hist:
            if a & CHANCE_BIT:
                try:
                    eng.apply_chance_outcome(a & ~CHANCE_BIT)
                except Exception:
                    ok = False; break
                continue
            try:
                cp = int(eng.current_player())
            except Exception:
                ok = False; break
            # Is this a GnnMcts-seat real decision (>1 legal)?
            if cp == gnnmcts_seat:
                legal = [int(x) for x in eng.legal_actions()]
                if len(legal) > 1:
                    total_decisions += 1
                    pg = puregnn_argmax(model, eng, device)
                    actual = a  # what MCTS actually played
                    if pg is not None and pg != actual:
                        divergences += 1
                        div_mcts_cat[categorize(actual)] += 1
                        div_puregnn_cat[categorize(pg)] += 1
                    else:
                        agree_cat[categorize(actual)] += 1
            try:
                eng.step(a)
            except Exception:
                ok = False; break
        if not ok:
            fails += 1

    print(f"replay failures: {fails}/{len(df)}")
    print(f"GnnMcts decisions analyzed: {total_decisions}")
    print(f"DIVERGENCES (MCTS != PureGnn argmax): {divergences} "
          f"({100*divergences/max(total_decisions,1):.1f}%)\n")

    print("=== When they DIVERGED — what MCTS chose (the move PureGnn MISSED) ===")
    for cat, n in div_mcts_cat.most_common():
        print(f"  {cat:<20} {n:5}  ({100*n/max(divergences,1):4.1f}% of divergences)")
    print("\n=== When they DIVERGED — what PureGnn WRONGLY wanted instead ===")
    for cat, n in div_puregnn_cat.most_common():
        print(f"  {cat:<20} {n:5}  ({100*n/max(divergences,1):4.1f}%)")
    print("\n=== When they AGREED — category breakdown ===")
    tot_agree = sum(agree_cat.values())
    for cat, n in agree_cat.most_common():
        print(f"  {cat:<20} {n:5}  ({100*n/max(tot_agree,1):4.1f}%)")

    # The KEY signal: per-category divergence RATE (how often does the policy
    # get THIS action type wrong relative to how often it comes up?)
    print("\n=== POLICY BLIND SPOTS — divergence rate by action type MCTS chose ===")
    print("    (high rate = search frequently overrides the policy on this move type)")
    all_cats = set(div_mcts_cat) | set(agree_cat)
    rows = []
    for cat in all_cats:
        d = div_mcts_cat.get(cat, 0)
        ag = agree_cat.get(cat, 0)
        tot = d + ag
        if tot >= 5:
            rows.append((cat, d, tot, 100*d/tot))
    for cat, d, tot, rate in sorted(rows, key=lambda x: -x[3]):
        print(f"  {cat:<20} {d:4}/{tot:<4} chosen-by-MCTS were policy-misses  ({rate:4.1f}%)")


if __name__ == "__main__":
    sys.exit(main() or 0)
