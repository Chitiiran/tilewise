"""Mid-game action analysis for Cell 1 (Cand 8 + Cand 10) ep10 mid-tournament.

Adapted from scratch_midgame_actions.py. Differences:
  - TR points to Cell 1's ep10 tournament dir.
  - SEATING is [GnnMcts, PureGnn, LookaheadMctsV3, Random] (cited e10_v3_tournament.py:48).
  - Rotation convention identical: seat s in rot r plays role SEATING[(s + r) % 4]
    because the tournament uses seating[rot_idx:] + seating[:rot_idx] (e10_v3_tournament.py:87).

Goal (Option 2 from the next-actions list): measure PureGnn BuildCity-per-turn rate to
decide whether Cand 2 (city-upgrade target boost) should run on top of Cand 8 + Cand 10.

Baseline gap to close (cited journals):
  - Pass-100k 1200-game tournament:
      LookaheadV3 cities/100 turns = 6.32
      PureGnnA            = 6.03  (100k corpus mostly closed the gap)
      PureGnnB            = 3.11  (pass-3 corpus, half the rate)
  - Decision rule: if Cell 1 PureGnn closed >= half the gap to Lookahead's 6.32,
    skip Cand 2. Otherwise run Cand 2.
"""
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd

ROOT = Path("/mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study")
TR = ROOT / "runs/v3/loss_aug/01_cand8_cand10_h128_l4/training_h128_l4/mid_tournaments/2026-05-12T05-54-e10_v3_tournament"
SEATING = ["GnnMcts", "PureGnn", "LookaheadMctsV3", "Random"]  # cited e10_v3_tournament.py:48

CHANCE_BIT = 0x80000000

# Action ID ranges (cited playback.py:258-300)
SETTLE_RANGE = (0, 54)
CITY_RANGE = (54, 108)
ROAD_RANGE = (108, 180)
ROBBER_RANGE = (180, 199)
DISCARD_RANGE = (199, 204)
ENDTURN = 204
ROLL_DICE = 205
TRADE_BANK_RANGE = (206, 226)
BUY_DEV = 226
PLAY_KNIGHT = 227
PLAY_ROADBUILDING = 228
PLAY_MONO_RANGE = (229, 234)
PLAY_YOP_RANGE = (234, 259)
PLAY_VP = 259
PROPOSE_TRADE_RANGE = (260, 280)


def categorize(a: int) -> str:
    if SETTLE_RANGE[0] <= a < SETTLE_RANGE[1]: return "BuildSettlement"
    if CITY_RANGE[0] <= a < CITY_RANGE[1]: return "BuildCity"
    if ROAD_RANGE[0] <= a < ROAD_RANGE[1]: return "BuildRoad"
    if ROBBER_RANGE[0] <= a < ROBBER_RANGE[1]: return "MoveRobber"
    if DISCARD_RANGE[0] <= a < DISCARD_RANGE[1]: return "Discard"
    if a == ENDTURN: return "EndTurn"
    if a == ROLL_DICE: return "RollDice"
    if TRADE_BANK_RANGE[0] <= a < TRADE_BANK_RANGE[1]: return "TradeBank"
    if a == BUY_DEV: return "BuyDevCard"
    if a == PLAY_KNIGHT: return "PlayKnight"
    if a == PLAY_ROADBUILDING: return "PlayRoadBuilding"
    if PLAY_MONO_RANGE[0] <= a < PLAY_MONO_RANGE[1]: return "PlayMonopoly"
    if PLAY_YOP_RANGE[0] <= a < PLAY_YOP_RANGE[1]: return "PlayYearOfPlenty"
    if a == PLAY_VP: return "PlayVpCard"
    if PROPOSE_TRADE_RANGE[0] <= a < PROPOSE_TRADE_RANGE[1]: return "ProposeTrade"
    return f"unknown:{a}"


def role_for(seat: int, rot: int) -> str:
    return SEATING[(seat + rot) % 4]


def load_all() -> pd.DataFrame:
    rows = []
    n_workers = 0
    n_files = 0
    for w in sorted(TR.glob("worker*")):
        n_workers += 1
        for parq in sorted(w.glob("games.rot=*.parquet")):
            rot = int(parq.name.split(".")[1].split("=")[1])
            df = pq.read_table(parq).to_pandas()
            df["rot"] = rot
            rows.append(df)
            n_files += 1
    print(f"loaded {n_files} parquets from {n_workers} workers")
    return pd.concat(rows, ignore_index=True)


def per_game_action_counts(action_history, rot: int) -> dict:
    out = {role: {} for role in SEATING}
    out["_total_endturns"] = 0
    out["_post_setup_actions_per_role"] = {role: 0 for role in SEATING}

    if len(action_history) < 17:
        return out

    turn_owner = 0
    for i in range(16, len(action_history)):
        a = int(action_history[i])
        if a & CHANCE_BIT:
            continue
        cat = categorize(a)
        if cat == "Discard":
            continue
        role = role_for(turn_owner, rot)
        out[role][cat] = out[role].get(cat, 0) + 1
        out["_post_setup_actions_per_role"][role] += 1
        if cat == "EndTurn":
            out["_total_endturns"] += 1
            turn_owner = (turn_owner + 1) % 4
    return out


def main():
    g = load_all()
    print(f"total games: {len(g)}")

    role_totals = {r: {} for r in SEATING}
    role_actions_total = {r: 0 for r in SEATING}

    for _, row in g.iterrows():
        per = per_game_action_counts(row["action_history"], int(row["rot"]))
        for role in SEATING:
            for cat, n in per[role].items():
                role_totals[role][cat] = role_totals[role].get(cat, 0) + n
            role_actions_total[role] += per["_post_setup_actions_per_role"][role]

    cats = sorted({cat for r in SEATING for cat in role_totals[r]})
    print(f"\n=== per-role action counts (raw, post-setup) ===")
    print(f"{'category':<20s} " + "".join(f"{r:>16s}" for r in SEATING))
    for cat in cats:
        line = f"{cat:<20s} "
        for r in SEATING:
            n = role_totals[r].get(cat, 0)
            line += f"{n:>16d}"
        print(line)
    print(f"{'TOTAL ACTIONS':<20s} " + "".join(f"{role_actions_total[r]:>16d}" for r in SEATING))

    print(f"\n=== headline rates per 100 turns (vs pass-100k baseline) ===")
    print(f"  Baseline cited (pass-100k, 1200 games):")
    print(f"    LookaheadV3 cities/100 turns = 6.32")
    print(f"    PureGnnA                      = 6.03")
    print(f"    PureGnnB                      = 3.11")
    print(f"    LookaheadV3 roads/100 turns   = 23.05")
    print(f"    PureGnnA                      = 11.62")
    print(f"    PureGnnB                      = 12.44")
    print()
    for r in SEATING:
        endturns = role_totals[r].get("EndTurn", 0)
        cities = role_totals[r].get("BuildCity", 0)
        roads = role_totals[r].get("BuildRoad", 0)
        settles = role_totals[r].get("BuildSettlement", 0)
        buyds = role_totals[r].get("BuyDevCard", 0)
        knights = role_totals[r].get("PlayKnight", 0)
        vpcards = role_totals[r].get("PlayVpCard", 0)
        proposes = role_totals[r].get("ProposeTrade", 0)
        tot = role_actions_total[r]
        per100 = lambda n: 100 * n / max(endturns, 1)
        print(f"  {r}:")
        print(f"    EndTurns:           {endturns}")
        print(f"    BuildCity:          {cities:>6d}   ({per100(cities):5.2f} per 100 turns)")
        print(f"    BuildSettlement:    {settles:>6d}   ({per100(settles):5.2f} per 100 turns)")
        print(f"    BuildRoad:          {roads:>6d}   ({per100(roads):5.2f} per 100 turns)")
        print(f"    BuyDevCard:         {buyds:>6d}   ({per100(buyds):5.2f} per 100 turns)")
        print(f"    PlayVpCard:         {vpcards:>6d}   ({per100(vpcards):5.2f} per 100 turns)")
        print(f"    PlayKnight:         {knights:>6d}   ({per100(knights):5.2f} per 100 turns)")
        propose_pct = 100 * proposes / max(tot, 1)
        print(f"    ProposeTrade:       {proposes:>6d}   ({propose_pct:5.2f}% of all actions)")

    # Decision gate
    print(f"\n=== Cand 2 decision gate ===")
    pg_cities = role_totals["PureGnn"].get("BuildCity", 0)
    pg_endturns = role_totals["PureGnn"].get("EndTurn", 0)
    pg_rate = 100 * pg_cities / max(pg_endturns, 1)
    look_rate_baseline = 6.32
    pgb_rate_baseline = 3.11  # the gap to close
    gap_total = look_rate_baseline - pgb_rate_baseline  # 3.21
    gap_closed = pg_rate - pgb_rate_baseline
    pct_closed = 100 * gap_closed / gap_total if gap_total > 0 else 0
    print(f"  PureGnn (Cell 1 ep10) BuildCity rate: {pg_rate:.2f} per 100 turns")
    print(f"  Pass-100k Lookahead baseline:         {look_rate_baseline:.2f}")
    print(f"  Pass-3 PureGnnB baseline (worst):     {pgb_rate_baseline:.2f}")
    print(f"  Gap closed vs PureGnnB:               {gap_closed:+.2f} ({pct_closed:.1f}% of {gap_total:.2f} gap)")
    if pct_closed >= 50:
        print(f"  DECISION: SKIP Cand 2 — gap is mostly closed under Cand 8+10.")
    elif pg_rate >= look_rate_baseline * 0.85:
        print(f"  DECISION: SKIP Cand 2 — within 15% of Lookahead's city rate.")
    else:
        print(f"  DECISION: RUN Cand 2 — city gap remains; structural fix worth trying.")


if __name__ == "__main__":
    sys.exit(main() or 0)
