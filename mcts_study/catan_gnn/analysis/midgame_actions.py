"""Mid-game action analysis (per-role action class rates per 100 turns).

Re-usable module variant of scratch_midgame_actions.py. Takes a
tournament directory and a SEATING list; emits a table of action-class
counts and per-turn rates for each role.

Standard SEATING (single-PureGnn tournaments):
    ["GnnMcts", "PureGnn", "LookaheadMctsV3", "Random"]

Dual-PureGnn SEATING (used in 1200-game tournament from 2026-05-09):
    ["PureGnnA", "PureGnnB", "LookaheadMctsV3", "Random"]

Usage as CLI (from mcts_study/):
    python -m catan_gnn.analysis.midgame_actions \\
        --run-dir runs/v3/loss_aug/01_cand8_cand10_h128_l4/.../mid_tournaments/<ts> \\
        --seating "GnnMcts,PureGnn,LookaheadMctsV3,Random"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


CHANCE_BIT = 0x80000000

# Action ID ranges (cited catan_engine/src/actions.rs:121-127).
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


def role_for(seat: int, rot: int, seating: list[str]) -> str:
    return seating[(seat + rot) % 4]


def load_all(tournament_dir: Path) -> pd.DataFrame:
    rows = []
    for parq in sorted(tournament_dir.rglob("games.rot=*.parquet")):
        rot = int(parq.name.split(".")[1].split("=")[1])
        df = pq.read_table(str(parq)).to_pandas()
        df["rot"] = rot
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No games.rot=*.parquet under {tournament_dir}")
    return pd.concat(rows, ignore_index=True)


def per_game_action_counts(action_history, rot: int, seating: list[str]) -> dict:
    """Per-game role x action_class counts. Excludes setup (idx 0..15)
    and Discards (forced)."""
    out = {role: {} for role in seating}
    out["_post_setup_actions_per_role"] = {role: 0 for role in seating}

    if len(action_history) < 17:
        return out

    turn_owner = 0  # seat 0 starts post-setup (cited Catan-Lite rules)
    for i in range(16, len(action_history)):
        a = int(action_history[i])
        if a & CHANCE_BIT:
            continue
        cat = categorize(a)
        if cat == "Discard":
            continue
        role = role_for(turn_owner, rot, seating)
        out[role][cat] = out[role].get(cat, 0) + 1
        out["_post_setup_actions_per_role"][role] += 1
        if cat == "EndTurn":
            turn_owner = (turn_owner + 1) % 4
    return out


def analyze(tournament_dir: Path, seating: list[str]) -> dict:
    """Aggregate analysis for a tournament directory. Returns a dict of
    per-role action class totals + per-turn rates."""
    g = load_all(tournament_dir)
    role_totals = {r: {} for r in seating}
    role_actions_total = {r: 0 for r in seating}
    for _, row in g.iterrows():
        per = per_game_action_counts(row["action_history"], int(row["rot"]), seating)
        for role in seating:
            for cat, n in per[role].items():
                role_totals[role][cat] = role_totals[role].get(cat, 0) + n
            role_actions_total[role] += per["_post_setup_actions_per_role"][role]
    return {
        "n_games": len(g),
        "role_totals": role_totals,
        "role_actions_total": role_actions_total,
        "seating": seating,
    }


def print_report(result: dict, label: str = "tournament") -> None:
    print(f"\n=== {label} ({result['n_games']} games) ===")
    seating = result["seating"]
    role_totals = result["role_totals"]
    role_actions_total = result["role_actions_total"]

    cats = sorted({cat for r in seating for cat in role_totals[r]})
    print(f"\n  Per-role action counts:")
    print(f"    {'category':<20s} " + "".join(f"{r:>15s}" for r in seating))
    for cat in cats:
        line = f"    {cat:<20s} "
        for r in seating:
            line += f"{role_totals[r].get(cat, 0):>15d}"
        print(line)
    print(f"    {'TOTAL ACTIONS':<20s} " + "".join(f"{role_actions_total[r]:>15d}" for r in seating))

    # Per-turn rates: per 100 EndTurns (one turn ~= one EndTurn).
    print(f"\n  Per 100 turns (EndTurn=action 204):")
    print(f"    {'category':<20s} " + "".join(f"{r:>15s}" for r in seating))
    interesting = ["BuildCity", "BuildSettlement", "BuildRoad", "BuyDevCard",
                   "ProposeTrade", "TradeBank", "PlayKnight", "PlayRoadBuilding",
                   "PlayMonopoly", "PlayYearOfPlenty", "PlayVpCard",
                   "RollDice", "EndTurn"]
    for cat in interesting:
        line = f"    {cat:<20s} "
        for r in seating:
            n = role_totals[r].get(cat, 0)
            endturns = role_totals[r].get("EndTurn", 0)
            rate = (100 * n / endturns) if endturns else 0
            line += f"{rate:>14.2f}"
        print(line)

    # ProposeTrade as % of all post-setup actions (cited diagnostic).
    print(f"\n  ProposeTrade as % of all post-setup actions:")
    for r in seating:
        prop = role_totals[r].get("ProposeTrade", 0)
        tot = role_actions_total[r]
        pct = (100 * prop / tot) if tot else 0
        print(f"    {r:>20s}: {prop} / {tot} = {pct:.2f}%")


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Tournament dir containing worker*/games.rot=*.parquet")
    p.add_argument("--seating", type=str,
                   default="GnnMcts,PureGnn,LookaheadMctsV3,Random",
                   help="Comma-separated role names matching e10's _BASE_SEATING")
    p.add_argument("--label", type=str, default="tournament",
                   help="Label for the report")
    args = p.parse_args()
    seating = [s.strip() for s in args.seating.split(",")]
    result = analyze(args.run_dir, seating)
    print_report(result, label=args.label)


if __name__ == "__main__":
    cli_main()
