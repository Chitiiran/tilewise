"""Behavioral analysis on the 1200-game e10c head-to-head tournament.

Walks every game's action_history and tallies per-role per-100-turns rates for:
  BuildCity, BuildSettlement, BuildRoad, BuyDevCard, PlayKnight,
  PlayMonopoly, PlayRoadBuilding, PlayYearOfPlenty, PlayVpCard,
  TradeBank, ProposeTrade, MoveRobber, EndTurn

Plus derived metrics:
  - Roads-per-settlement ratio (Cand 11's target metric)
  - ProposeTrade % of all post-setup actions
  - Inaction proxy (RollDice + EndTurn / total)

Adapted from scratch_midgame_actions_cell1_ep10.py. Differences:
  - TR points to the 1200-game e10c tournament.
  - SEATING is the e10c lineup: PureGnnA / PureGnnB / PureGnnC / LookaheadMctsV3.
  - Labels are pulled from the per-worker config.json so the output table
    uses the human-readable cell names.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd

ROOT = Path("/mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study")
TR = ROOT / "runs/v3/e10c_4way_1200_2026_05_26/2026-05-26T15-59-e10c_triple_gnn"

CHANCE_BIT = 0x80000000

# Action ID ranges (cited playback.py:258-300, road_pip_prior.py:32, action_classes.py:19-34)
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


def load_all_with_labels():
    """Load every games.rot=*.parquet across workers + return seating + label map."""
    # Read seating + labels from worker0/config.json
    cfg = json.loads((TR / "worker0" / "config.json").read_text())
    seating = cfg["seating"]  # ['PureGnnA','PureGnnB','PureGnnC','LookaheadMctsV3']
    label_map = {
        "PureGnnA": cfg["label_a"],
        "PureGnnB": cfg["label_b"],
        "PureGnnC": cfg["label_c"],
        "LookaheadMctsV3": "LookaheadMctsV3",
    }
    rows = []
    n_files = 0
    for w in sorted(TR.glob("worker*")):
        for parq in sorted(w.glob("games.rot=*.parquet")):
            rot = int(parq.name.split(".")[1].split("=")[1])
            df = pq.read_table(parq).to_pandas()
            df["rot"] = rot
            rows.append(df)
            n_files += 1
    return pd.concat(rows, ignore_index=True), seating, label_map, n_files


def per_game_action_counts(action_history, rot: int, seating: list[str]) -> dict:
    """Returns {role: {category: count}} + post-setup totals + endturns."""
    out = {role: {} for role in seating}
    out["_total_endturns"] = 0
    out["_post_setup_actions_per_role"] = {role: 0 for role in seating}

    if len(action_history) < 17:
        return out

    turn_owner = 0  # seat 0 starts post-setup
    for i in range(16, len(action_history)):
        a = int(action_history[i])
        if a & CHANCE_BIT:
            continue
        cat = categorize(a)
        if cat == "Discard":
            continue  # forced; don't attribute
        # role_for(seat, rot) = SEATING[(seat + rot) % 4]
        role = seating[(turn_owner + rot) % 4]
        out[role][cat] = out[role].get(cat, 0) + 1
        out["_post_setup_actions_per_role"][role] += 1
        if cat == "EndTurn":
            out["_total_endturns"] += 1
            turn_owner = (turn_owner + 1) % 4
    return out


def main():
    g, seating, label_map, n_files = load_all_with_labels()
    print(f"Loaded {n_files} parquets, {len(g)} games total")
    print(f"Seating (slot 0..3): {seating}")
    print(f"Labels:")
    for k, v in label_map.items():
        print(f"  {k:20s} = {v}")
    print()

    role_totals = {r: {} for r in seating}
    role_actions_total = {r: 0 for r in seating}

    for _, row in g.iterrows():
        per = per_game_action_counts(row["action_history"], int(row["rot"]), seating)
        for role in seating:
            for cat, n in per[role].items():
                role_totals[role][cat] = role_totals[role].get(cat, 0) + n
            role_actions_total[role] += per["_post_setup_actions_per_role"][role]

    # Display roles by label_map (human-readable)
    display_roles = [(role, label_map[role]) for role in seating]
    label_col_width = max(len(lbl) for _, lbl in display_roles) + 2

    cats = sorted({cat for r in seating for cat in role_totals[r]})

    # === Per-role action counts (raw, post-setup) ===
    print(f"=== Per-role action counts (raw, post-setup, n={len(g)} games) ===")
    header = f"{'category':<20s} " + "".join(f"{lbl:>{label_col_width}s}" for _, lbl in display_roles)
    print(header)
    for cat in cats:
        line = f"{cat:<20s} "
        for role, _ in display_roles:
            n = role_totals[role].get(cat, 0)
            line += f"{n:>{label_col_width}d}"
        print(line)
    total_line = f"{'TOTAL ACTIONS':<20s} "
    for role, _ in display_roles:
        total_line += f"{role_actions_total[role]:>{label_col_width}d}"
    print(total_line)
    print()

    # === Per-role rate per 100 turns ===
    print(f"=== Per-role rate per 100 turns (action count / EndTurns x 100) ===")
    print(header)
    for cat in cats:
        line = f"{cat:<20s} "
        for role, _ in display_roles:
            n = role_totals[role].get(cat, 0)
            endturns = role_totals[role].get("EndTurn", 0)
            rate = (100 * n / endturns) if endturns else 0
            line += f"{rate:>{label_col_width-1}.2f} "
        print(line)
    print()

    # === ProposeTrade % of all post-setup actions ===
    print(f"=== ProposeTrade as fraction of total post-setup actions ===")
    print(f"{'role':<20s} {'ProposeTrade':>15s} {'total':>10s} {'% of all actions':>20s}")
    for role, lbl in display_roles:
        prop = role_totals[role].get("ProposeTrade", 0)
        tot = role_actions_total[role]
        pct = (100 * prop / tot) if tot else 0
        print(f"{lbl:<20s} {prop:>15d} {tot:>10d} {pct:>19.2f}%")
    print()

    # === Inaction proxy: RollDice + EndTurn / total ===
    print(f"=== Inaction proxy (RollDice + EndTurn / total post-setup actions) ===")
    print(f"{'role':<20s} {'idle':>15s} {'total':>10s} {'% of all':>15s}")
    for role, lbl in display_roles:
        idle = role_totals[role].get("RollDice", 0) + role_totals[role].get("EndTurn", 0)
        tot = role_actions_total[role]
        pct = (100 * idle / tot) if tot else 0
        print(f"{lbl:<20s} {idle:>15d} {tot:>10d} {pct:>14.2f}%")
    print()

    # === Headline derived metrics ===
    print(f"=== Headline derived metrics ===")
    print(f"{'role':<28s} {'roads/100':>10s} {'settle/100':>11s} {'roads/settle':>13s} {'cities/100':>11s} {'devcard/100':>12s}")
    for role, lbl in display_roles:
        endturns = role_totals[role].get("EndTurn", 0)
        roads = role_totals[role].get("BuildRoad", 0)
        settles = role_totals[role].get("BuildSettlement", 0)
        cities = role_totals[role].get("BuildCity", 0)
        devs = role_totals[role].get("BuyDevCard", 0)
        per100 = lambda n: (100 * n / endturns) if endturns else 0
        rps = (roads / settles) if settles else float("inf")
        print(f"{lbl:<28s} {per100(roads):>9.2f}  {per100(settles):>10.2f}  {rps:>12.2f}  {per100(cities):>10.2f}  {per100(devs):>11.2f}")

    print()
    print(f"=== Reference values (cited prior journals) ===")
    print(f"  LookaheadV3 in mid-tournament (Cell 1 ep10):  18.10 roads/100, 4.53 settle/100, ratio=4.0")
    print(f"  PureGnn (Cell 1 ep10):                        17.49 roads/100, 2.43 settle/100, ratio=7.2")
    print(f"  Pass-100k 1200-game (PureGnnA / Cand 11 unrelated lineage):  11.62 roads, 6.03 cities")


if __name__ == "__main__":
    sys.exit(main() or 0)
