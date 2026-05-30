"""Deep behavioral analysis on the 1200-game full-Catan tournament.

Walks every game from `runs/v3/e10d_4gnn_fullcatan_1200_2026_05_27/`,
replays the action_history through the engine, and extracts a wide
behavioral profile per cell. Stratified by winner/loser where relevant.

Groups collected:
  1. BUILD DYNAMICS — roads / settlements / cities per game, by W/L,
     roads-per-settlement ratio.
  2. BONUS ECONOMY — LR length, LA knights played, % wins with LR
     bonus, % wins with LA bonus, VP composition (from buildings vs
     from bonuses vs VP cards).
  3. TRADE DYNAMICS — TradeBank attempts, ProposeTrade attempts, trade
     success rate (compare proposer's hand before/after ProposeTrade),
     resource flow (net ore + wheat + sheep + brick + wood traded).
  4. CLOSEOUT — game length distribution by winner, time-to-first-city
     in EndTurn ticks, VP trajectory snapshots (every 10 turns).
  5. PORT USAGE — count of own settlements on port vertices, by W/L.
  6. RESOURCE SPECIALIZATION — by hex resource type adjacent to settled
     vertices.
  7. ROBBER TARGETING — when this cell moved the robber, which player's
     settlements ended up adjacent to the robber hex.

Single replay pass per game. Expensive: ~20-30 min for 1200 games.

Cited:
  - SCALAR_VP=8, SCALAR_LR_LEN=26, SCALAR_KNIGHTS=30, SCALAR_SETTL_BUILT=34,
    SCALAR_CITY_BUILT=38, SCALAR_ROAD_BUILT=42, SCALAR_LR_HOLDER=52,
    SCALAR_LA_HOLDER=53 (observation.rs:50-61).
  - Hand slots: scalars[0..5) = viewer hand (wood, brick, sheep, wheat, ore).
  - Per-player blocks are viewer-perspective; index 0 is always viewer.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from catan_bot import _engine

ROOT = Path("/mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study")
TR = ROOT / "runs/v3/e10d_4gnn_fullcatan_1200_2026_05_27/2026-05-27T19-52-e10d_quad_gnn"

CHANCE_BIT = 0x80000000

# Action ID ranges (cited)
SETTLE_RANGE = (0, 54)
CITY_RANGE = (54, 108)
ROAD_RANGE = (108, 180)
ROBBER_RANGE = (180, 199)   # MoveRobber to hex_id
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

# Resource indices (cited observation.rs:12)
RES_WOOD, RES_BRICK, RES_SHEEP, RES_WHEAT, RES_ORE = 0, 1, 2, 3, 4
RES_NAMES = ["wood", "brick", "sheep", "wheat", "ore"]

# Scalar offsets (cited)
SCALAR_HAND = 0
SCALAR_VP = 8
SCALAR_LR_LEN = 26
SCALAR_KNIGHTS = 30
SCALAR_SETTL_BUILT = 34
SCALAR_CITY_BUILT = 38
SCALAR_ROAD_BUILT = 42
SCALAR_PORTS = 46
SCALAR_LR_HOLDER = 52
SCALAR_LA_HOLDER = 53


def _decode_trade(a: int, base: int) -> tuple[int, int]:
    """Returns (give_res, get_res) for a TradeBank or ProposeTrade action."""
    idx = a - base
    give = idx // 4
    get_in_remaining = idx % 4
    others = [r for r in range(5) if r != give]
    return give, others[get_in_remaining]


def _categorize(a: int) -> str:
    if SETTLE_RANGE[0] <= a < SETTLE_RANGE[1]: return "settle"
    if CITY_RANGE[0] <= a < CITY_RANGE[1]: return "city"
    if ROAD_RANGE[0] <= a < ROAD_RANGE[1]: return "road"
    if ROBBER_RANGE[0] <= a < ROBBER_RANGE[1]: return "robber"
    if DISCARD_RANGE[0] <= a < DISCARD_RANGE[1]: return "discard"
    if a == ENDTURN: return "endturn"
    if a == ROLL_DICE: return "roll"
    if TRADE_BANK_RANGE[0] <= a < TRADE_BANK_RANGE[1]: return "trade_bank"
    if a == BUY_DEV: return "buy_dev"
    if a == PLAY_KNIGHT: return "play_knight"
    if a == PLAY_ROADBUILDING: return "play_rb"
    if PLAY_MONO_RANGE[0] <= a < PLAY_MONO_RANGE[1]: return "play_mono"
    if PLAY_YOP_RANGE[0] <= a < PLAY_YOP_RANGE[1]: return "play_yop"
    if a == PLAY_VP: return "play_vp"
    if PROPOSE_TRADE_RANGE[0] <= a < PROPOSE_TRADE_RANGE[1]: return "propose_trade"
    return "unknown"


def _build_vertex_to_hexes() -> dict[int, list[int]]:
    from catan_gnn.adjacency import HEX_TO_VERTICES
    v2h: dict[int, list[int]] = {v: [] for v in range(54)}
    for h, vs in enumerate(HEX_TO_VERTICES):
        for v in vs:
            v2h[int(v)].append(h)
    return v2h


# Static board topology — built once
V2H = _build_vertex_to_hexes()


def analyze_game(seed: int, action_history, seating: list[str], rot: int,
                 winner_seat: int, final_vp: list[int]):
    """Replay one game. Return per-role metrics dict.

    Returns dict[role][metric] -> value. For aggregation across games,
    metrics are summed (except per-game distributions which append).
    """
    eng = _engine.Engine.with_rules(int(seed), 10, True)
    obs0 = eng.observation_for(0)

    # Static board info from initial observation
    hex_feat = obs0["hex_features"]
    hex_resource = []  # 0..4 or -1 for desert
    for h in range(19):
        if float(hex_feat[h, 7]) >= 0.5:
            hex_resource.append(-1)
        else:
            for r in range(5):
                if float(hex_feat[h, r]) >= 0.5:
                    hex_resource.append(r)
                    break
            else:
                hex_resource.append(-1)

    # Per-vertex port flags from initial vertex_features
    vf0 = obs0["vertex_features"]
    vertex_has_port = [
        any(float(vf0[v, 7 + k]) >= 0.5 for k in range(6))
        for v in range(54)
    ]

    def _role_for(seat: int) -> str:
        return seating[(seat + rot) % 4]

    # Per-role counters
    out = {r: {
        "n_settle_built": 0,         # main-phase settlements only (post-setup)
        "n_city_built": 0,
        "n_road_built": 0,           # main-phase roads only
        "n_trade_bank": 0,
        "n_propose_trade": 0,
        "n_propose_trade_succeeded": 0,
        "ore_given": 0, "ore_received": 0,
        "wheat_given": 0, "wheat_received": 0,
        "wood_given": 0, "wood_received": 0,
        "brick_given": 0, "brick_received": 0,
        "sheep_given": 0, "sheep_received": 0,
        "n_buy_dev": 0,
        "n_play_knight": 0,
        "n_play_rb": 0, "n_play_mono": 0, "n_play_yop": 0,
        # Settlement placement profile (all settlements including setup)
        "settle_vertices": [],        # list of vertex ids settled at
        "settle_on_port": 0,          # count of own settlements on port vertices
        # Robber actions taken
        "robber_moves": 0,            # how many MoveRobber actions this role took
        "robber_targeted_seats": [0, 0, 0, 0],   # who got blocked
        # Endturns
        "n_endturn": 0,
        "first_city_endturn": -1,     # endturn count when first city was built; -1 if never
    } for r in seating}

    n_endturn_global = 0

    for i, a_raw in enumerate(action_history):
        a = int(a_raw)
        if a & CHANCE_BIT:
            outcome = a & ~CHANCE_BIT
            try:
                eng.apply_chance_outcome(outcome)
            except Exception:
                return None
            continue

        try:
            cp = int(eng.current_player())
            role = _role_for(cp)
        except Exception:
            return None

        cat = _categorize(a)

        # Build/play accounting (only after setup ends, i.e. i >= 16)
        if i >= 16:
            if cat == "settle":
                out[role]["n_settle_built"] += 1
            elif cat == "city":
                out[role]["n_city_built"] += 1
                if out[role]["first_city_endturn"] < 0:
                    out[role]["first_city_endturn"] = n_endturn_global
            elif cat == "road":
                out[role]["n_road_built"] += 1
            elif cat == "trade_bank":
                out[role]["n_trade_bank"] += 1
                give, get = _decode_trade(a, 206)
                # TradeBank: 4-for-1 (or port discount), simplified as +1/-1 for tally
                out[role][f"{RES_NAMES[give]}_given"] += 1
                out[role][f"{RES_NAMES[get]}_received"] += 1
            elif cat == "propose_trade":
                out[role]["n_propose_trade"] += 1
                give, get = _decode_trade(a, 260)
                # Note: we tally "intended" give/receive here; success check
                # happens in the post-step block (below) using before/after
                # hand comparison.
                out[role][f"{RES_NAMES[give]}_given"] += 1
                out[role][f"{RES_NAMES[get]}_received"] += 1
            elif cat == "buy_dev":
                out[role]["n_buy_dev"] += 1
            elif cat == "play_knight":
                out[role]["n_play_knight"] += 1
            elif cat == "play_rb":
                out[role]["n_play_rb"] += 1
            elif cat == "play_mono":
                out[role]["n_play_mono"] += 1
            elif cat == "play_yop":
                out[role]["n_play_yop"] += 1
            elif cat == "robber":
                out[role]["robber_moves"] += 1
                # The action_id - 180 is the target hex. Identify which players
                # have settlements on that hex (after engine applies). We'll
                # check post-step.
                _robber_pending = (cp, a - 180)

        # Setup-phase settlements: track placement vertex (any phase actually)
        if cat == "settle":
            out[role]["settle_vertices"].append(a)
            if vertex_has_port[a]:
                out[role]["settle_on_port"] += 1

        if cat == "endturn":
            out[role]["n_endturn"] += 1
            n_endturn_global += 1

        # For ProposeTrade success check: take "before" before stepping
        _trade_pending = None
        _robber_pending = None
        if i >= 16 and cat == "propose_trade":
            try:
                obs_b = eng.observation_for(cp)
                _trade_pending = (cp, _decode_trade(a, 260)[0], int(obs_b["scalars"][_decode_trade(a, 260)[0]]))
            except Exception:
                _trade_pending = None
        if cat == "robber":
            _robber_pending = (cp, a - 180)

        # Apply action
        try:
            eng.step(a)
        except Exception:
            return None

        # Post-step checks
        if _trade_pending is not None:
            try:
                cp_t, give_t, hand_before = _trade_pending
                obs_after = eng.observation_for(cp_t)
                hand_after = int(obs_after["scalars"][give_t])
                if hand_after < hand_before:  # gave away the resource → trade succeeded
                    out[role]["n_propose_trade_succeeded"] += 1
            except Exception:
                pass
        if _robber_pending is not None:
            try:
                _, target_hex = _robber_pending
                # Identify which players (seats) have settlements on that hex
                # Walk vertex_features. Find vertices adjacent to target_hex,
                # check ownership.
                obs_p = eng.observation_for(0)
                vf = obs_p["vertex_features"]
                from catan_gnn.adjacency import HEX_TO_VERTICES
                for v in HEX_TO_VERTICES[target_hex]:
                    is_settle = float(vf[v, 1]) >= 0.5
                    is_city = float(vf[v, 2]) >= 0.5
                    if not (is_settle or is_city):
                        continue
                    for p in range(4):
                        if float(vf[v, 3 + p]) >= 0.5:
                            # In viewer=0 perspective, col 3+p = player p
                            out[role]["robber_targeted_seats"][p] += 1
                            break
            except Exception:
                pass

    # Read final state per player
    for seat in range(4):
        role = _role_for(seat)
        try:
            obs_p = eng.observation_for(seat)
            scalars = obs_p["scalars"]
            # Per the observation layout, when viewer=seat, scalars[X+0] is THIS viewer.
            lr_len = float(scalars[SCALAR_LR_LEN]) * 15.0  # de-normalize (cited MAX_ROADS=15)
            knights = float(scalars[SCALAR_KNIGHTS]) * 14.0  # de-normalize (cited KNIGHT_DECK_TOTAL=14)
            lr_holder = float(scalars[SCALAR_LR_HOLDER]) >= 0.5
            la_holder = float(scalars[SCALAR_LA_HOLDER]) >= 0.5
            vp_total = int(scalars[SCALAR_VP])  # raw 0..10
            out[role]["lr_length"] = lr_len
            out[role]["knights_played"] = knights
            out[role]["lr_holder"] = bool(lr_holder)
            out[role]["la_holder"] = bool(la_holder)
            out[role]["final_vp"] = vp_total
            out[role]["seat"] = seat
        except Exception:
            pass

    out["_winner_seat"] = winner_seat
    out["_winner_role"] = _role_for(winner_seat) if winner_seat >= 0 else None
    out["_game_length"] = len(action_history)
    out["_n_endturn_total"] = n_endturn_global
    return out


def aggregate(per_game_list: list[dict], seating: list[str], labels: dict[str, str]):
    """Aggregate per-game results across all games, stratified by W/L."""
    buckets = ["overall", "in_wins", "in_losses"]
    summable = [
        "n_settle_built", "n_city_built", "n_road_built",
        "n_trade_bank", "n_propose_trade", "n_propose_trade_succeeded",
        "ore_given", "ore_received", "wheat_given", "wheat_received",
        "wood_given", "wood_received", "brick_given", "brick_received",
        "sheep_given", "sheep_received",
        "n_buy_dev", "n_play_knight", "n_play_rb", "n_play_mono", "n_play_yop",
        "settle_on_port", "robber_moves", "n_endturn",
    ]
    listable = ["lr_length", "knights_played", "final_vp", "first_city_endturn"]
    boolable = ["lr_holder", "la_holder"]

    agg = {b: {r: defaultdict(int) for r in seating} for b in buckets}
    agg_lists = {b: {r: defaultdict(list) for r in seating} for b in buckets}
    games_per_bucket = {b: {r: 0 for r in seating} for b in buckets}

    for g in per_game_list:
        if g is None:
            continue
        winner_role = g.get("_winner_role")
        for role in seating:
            roled = g[role]
            for b in ["overall"]:
                games_per_bucket[b][role] += 1
                for k in summable:
                    agg[b][role][k] += roled.get(k, 0)
                for k in listable:
                    if k in roled:
                        agg_lists[b][role][k].append(roled[k])
                for k in boolable:
                    if roled.get(k, False):
                        agg[b][role][k] += 1
                # Robber targeting: who did this role target most?
                if "robber_targeted_seats" in roled:
                    for tgt_seat, n in enumerate(roled["robber_targeted_seats"]):
                        agg[b][role][f"robber_target_seat_{tgt_seat}"] += n
            bucket = "in_wins" if winner_role == role else "in_losses"
            games_per_bucket[bucket][role] += 1
            for k in summable:
                agg[bucket][role][k] += roled.get(k, 0)
            for k in listable:
                if k in roled:
                    agg_lists[bucket][role][k].append(roled[k])
            for k in boolable:
                if roled.get(k, False):
                    agg[bucket][role][k] += 1

    return agg, agg_lists, games_per_bucket


def main():
    cfg = json.loads((TR / "worker0" / "config.json").read_text())
    seating = cfg["seating"]
    labels = {
        "PureGnnA": cfg["label_a"],
        "PureGnnB": cfg["label_b"],
        "PureGnnC": cfg["label_c"],
        "PureGnnD": cfg["label_d"],
    }
    short = {
        cfg["label_a"]: "Cell0-vanilla",
        cfg["label_b"]: "Cell1-cand8+10",
        cfg["label_c"]: "Cell5v2-cand11",
        cfg["label_d"]: "Cell6-stack",
    }

    rows = []
    for w in sorted(TR.glob("worker*")):
        for parq in sorted(w.glob("games.rot=*.parquet")):
            rot = int(parq.name.split(".")[1].split("=")[1])
            df = pq.read_table(str(parq)).to_pandas()
            df["rot"] = rot
            rows.append(df)
    g = pd.concat(rows, ignore_index=True)
    n_total = len(g)
    print(f"Loaded {n_total} games")
    print(f"Seating: {seating}")
    print(f"Labels: {labels}")
    print()

    per_game_list = []
    failures = 0
    for i, (_, row) in enumerate(g.iterrows()):
        result = analyze_game(
            int(row["seed"]), row["action_history"], seating,
            int(row["rot"]), int(row["winner"]), row["final_vp"],
        )
        if result is None:
            failures += 1
        else:
            per_game_list.append(result)
        if (i + 1) % 100 == 0:
            print(f"  progress: {i+1}/{n_total} (failures: {failures})")

    print(f"  done. {len(per_game_list)}/{n_total} games analyzed; {failures} replay failures.")
    print()

    agg, agg_lists, games_per_bucket = aggregate(per_game_list, seating, labels)

    # ------------ Render ------------
    short_labels = [(role, short[labels[role]]) for role in seating]
    lblw = 16

    # ============ 1. BUILD DYNAMICS ============
    print(f"\n{'='*30} 1. BUILD DYNAMICS (per game, post-setup) {'='*30}")
    for bucket in ["overall", "in_wins", "in_losses"]:
        print(f"\n--- {bucket} ---")
        print(f"  {'cell':<{lblw}s} {'games':>6s} {'roads/g':>8s} {'settle/g':>9s} {'cities/g':>9s} {'roads/settle':>13s} {'dev/g':>7s} {'knights/g':>10s}")
        for role, lbl in short_labels:
            n_games = games_per_bucket[bucket][role]
            if n_games == 0:
                continue
            r = agg[bucket][role].get("n_road_built", 0) / n_games
            s = agg[bucket][role].get("n_settle_built", 0) / n_games
            c = agg[bucket][role].get("n_city_built", 0) / n_games
            d = agg[bucket][role].get("n_buy_dev", 0) / n_games
            k = agg[bucket][role].get("n_play_knight", 0) / n_games
            rs_total_settle = max(agg[bucket][role].get("n_settle_built", 0), 1)
            r_per_s = agg[bucket][role].get("n_road_built", 0) / rs_total_settle
            print(f"  {lbl:<{lblw}s} {n_games:>6d} {r:>8.2f} {s:>9.2f} {c:>9.2f} {r_per_s:>12.2f} {d:>7.2f} {k:>9.2f}")

    # ============ 2. BONUS ECONOMY ============
    print(f"\n{'='*30} 2. BONUS ECONOMY (final-state per game) {'='*30}")
    for bucket in ["overall", "in_wins", "in_losses"]:
        print(f"\n--- {bucket} ---")
        print(f"  {'cell':<{lblw}s} {'games':>6s} {'mean_LR':>8s} {'mean_knights':>13s} {'%LR_held':>9s} {'%LA_held':>9s} {'mean_VP':>8s}")
        for role, lbl in short_labels:
            n_games = games_per_bucket[bucket][role]
            if n_games == 0:
                continue
            lr_list = agg_lists[bucket][role].get("lr_length", [])
            k_list = agg_lists[bucket][role].get("knights_played", [])
            vp_list = agg_lists[bucket][role].get("final_vp", [])
            mean_lr = np.mean(lr_list) if lr_list else 0
            mean_k = np.mean(k_list) if k_list else 0
            mean_vp = np.mean(vp_list) if vp_list else 0
            pct_lr = 100 * agg[bucket][role].get("lr_holder", 0) / n_games
            pct_la = 100 * agg[bucket][role].get("la_holder", 0) / n_games
            print(f"  {lbl:<{lblw}s} {n_games:>6d} {mean_lr:>8.2f} {mean_k:>13.2f} {pct_lr:>8.1f}% {pct_la:>8.1f}% {mean_vp:>8.2f}")

    # WINNERS ONLY: VP composition
    print(f"\n--- Winners only: VP composition (mean across wins) ---")
    print(f"  {'cell':<{lblw}s} {'wins':>5s} {'LR_held%':>9s} {'LA_held%':>9s} {'cities_in_wins':>15s} {'settles_in_wins':>16s}")
    for role, lbl in short_labels:
        n_wins = games_per_bucket["in_wins"][role]
        if n_wins == 0:
            print(f"  {lbl:<{lblw}s} {0:>5d}  (no wins)")
            continue
        pct_lr = 100 * agg["in_wins"][role].get("lr_holder", 0) / n_wins
        pct_la = 100 * agg["in_wins"][role].get("la_holder", 0) / n_wins
        cities_per_win = agg["in_wins"][role].get("n_city_built", 0) / n_wins
        settles_per_win = agg["in_wins"][role].get("n_settle_built", 0) / n_wins
        print(f"  {lbl:<{lblw}s} {n_wins:>5d} {pct_lr:>8.1f}% {pct_la:>8.1f}% {cities_per_win:>14.2f} {settles_per_win:>15.2f}")

    # ============ 3. TRADE DYNAMICS ============
    print(f"\n{'='*30} 3. TRADE DYNAMICS {'='*30}")
    for bucket in ["overall", "in_wins", "in_losses"]:
        print(f"\n--- {bucket} ---")
        print(f"  {'cell':<{lblw}s} {'games':>6s} {'bank/g':>7s} {'propose/g':>10s} {'success%':>9s} {'ore_net/g':>10s} {'wheat_net/g':>12s} {'sheep_net/g':>12s}")
        for role, lbl in short_labels:
            n_games = games_per_bucket[bucket][role]
            if n_games == 0:
                continue
            bank = agg[bucket][role].get("n_trade_bank", 0) / n_games
            prop = agg[bucket][role].get("n_propose_trade", 0) / n_games
            prop_n = agg[bucket][role].get("n_propose_trade", 0)
            prop_s = agg[bucket][role].get("n_propose_trade_succeeded", 0)
            sr = 100 * prop_s / prop_n if prop_n else 0
            ore_net = (agg[bucket][role].get("ore_received", 0) - agg[bucket][role].get("ore_given", 0)) / n_games
            wheat_net = (agg[bucket][role].get("wheat_received", 0) - agg[bucket][role].get("wheat_given", 0)) / n_games
            sheep_net = (agg[bucket][role].get("sheep_received", 0) - agg[bucket][role].get("sheep_given", 0)) / n_games
            print(f"  {lbl:<{lblw}s} {n_games:>6d} {bank:>7.2f} {prop:>10.2f} {sr:>8.1f}% {ore_net:>+9.2f} {wheat_net:>+11.2f} {sheep_net:>+11.2f}")

    # ============ 4. CLOSEOUT ============
    print(f"\n{'='*30} 4. CLOSEOUT / GAME LENGTH {'='*30}")
    print(f"\n--- All games: game length distribution per winning cell ---")
    print(f"  {'cell':<{lblw}s} {'wins':>5s} {'min':>5s} {'p25':>6s} {'median':>7s} {'p75':>6s} {'max':>6s}")
    for role, lbl in short_labels:
        wins = games_per_bucket["in_wins"][role]
        if wins == 0:
            print(f"  {lbl:<{lblw}s} {0:>5d}  (no wins)")
            continue
        # game length isn't in our per-role dict; compute it from per_game_list
        lens = [g["_game_length"] for g in per_game_list if g.get("_winner_role") == role]
        if not lens:
            continue
        ls = sorted(lens)
        print(f"  {lbl:<{lblw}s} {len(lens):>5d} {ls[0]:>5d} {ls[len(ls)//4]:>6d} {ls[len(ls)//2]:>7d} {ls[3*len(ls)//4]:>6d} {ls[-1]:>6d}")

    print(f"\n--- Time-to-first-city (in EndTurn ticks) ---")
    print(f"  {'cell':<{lblw}s} {'games_w/city':>13s} {'mean_endturn_at_first_city':>28s}")
    for role, lbl in short_labels:
        # in_wins bucket: this cell DID win at least 10 VP, so should have built cities
        firsts = agg_lists["overall"][role].get("first_city_endturn", [])
        with_city = [x for x in firsts if x >= 0]
        if not with_city:
            print(f"  {lbl:<{lblw}s} {0:>13d}  (no cities)")
            continue
        print(f"  {lbl:<{lblw}s} {len(with_city):>13d} {np.mean(with_city):>27.1f}")

    # ============ 5. PORT USAGE ============
    print(f"\n{'='*30} 5. PORT USAGE {'='*30}")
    for bucket in ["overall", "in_wins", "in_losses"]:
        print(f"\n--- {bucket} ---")
        print(f"  {'cell':<{lblw}s} {'games':>6s} {'settle_total':>13s} {'settle_on_port':>15s} {'%on_port':>9s}")
        for role, lbl in short_labels:
            n_games = games_per_bucket[bucket][role]
            if n_games == 0:
                continue
            total_settle = 0
            on_port = agg[bucket][role].get("settle_on_port", 0)
            # total settle (setup + main): use settle_vertices length per game
            for g in per_game_list:
                if bucket == "overall":
                    total_settle += len(g[role]["settle_vertices"])
                elif bucket == "in_wins" and g.get("_winner_role") == role:
                    total_settle += len(g[role]["settle_vertices"])
                elif bucket == "in_losses" and g.get("_winner_role") != role:
                    total_settle += len(g[role]["settle_vertices"])
            pct = 100 * on_port / max(total_settle, 1)
            print(f"  {lbl:<{lblw}s} {n_games:>6d} {total_settle:>13d} {on_port:>15d} {pct:>8.1f}%")

    # ============ 6. RESOURCE SPECIALIZATION ============
    print(f"\n{'='*30} 6. RESOURCE SPECIALIZATION (settlement-vertex hex types) {'='*30}")
    # Board is ABC balanced (fixed) per the v3 spec, so hex_resource is
    # identical across all games. Read it once from a fresh Engine.
    eng_static = _engine.Engine.with_rules(0, 10, True)
    obs_static = eng_static.observation_for(0)
    hex_feat_s = obs_static["hex_features"]
    hex_resource_static = []
    for h in range(19):
        if float(hex_feat_s[h, 7]) >= 0.5:
            hex_resource_static.append(-1)  # desert
        else:
            for r in range(5):
                if float(hex_feat_s[h, r]) >= 0.5:
                    hex_resource_static.append(r)
                    break
            else:
                hex_resource_static.append(-1)

    print(f"  {'cell':<{lblw}s} {'n_settlements':>14s} {'wood%':>7s} {'brick%':>7s} {'sheep%':>7s} {'wheat%':>7s} {'ore%':>6s} {'desert%':>8s}")
    for role, lbl in short_labels:
        res_counts = [0, 0, 0, 0, 0]
        desert_count = 0
        total_hex_touches = 0
        for g in per_game_list:
            for v in g[role]["settle_vertices"]:
                for h in V2H[v]:
                    r = hex_resource_static[h]
                    total_hex_touches += 1
                    if r < 0:
                        desert_count += 1
                    else:
                        res_counts[r] += 1
        if total_hex_touches == 0:
            continue
        n_settles = sum(len(g[role]["settle_vertices"]) for g in per_game_list)
        pct = [100 * c / total_hex_touches for c in res_counts]
        pct_d = 100 * desert_count / total_hex_touches
        print(f"  {lbl:<{lblw}s} {n_settles:>14d} {pct[0]:>6.1f}% {pct[1]:>6.1f}% {pct[2]:>6.1f}% {pct[3]:>6.1f}% {pct[4]:>5.1f}% {pct_d:>7.1f}%")

    # ============ 7. ROBBER TARGETING ============
    print(f"\n{'='*30} 7. ROBBER TARGETING (when this cell moved robber) {'='*30}")
    print(f"  {'cell':<{lblw}s} {'moves':>6s} {'tgt_seat_0':>11s} {'tgt_seat_1':>11s} {'tgt_seat_2':>11s} {'tgt_seat_3':>11s}")
    print(f"  (seat 0 = self in rot=0 seating order; targets are summed across rotations)")
    for role, lbl in short_labels:
        moves = agg["overall"][role].get("robber_moves", 0)
        s0 = agg["overall"][role].get("robber_target_seat_0", 0)
        s1 = agg["overall"][role].get("robber_target_seat_1", 0)
        s2 = agg["overall"][role].get("robber_target_seat_2", 0)
        s3 = agg["overall"][role].get("robber_target_seat_3", 0)
        print(f"  {lbl:<{lblw}s} {moves:>6d} {s0:>11d} {s1:>11d} {s2:>11d} {s3:>11d}")


if __name__ == "__main__":
    sys.exit(main() or 0)
