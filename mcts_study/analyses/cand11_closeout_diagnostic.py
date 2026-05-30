"""Cand 11 city-closeout diagnostic.

Walks the 1200-game e10c head-to-head parquets to figure out which of
three hypotheses explains Cand 11's ending-pip → win conversion gap
(37.9% Cand 11 vs 94.8% LookV3 when role had strict highest ending pip):

  Problem 1: Cand 11 places settlements at vertices that can't easily
             become cities (no ore + wheat adjacent).
  Problem 2: Cand 11 has the right settlements but trades away ore/wheat
             before upgrading.
  Problem 3: Cand 11 has the right settlements AND the resources but
             plays other actions first (knight, dev card, etc.) instead
             of building the city.

For each role we compute, broken out by overall / in-wins / in-losses:

  - settlements_city_ready: of own settlements placed at any point in
    the game, what fraction have BOTH ore and wheat as adjacent hex
    resources? (Problem 1 measure)
  - mean_turns_holding_city_resources: across all turns where the role
    is current_player AND has >= 3 ore + >= 2 wheat in hand AND owns
    >= 1 settlement, count turns spent before next BuildCity action.
    (Problem 3 measure)
  - net_ore_traded_away: sum over the role's TradeBank + ProposeTrade
    actions of (ore_given - ore_received), same for wheat.
    (Problem 2 measure)

The settlements_city_ready metric only depends on the board (static)
and the role's placement choice, so it's directly comparable across
roles. The hold-time and trade metrics need engine state at each step.

Output: a side-by-side table per role + per win/loss bucket.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from catan_bot import _engine

ROOT = Path("/mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study")
TR = ROOT / "runs/v3/e10c_4way_1200_2026_05_26/2026-05-26T15-59-e10c_triple_gnn"

CHANCE_BIT = 0x80000000

# Action ID ranges (cited)
SETTLE_RANGE = (0, 54)
CITY_RANGE = (54, 108)
TRADE_BANK_RANGE = (206, 226)
PROPOSE_TRADE_RANGE = (260, 280)

# Resource index in scalars (cited observation.rs:31)
#   scalars[0..5) = viewer hand counts: wood, brick, sheep, wheat, ore
# Per RESOURCE enum order: 0=Wood, 1=Brick, 2=Sheep, 3=Wheat, 4=Ore.
RES_WOOD, RES_BRICK, RES_SHEEP, RES_WHEAT, RES_ORE = 0, 1, 2, 3, 4

# Cited Catan city cost: 3 ore + 2 wheat
CITY_COST_ORE = 3
CITY_COST_WHEAT = 2

# TradeBank layout: 20 actions = 5 give x 4 get (cited playback.py:271-278)
# action_id = 206 + give_res * 4 + get_idx, where get_idx skips the give resource
# We just need give_res and get_res for each trade action_id.
def _decode_trade_bank(a: int) -> tuple[int, int]:
    """Returns (give_res, get_res) for a TradeBank action."""
    idx = a - 206
    give = idx // 4
    get_in_remaining = idx % 4
    # Get resource is the get_in_remaining-th resource SKIPPING give
    others = [r for r in range(5) if r != give]
    get = others[get_in_remaining]
    return give, get


def _decode_propose_trade(a: int) -> tuple[int, int]:
    """Returns (give_res, get_res) for a ProposeTrade action.
    Same 5x4 layout starting at 260."""
    idx = a - 260
    give = idx // 4
    get_in_remaining = idx % 4
    others = [r for r in range(5) if r != give]
    get = others[get_in_remaining]
    return give, get


def _resource_of_hex(hex_features_row) -> int:
    """Returns resource index (0..4) or -1 if desert.
    hex_features[h, 0..5] is the resource one-hot per observation.rs.
    """
    for r in range(5):
        if float(hex_features_row[r]) >= 0.5:
            return r
    return -1


def _build_vertex_to_hexes() -> dict[int, list[int]]:
    from catan_gnn.adjacency import HEX_TO_VERTICES
    v2h: dict[int, list[int]] = {v: [] for v in range(54)}
    for h, vs in enumerate(HEX_TO_VERTICES):
        for v in vs:
            v2h[int(v)].append(h)
    return v2h


def analyze_game(seed: int, action_history, seating: list[str], rot: int,
                 winner_seat: int):
    """Walk a single game, accumulate per-role metrics.

    Returns dict[role][metric] -> incremental counts; caller sums across games.

    Per-role metrics collected:
      - n_settlements_placed
      - n_settlements_city_ready (with ore AND wheat adjacent)
      - n_settlements_partial_ready (with ore XOR wheat)
      - n_settlements_dry (neither)
      - ore_given, ore_received, wheat_given, wheat_received
      - turns_with_city_resources (turns where role was current_player
        AND had >= 3 ore + 2 wheat AND owned >= 1 settlement)
      - cities_built (BuildCity actions this role took)
    """
    v2h = _build_vertex_to_hexes()
    eng = _engine.Engine(int(seed))
    obs0 = eng.observation_for(0)
    hex_feat = obs0["hex_features"]
    # Per-hex resource index
    hex_res = [_resource_of_hex(hex_feat[h]) for h in range(19)]

    # Per-vertex: does it touch ore? wheat?
    v_has_ore = [any(hex_res[h] == RES_ORE for h in v2h[v]) for v in range(54)]
    v_has_wheat = [any(hex_res[h] == RES_WHEAT for h in v2h[v]) for v in range(54)]

    def _role_for(seat: int) -> str:
        return seating[(seat + rot) % 4]

    # Per-role counters
    metrics = {r: {
        "n_settlements_placed": 0,
        "n_settlements_city_ready": 0,
        "n_settlements_partial_ready": 0,
        "n_settlements_dry": 0,
        "ore_given": 0, "ore_received": 0,
        "wheat_given": 0, "wheat_received": 0,
        "turns_with_city_resources": 0,
        "cities_built": 0,
        "n_action_steps": 0,
    } for r in seating}

    # Walk the game step-by-step.
    # We need:
    #   - on BuildSettlement: which role + which vertex (count city-readiness)
    #   - on BuildCity: which role (count cities_built)
    #   - on TradeBank: which role + give/get resources
    #   - on ProposeTrade: which role + give/get (we count proposer's intent;
    #       engine resolves; we don't easily see if it succeeded without before/after hand state)
    #   - at start of each player's turn-action segment: their hand state
    #
    # For the "turns_with_city_resources" metric, we sample the engine's hand
    # state at every non-chance action_history step where the action belongs
    # to the current player. The engine's `observation_for(viewer)` puts the
    # viewer's hand in scalars[0..5]. So we call observation_for(current_player)
    # at each step and check ore + wheat.

    for i, a_raw in enumerate(action_history):
        a = int(a_raw)
        if a & CHANCE_BIT:
            outcome = a & ~CHANCE_BIT
            try:
                eng.apply_chance_outcome(outcome)
            except Exception:
                return None
            continue

        # Pre-action state
        try:
            cp = int(eng.current_player())
            role = _role_for(cp)
        except Exception:
            return None
        metrics[role]["n_action_steps"] += 1

        # Hand state check for "turns_with_city_resources"
        # Only count once per turn — use the player's FIRST action of their turn
        # as the sample point. Detect "first action of turn" by checking that
        # the previous non-chance action was EndTurn (or this is the start).
        # Simpler heuristic: just check at every action step where the player
        # owns a settlement; divide by EndTurn count to get per-turn average.
        # We'll just count action-step samples here and normalize later.
        try:
            obs = eng.observation_for(cp)
            scalars = obs["scalars"]
            hand_ore = int(scalars[RES_ORE])
            hand_wheat = int(scalars[RES_WHEAT])
            # Owns a settlement?
            vf = obs["vertex_features"]
            # In viewer=cp perspective, col 3 is viewer (cp) himself.
            owns_settle = False
            for v in range(54):
                if float(vf[v, 1]) >= 0.5 and float(vf[v, 3]) >= 0.5:
                    owns_settle = True
                    break
            if (hand_ore >= CITY_COST_ORE and hand_wheat >= CITY_COST_WHEAT
                    and owns_settle):
                metrics[role]["turns_with_city_resources"] += 1
        except Exception:
            pass

        # Categorize action
        if SETTLE_RANGE[0] <= a < SETTLE_RANGE[1]:
            v = a
            metrics[role]["n_settlements_placed"] += 1
            ore = v_has_ore[v]
            wht = v_has_wheat[v]
            if ore and wht:
                metrics[role]["n_settlements_city_ready"] += 1
            elif ore or wht:
                metrics[role]["n_settlements_partial_ready"] += 1
            else:
                metrics[role]["n_settlements_dry"] += 1
        elif CITY_RANGE[0] <= a < CITY_RANGE[1]:
            metrics[role]["cities_built"] += 1
        elif TRADE_BANK_RANGE[0] <= a < TRADE_BANK_RANGE[1]:
            give, get = _decode_trade_bank(a)
            if give == RES_ORE: metrics[role]["ore_given"] += 4   # 4:1 (or port; we approximate as 4)
            if give == RES_WHEAT: metrics[role]["wheat_given"] += 4
            if get == RES_ORE: metrics[role]["ore_received"] += 1
            if get == RES_WHEAT: metrics[role]["wheat_received"] += 1
        elif PROPOSE_TRADE_RANGE[0] <= a < PROPOSE_TRADE_RANGE[1]:
            give, get = _decode_propose_trade(a)
            # 1-for-1 player trade (cited rules.rs:181 simplified). Engine
            # auto-resolves; we count it as a *proposed* trade. Whether it
            # succeeded would require before/after hand state comparison.
            # For our diagnostic we use this as "intended trade flow."
            if give == RES_ORE: metrics[role]["ore_given"] += 1
            if give == RES_WHEAT: metrics[role]["wheat_given"] += 1
            if get == RES_ORE: metrics[role]["ore_received"] += 1
            if get == RES_WHEAT: metrics[role]["wheat_received"] += 1

        # Apply action
        try:
            eng.step(a)
        except Exception:
            return None

    metrics["_winner_role"] = _role_for(winner_seat) if winner_seat >= 0 else None
    return metrics


def main():
    cfg = json.loads((TR / "worker0" / "config.json").read_text())
    seating = cfg["seating"]
    label_map = {
        "PureGnnA": cfg["label_a"],
        "PureGnnB": cfg["label_b"],
        "PureGnnC": cfg["label_c"],
        "LookaheadMctsV3": "LookaheadMctsV3",
    }

    # Load all games
    rows = []
    for w in sorted(TR.glob("worker*")):
        for parq in sorted(w.glob("games.rot=*.parquet")):
            rot = int(parq.name.split(".")[1].split("=")[1])
            df = pq.read_table(str(parq)).to_pandas()
            df["rot"] = rot
            rows.append(df)
    g = pd.concat(rows, ignore_index=True)
    print(f"Loaded {len(g)} games")
    print(f"Seating: {seating}")
    print()

    # Accumulate per-role per-bucket (overall / win / loss)
    buckets = ["overall", "in_wins", "in_losses"]
    agg = {b: {r: {} for r in seating} for b in buckets}

    failures = 0
    for i, (_, row) in enumerate(g.iterrows()):
        rot = int(row["rot"])
        winner_seat = int(row["winner"])
        per_game = analyze_game(int(row["seed"]), row["action_history"],
                                seating, rot, winner_seat)
        if per_game is None:
            failures += 1
            continue
        winner_role = per_game.get("_winner_role")
        for role in seating:
            for k, v in per_game[role].items():
                for b in ["overall"]:
                    agg[b][role][k] = agg[b][role].get(k, 0) + v
                if winner_role == role:
                    agg["in_wins"][role][k] = agg["in_wins"][role].get(k, 0) + v
                else:
                    agg["in_losses"][role][k] = agg["in_losses"][role].get(k, 0) + v
        if (i + 1) % 200 == 0:
            print(f"  progress: {i+1}/{len(g)} (failures: {failures})")

    print(f"  done: {len(g)} games, {failures} replay failures")
    print()

    # Render per bucket
    for b in buckets:
        print(f"=== {b.upper()} ===")
        print(f"  {'role':<26s} {'n_settle':>9s} {'%ready':>8s} {'%partial':>9s} {'%dry':>7s} "
              f"{'cities':>7s} {'turns_w/cityres':>16s} {'ore_net':>8s} {'wheat_net':>10s}")
        for role in seating:
            m = agg[b][role]
            n_settle = m.get("n_settlements_placed", 0)
            ready = m.get("n_settlements_city_ready", 0)
            partial = m.get("n_settlements_partial_ready", 0)
            dry = m.get("n_settlements_dry", 0)
            cities = m.get("cities_built", 0)
            twcr = m.get("turns_with_city_resources", 0)
            ore_net = m.get("ore_received", 0) - m.get("ore_given", 0)
            wheat_net = m.get("wheat_received", 0) - m.get("wheat_given", 0)
            pct = lambda x: (100 * x / n_settle) if n_settle else 0
            lbl = label_map[role]
            print(f"  {lbl:<26s} {n_settle:>9d} {pct(ready):>7.1f}% {pct(partial):>8.1f}% "
                  f"{pct(dry):>6.1f}% {cities:>7d} {twcr:>16d} {ore_net:>+8d} {wheat_net:>+10d}")
        print()

    print("=== Derived: closeout efficiency ===")
    print(f"  cities_per_turn_with_city_resources (cities built / turns held resources)")
    print(f"  Higher = closes out faster when ready.")
    print(f"  {'role':<26s} {'overall':>10s} {'in_wins':>10s} {'in_losses':>10s}")
    for role in seating:
        line = f"  {label_map[role]:<26s}"
        for b in buckets:
            cities = agg[b][role].get("cities_built", 0)
            twcr = agg[b][role].get("turns_with_city_resources", 0)
            ratio = (cities / twcr) if twcr else 0
            line += f" {ratio:>9.3f} "
        print(line)


if __name__ == "__main__":
    sys.exit(main() or 0)
