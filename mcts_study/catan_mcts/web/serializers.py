"""Per-state engine serializer + action description, shared by the live web
server and the offline replay viewer (catan_mcts.playback).

Moved verbatim out of catan_mcts.playback during the Phase-1 extraction.
"""
from __future__ import annotations

import numpy as np


# ===================== action description (v2: 280 actions) =====================

def action_desc(a: int) -> str:
    a = int(a)
    if 0 <= a < 54:    return f"BuildSettlement(v={a})"
    if 54 <= a < 108:  return f"BuildCity(v={a - 54})"
    if 108 <= a < 180: return f"BuildRoad(e={a - 108})"
    if 180 <= a < 199: return f"MoveRobber(h={a - 180})"
    if 199 <= a < 204:
        names = ["Wood", "Brick", "Sheep", "Wheat", "Ore"]
        return f"Discard({names[a - 199]})"
    if a == 204: return "EndTurn"
    if a == 205: return "RollDice"
    # ----- v2 additions -----
    if 206 <= a < 226:
        # TradeBank: 5 give-resource × 4 valid get-resources (skipping the give itself)
        idx = a - 206
        give = idx // 4
        get_idx_in_others = idx % 4
        others = [r for r in range(5) if r != give]
        get = others[get_idx_in_others]
        names = ["Wd", "Bk", "Sh", "Wh", "Or"]
        return f"TradeBank({names[give]}→{names[get]})"
    if a == 226: return "BuyDevCard"
    if a == 227: return "PlayKnight"
    if a == 228: return "PlayRoadBuilding"
    if 229 <= a < 234:
        names = ["Wood", "Brick", "Sheep", "Wheat", "Ore"]
        return f"PlayMonopoly({names[a - 229]})"
    if 234 <= a < 259:
        idx = a - 234
        r1 = idx // 5
        r2 = idx % 5
        names = ["Wd", "Bk", "Sh", "Wh", "Or"]
        return f"PlayYearOfPlenty({names[r1]}+{names[r2]})"
    if a == 259: return "PlayVpCard"
    if 260 <= a < 280:
        idx = a - 260
        give = idx // 4
        get_idx_in_others = idx % 4
        others = [r for r in range(5) if r != give]
        get = others[get_idx_in_others]
        names = ["Wd", "Bk", "Sh", "Wh", "Or"]
        return f"ProposeTrade({names[give]}↔{names[get]})"
    return f"<unknown:{a}>"


# Scalar layout (matches catan_engine::observation::SCALAR_*) — for parsing per-step state.
SCALAR_HAND = 0
SCALAR_OPP_HAND_SIZES = 5
SCALAR_VP = 8
SCALAR_TURN = 12
SCALAR_PHASE = 13
SCALAR_DEV_HELD = 21
SCALAR_LR_LEN = 26
SCALAR_KNIGHTS = 30
SCALAR_SETTL_BUILT = 34
SCALAR_CITY_BUILT = 38
SCALAR_ROAD_BUILT = 42
SCALAR_PORTS = 46
SCALAR_LR_HOLDER = 52
SCALAR_LA_HOLDER = 53
SCALAR_BANK = 54

DEV_CARD_NAMES = ["Knight", "RoadBldg", "Mono", "YOP", "VP"]
PORT_NAMES = ["3:1", "Wd 2:1", "Bk 2:1", "Sh 2:1", "Wh 2:1", "Or 2:1"]
PHASE_NAMES = ["Setup1", "Setup2", "Roll", "Main", "Discard", "MoveRobber", "Steal", "Done"]

MAX_SETTLEMENTS = 5
MAX_CITIES = 4
MAX_ROADS = 15


def serialize_state(eng, narration: str) -> dict:
    """Snapshot the engine into a JS-ready per-state dict.

    Each state dict has the keys: n (narration), cp (current player or -1 at
    terminal), s (settlements as [(v, owner), ...]), c (cities), r (roads),
    rh (robber hex), vp (live VPs), hands (per-player breakdown + total),
    bank (5-vec), dev_held (per-player [5]: knight, RB, mono, YOP, VP),
    lr_len (per-player), knights (per-player), built (per-player
    {settle, city, road}), ports (per-player [6]), lr_holder, la_holder,
    vp_played (per-player).
    """
    cp = -1 if eng.is_terminal() else int(eng.current_player())
    obs = eng.observation()
    vfeat = obs["vertex_features"]
    efeat = obs["edge_features"]
    hfeat = obs["hex_features"]
    # Use observation_for(0) to get an absolute (non-rotated) view —
    # observation() is rotated by current_player which makes decoding
    # owners ambiguous at terminal (cp=-1).
    obs_abs = eng.observation_for(0)
    scalars = obs_abs["scalars"]
    # all_hands() and bank() are absolute (no rotation).
    hands_arr = eng.all_hands()  # [4, 5] uint8
    bank = list(map(int, eng.bank()))

    # Decode buildings from absolute observation_for(0).
    vfeat_abs = obs_abs["vertex_features"]
    efeat_abs = obs_abs["edge_features"]
    settlements: list[tuple[int, int]] = []
    cities: list[tuple[int, int]] = []
    for v in range(54):
        f = vfeat_abs[v]
        if f[1] > 0.5 or f[2] > 0.5:
            # owner_persp at observation_for(0) is already absolute.
            owner_abs = int(np.argmax(f[3:7]))
            if f[2] > 0.5:
                cities.append((v, owner_abs))
            else:
                settlements.append((v, owner_abs))
    roads: list[tuple[int, int]] = []
    for e in range(72):
        f = efeat_abs[e]
        if f[1] > 0.5:
            owner_abs = int(np.argmax(f[2:6]))
            roads.append((e, owner_abs))
    # Robber hex from observation (perspective doesn't affect hex features).
    robber_hex = -1
    for h in range(19):
        if hfeat[h][6] > 0.5:
            robber_hex = h
            break
    # Live VPs from absolute scalars. With viewer=0, scalars[VP+i] is player i's VP.
    vps = [int(round(scalars[SCALAR_VP + i])) for i in range(4)]
    if eng.is_terminal():
        stats = eng.stats()
        vps = [int(stats["players"][p]["vp_final"]) for p in range(4)]

    # Per-player v2 fields (need 4 separate observation_for calls so each
    # player's dev_held / port flags / LR / LA holder is correct in their
    # own absolute frame). Hands/bank/built we already have.
    per_player: list[dict] = []
    for p in range(4):
        obs_p = eng.observation_for(p)
        sp = obs_p["scalars"]
        dev_held = [int(round(sp[SCALAR_DEV_HELD + k])) for k in range(5)]
        ports = [bool(round(sp[SCALAR_PORTS + i])) for i in range(6)]
        holds_lr = bool(round(sp[SCALAR_LR_HOLDER]))
        holds_la = bool(round(sp[SCALAR_LA_HOLDER]))
        per_player.append({
            "dev_held": dev_held,
            "ports": ports,
            "holds_lr": holds_lr,
            "holds_la": holds_la,
        })

    # LR length and knights_played and buildings_built — perspective-
    # rotated by viewer; we read the all-players block from viewer=0
    # where index i = absolute player i.
    scalars0 = obs_abs["scalars"]
    lr_len = [int(round(scalars0[SCALAR_LR_LEN + i] * MAX_ROADS)) for i in range(4)]
    knights = [int(round(scalars0[SCALAR_KNIGHTS + i] * 14)) for i in range(4)]
    settle_built = [int(round(scalars0[SCALAR_SETTL_BUILT + i] * MAX_SETTLEMENTS)) for i in range(4)]
    city_built = [int(round(scalars0[SCALAR_CITY_BUILT + i] * MAX_CITIES)) for i in range(4)]
    road_built = [int(round(scalars0[SCALAR_ROAD_BUILT + i] * MAX_ROADS)) for i in range(4)]

    lr_holder = next((p for p in range(4) if per_player[p]["holds_lr"]), -1)
    la_holder = next((p for p in range(4) if per_player[p]["holds_la"]), -1)

    # Played VP cards: derived from VP arithmetic since the engine doesn't
    # expose state.dev_cards_played[VP] through the observation. The
    # engine grants 1 VP for each settlement, 2 for each city (settlement
    # +1 absorbed into the city), +2 for the LR/LA bonus holders, and
    # +1 per VP card already drawn (auto-applied since "no hidden info").
    # So:  vp_card_count = vp - settlements - 2*cities - 2*lr - 2*la
    vp_played = []
    for p in range(4):
        base = settle_built[p] + 2 * city_built[p]
        if lr_holder == p:
            base += 2
        if la_holder == p:
            base += 2
        vp_played.append(max(0, vps[p] - base))

    # Phase: index 13..21 is one-hot.
    phase_idx = -1
    for k in range(8):
        if scalars0[SCALAR_PHASE + k] > 0.5:
            phase_idx = k
            break
    phase_name = PHASE_NAMES[phase_idx] if 0 <= phase_idx < 8 else "?"

    hands_breakdown = []
    for p in range(4):
        h = list(map(int, hands_arr[p]))
        hands_breakdown.append({"breakdown": h, "total": sum(h)})

    return {
        "n": narration, "cp": cp, "phase": phase_name,
        "s": settlements, "c": cities, "r": roads, "rh": robber_hex,
        "vp": vps, "hands": hands_breakdown, "bank": bank,
        "dev_held": [pp["dev_held"] for pp in per_player],
        "ports": [pp["ports"] for pp in per_player],
        "lr_len": lr_len, "knights": knights,
        "built": [{"settle": settle_built[p], "city": city_built[p], "road": road_built[p]} for p in range(4)],
        "lr_holder": lr_holder, "la_holder": la_holder, "vp_played": vp_played,
    }
