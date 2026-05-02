"""v2 self-contained replay viewer.

Replays a recorded game once through the v2 engine and emits a single
self-contained HTML file with a static board PNG, per-step SVG overlays,
and a side panel showing every player's hand, dev cards, longest road,
largest army, building inventory, and bank state — all read directly from
the engine (no card-tracker reconstruction needed in v2).

Input: a v2 parquet run_dir (containing games.v2cell.parquet +
       moves.v2cell.parquet shards) and a seed.

Output: <out_dir>/index.html — open with file:// (double-click).

Usage:
    python -m catan_mcts.playback <run_dir> <seed> [<out_dir>]
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from catan_bot import _engine

ROW_LENGTHS = [3, 4, 5, 4, 3]
HEX_RADIUS = 1.0

HEX_ROW_COL = {}
_hid = 0
for _r, _n in enumerate(ROW_LENGTHS):
    for _c in range(_n):
        HEX_ROW_COL[_hid] = (_r, _c)
        _hid += 1


def _hex_center_pointy(hex_id: int) -> tuple[float, float]:
    row, col = HEX_ROW_COL[hex_id]
    n_in_row = ROW_LENGTHS[row]
    spacing_x = math.sqrt(3) * HEX_RADIUS
    x_offset = -(n_in_row - 1) * spacing_x / 2
    x = x_offset + col * spacing_x
    y = -row * 1.5 * HEX_RADIUS
    return x, y


def _build_layout():
    from catan_gnn.adjacency import HEX_TO_VERTICES, EDGE_TO_VERTICES

    angles = [math.radians(a) for a in (90, 30, -30, -90, -150, 150)]
    corner_offsets = [(math.cos(a), math.sin(a)) for a in angles]

    vertex_xy = {}
    for hex_id, vert_ids in enumerate(HEX_TO_VERTICES):
        cx, cy = _hex_center_pointy(hex_id)
        for slot, vid in enumerate(vert_ids):
            ox, oy = corner_offsets[slot]
            vertex_xy[vid] = (cx + ox, cy + oy)

    edges = []
    for eid, (v1, v2) in enumerate(EDGE_TO_VERTICES):
        x1, y1 = vertex_xy[v1]
        x2, y2 = vertex_xy[v2]
        edges.append((x1, y1, x2, y2))

    hex_centers = [_hex_center_pointy(h) for h in range(19)]
    return vertex_xy, edges, hex_centers


# Plot bounds — these MUST match what the JS uses to map data->pixel.
XLIM = (-6.2, 6.2)
YLIM = (-8.0, 2.5)
FIG_WIDTH_INCHES = 10.0
FIG_HEIGHT_INCHES = (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0]) * FIG_WIDTH_INCHES
FIG_DPI = 100

RESOURCE_COLORS = {0: "#3d8b37", 1: "#a04020", 2: "#90c060", 3: "#e6c243", 4: "#7a7a7a"}
DESERT_COLOR = "#d4b483"
RESOURCE_LABEL = {0: "Wood", 1: "Brick", 2: "Sheep", 3: "Wheat", 4: "Ore"}
RESOURCE_EMOJI = {0: "🌲", 1: "🧱", 2: "🐑", 3: "🌾", 4: "⛰️"}
RESOURCE_LETTER = {0: "W", 1: "B", 2: "Sh", 3: "Wh", 4: "Or"}

# Standard Catan port layout — mirrors catan_engine/src/board.rs::standard_ports().
# Each entry: (kind, [v1, v2]) — kind is "3:1" or one of "wood"/"brick"/"sheep"/"wheat"/"ore".
PORTS = [
    ("3:1",   [0, 4]),
    ("brick", [2, 5]),
    ("3:1",   [10, 15]),
    ("wood",  [26, 32]),
    ("3:1",   [46, 50]),
    ("wheat", [49, 52]),
    ("ore",   [47, 51]),
    ("3:1",   [33, 38]),
    ("sheep", [11, 16]),
]
PORT_KIND_TO_RESOURCE_IDX = {"wood": 0, "brick": 1, "sheep": 2, "wheat": 3, "ore": 4}


def _emoji_font_props():
    """Return matplotlib FontProperties for color-emoji rendering, or None.

    Matplotlib doesn't honor `fontfamily='Segoe UI Emoji'` directly — it must be
    given the font file. We probe the typical Win11 path; on WSL/Linux the font
    is rarely installed, so we return None and the caller falls back to letters.
    """
    candidates = [
        "C:/Windows/Fonts/seguiemj.ttf",
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/System/Library/Fonts/Apple Color Emoji.ttc",
    ]
    from matplotlib import font_manager
    for path in candidates:
        if Path(path).exists():
            return font_manager.FontProperties(fname=path)
    return None


def _shade(hex_color: str, factor: float) -> str:
    """Multiply each RGB channel by `factor` (0..1 darken, >1 lighten clamped)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"


def _render_static_board_png(seed: int, out_path: Path, vertex_xy: dict | None = None):
    """Render the v2 ABC board for this seed: hexes + dice numbers + ports.
    Buildings, roads, robber, dev cards are SVG overlays drawn in JS."""
    fig = plt.figure(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))
    ax = fig.add_axes([0, 0, 1, 1])
    eng = _engine.Engine(seed)
    obs = eng.observation()
    hex_features = obs["hex_features"]
    emoji_fp = _emoji_font_props()

    for h in range(19):
        cx, cy = _hex_center_pointy(h)
        feats = hex_features[h]
        res = feats[:5]
        if res.sum() < 0.5:
            color = DESERT_COLOR
            label = "Desert"
            dice_str = None
        else:
            ridx = int(np.argmax(res))
            color = RESOURCE_COLORS[ridx]
            label = RESOURCE_LABEL[ridx]
            dice_norm = float(feats[5])
            if abs(dice_norm) > 1e-6:
                dice_num = int(round(dice_norm * 5.0 + 7.0))
                dice_str = str(dice_num)
            else:
                dice_str = None
        angles = [math.pi / 6 + i * math.pi / 3 for i in range(6)]
        outer_pts = [(cx + HEX_RADIUS * math.cos(a), cy + HEX_RADIUS * math.sin(a)) for a in angles]
        inner_pts = [(cx + 0.85 * HEX_RADIUS * math.cos(a), cy + 0.85 * HEX_RADIUS * math.sin(a)) for a in angles]
        outer_color = _shade(color, 0.78)
        inner_color = _shade(color, 1.12)
        stroke = _shade(color, 0.45)
        outer_poly = plt.Polygon(outer_pts, facecolor=outer_color, edgecolor=stroke, linewidth=1.4)
        inner_poly = plt.Polygon(inner_pts, facecolor=inner_color, edgecolor="none")
        ax.add_patch(outer_poly)
        ax.add_patch(inner_poly)
        if res.sum() < 0.5:
            ax.text(cx, cy, "Desert", ha="center", va="center",
                    fontsize=10, color=_shade(DESERT_COLOR, 0.5), fontstyle="italic")
        else:
            ridx = int(np.argmax(res))
            if emoji_fp is not None:
                ax.text(cx, cy + 0.42, RESOURCE_EMOJI[ridx], ha="center", va="center",
                        fontsize=14, fontproperties=emoji_fp)
            else:
                # No emoji font available — fall back to a readable letter label
                # in the resource color, which works on any matplotlib install.
                ax.text(cx, cy + 0.42, RESOURCE_LETTER[ridx], ha="center", va="center",
                        fontsize=11, fontweight="bold", color=_shade(color, 0.35))
        if dice_str is not None:
            num = int(dice_str)
            is_hot = num in (6, 8)
            ring_color = "#cc2222" if is_hot else "#444444"
            text_color = "#cc2222" if is_hot else "#222222"
            shadow = plt.Circle((cx + 0.02, cy - 0.06), 0.30,
                                facecolor="black", edgecolor="none", alpha=0.25)
            ax.add_patch(shadow)
            disk = plt.Circle((cx, cy - 0.05), 0.30,
                              facecolor="#fdf2c8", edgecolor=ring_color,
                              linewidth=2.0 if is_hot else 1.4)
            ax.add_patch(disk)
            ax.text(cx, cy - 0.02, dice_str, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)
            pips = 6 - abs(7 - num)
            ax.text(cx, cy - 0.18, "·" * pips, ha="center", va="center",
                    fontsize=8, color=text_color)

    # Port glyphs — small circles on the coast, with a connector to each port vertex.
    if vertex_xy is not None:
        board_cx = sum(p[0] for p in vertex_xy.values()) / len(vertex_xy)
        board_cy = sum(p[1] for p in vertex_xy.values()) / len(vertex_xy)
        for kind, (v1, v2) in PORTS:
            x1, y1 = vertex_xy[v1]
            x2, y2 = vertex_xy[v2]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            # Push outward from the board center along the perpendicular bisector.
            dx, dy = mx - board_cx, my - board_cy
            mag = (dx * dx + dy * dy) ** 0.5
            offset = 0.45
            if mag > 1e-6:
                dx, dy = dx / mag * offset, dy / mag * offset
            px, py = mx + dx, my + dy
            if kind == "3:1":
                face = "#e8e2c8"
                edge_c = "#5d4715"
                label = "3:1"
                text_c = "#222"
                ridx = None
            else:
                ridx = PORT_KIND_TO_RESOURCE_IDX[kind]
                face = RESOURCE_COLORS[ridx]
                edge_c = _shade(face, 0.45)
                text_c = "white"
                label = "2:1"
            # Connector lines from each port vertex to the port disk.
            ax.plot([x1, px], [y1, py], color=edge_c, linewidth=1.2, zorder=1, alpha=0.7)
            ax.plot([x2, px], [y2, py], color=edge_c, linewidth=1.2, zorder=1, alpha=0.7)
            disk = plt.Circle((px, py), 0.20, facecolor=face, edgecolor=edge_c,
                              linewidth=1.4, zorder=4)
            ax.add_patch(disk)
            ax.text(px, py + 0.03, label, ha="center", va="center",
                    fontsize=6, fontweight="bold", color=text_c, zorder=5)
            if ridx is not None:
                if emoji_fp is not None:
                    ax.text(px, py - 0.09, RESOURCE_EMOJI[ridx], ha="center", va="center",
                            fontsize=7, fontproperties=emoji_fp, zorder=5)
                else:
                    ax.text(px, py - 0.09, RESOURCE_LETTER[ridx], ha="center", va="center",
                            fontsize=6, fontweight="bold", color=text_c, zorder=5)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


# ===================== action description (v2: 280 actions) =====================

def _action_desc(a: int) -> str:
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


def _read_action_history(run_dir: Path, seed: int) -> tuple[list[int], int, list[int]]:
    """Find seed's action_history + winner + final_vp from a v2 run_dir.

    The recorder writes per-cell shards labelled like
    games.sims=25.v2cell.parquet (or moves.sims=25.v2cell.parquet); under
    workers> 1 they live in workerN/ subdirs. We just glob and concat."""
    games_files = sorted(run_dir.rglob("games*.parquet"))
    if not games_files:
        raise FileNotFoundError(f"no games*.parquet under {run_dir}")
    found = None
    for gp in games_files:
        t = pq.read_table(gp)
        if "action_history" not in t.column_names:
            continue
        seeds = t.column("seed").to_pylist()
        if seed not in seeds:
            continue
        i = seeds.index(seed)
        ah = t.column("action_history").to_pylist()[i]
        winner = t.column("winner").to_pylist()[i]
        final_vp = t.column("final_vp").to_pylist()[i]
        found = (list(map(int, ah)), int(winner), list(map(int, final_vp)))
        break
    if found is None:
        raise ValueError(f"seed {seed} not found in any games.parquet under {run_dir}")
    return found


def _replay_to_states(seed: int, history: list[int]) -> list[dict]:
    """Walk the engine through history, snapshotting per step.

    Returns a list of state dicts ready to ship to the JS overlay. Each
    state dict has the keys: n (narration), cp (current player or -1 at
    terminal), s (settlements as [(v, owner), ...]), c (cities), r (roads),
    rh (robber hex), vp (live VPs), hands (per-player breakdown + total),
    bank (5-vec), dev_held (per-player [5]: knight, RB, mono, YOP, VP),
    lr_len (per-player), knights_played (per-player), built (per-player
    {settle, city, road}), ports (per-player [6]), lr_holder, la_holder.
    """
    eng = _engine.Engine(seed)
    states: list[dict] = []
    CHANCE = 0x8000_0000

    def snapshot(narration: str):
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

        states.append({
            "n": narration,
            "cp": cp,
            "phase": phase_name,
            "s": settlements,
            "c": cities,
            "r": roads,
            "rh": robber_hex,
            "vp": vps,
            "hands": hands_breakdown,
            "bank": bank,
            "dev_held": [pp["dev_held"] for pp in per_player],
            "ports": [pp["ports"] for pp in per_player],
            "lr_len": lr_len,
            "knights": knights,
            "built": [
                {"settle": settle_built[p], "city": city_built[p], "road": road_built[p]}
                for p in range(4)
            ],
            "lr_holder": lr_holder,
            "la_holder": la_holder,
            "vp_played": vp_played,
        })

    snapshot("(initial state)")
    history_idx = 0
    step = 0
    last_action: int | None = None
    while not eng.is_terminal() and step < 30_000:
        if eng.is_chance_pending():
            if history_idx >= len(history):
                break
            a = history[history_idx]; history_idx += 1
            value = a & 0x7FFF_FFFF
            if not (a & CHANCE):
                # log dropped the chance flag; fall through optimistically
                value = a
            if value < 256:
                narr = f"CHANCE: dice → {value}"
            else:
                narr = f"CHANCE: steal p{value // 256} card{value % 256}"
            eng.apply_chance_outcome(value)
            last_action = a
        else:
            if history_idx >= len(history):
                break
            a = history[history_idx]; history_idx += 1
            cp = int(eng.current_player())
            narr = f"P{cp} {_action_desc(a)}"
            eng.step(int(a))
            last_action = a
        step += 1
        snapshot(narr)
        if step % 500 == 0:
            print(f"  step {step}, history_used={history_idx}/{len(history)}", flush=True)
    return states


# ===================== HTML viewer =====================

INDEX_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Catan v2 replay seed {{SEED}}</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 16px; background: #f5f5f5; color: #222; }
  h1 { font-size: 16px; margin: 0 0 8px; }
  .row { display: flex; gap: 16px; align-items: flex-start; }
  .board-col { flex: 0 0 auto; }
  .players-col { flex: 1 1 360px; min-width: 340px; }
  .panel { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 8px; }
  .controls { margin: 8px 0; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  button { padding: 5px 10px; font-size: 13px; cursor: pointer; }
  button.active { background: #ffe97a; }
  input[type=range] { flex: 1; min-width: 200px; }
  #status { font-family: monospace; font-size: 13px; }
  #narration { padding: 6px 10px; background: #1f2a3a; color: #f0e8c8;
               border: 1px solid #2c3a52; border-radius: 4px;
               margin: 6px 0; font-family: ui-monospace, monospace; font-size: 13px; min-height: 22px; }
  .dice-chip { background: #ffd633; color: #1f2a3a; padding: 0 6px;
               border-radius: 3px; font-weight: 700; margin-right: 6px; }
  .seat-tag-0 { color: #ff8c8c; font-weight: 700; }
  .seat-tag-1 { color: #88a6e6; font-weight: 700; }
  .seat-tag-2 { color: #88d4a0; font-weight: 700; }
  .seat-tag-3 { color: #e6b07a; font-weight: 700; }
  #boardWrap { position: relative; display: block; width: 700px; max-width: 100%; }
  #board { display: block; width: 100%; height: auto; }
  svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none; overflow: visible; }
  table.vp { border-collapse: collapse; font-size: 12px; width: 100%; table-layout: fixed; }
  table.vp th, table.vp td { border: 1px solid #ddd; padding: 3px 6px; vertical-align: top;
                              overflow: hidden; text-overflow: ellipsis; word-wrap: break-word; }
  table.vp th { background: #f0f0f0; text-align: left; }
  table.vp tr.cp-row td { background: linear-gradient(90deg, #fff7c4 0%, #fffae0 100%) !important;
                          border-top: 2px solid #f5b800; border-bottom: 2px solid #f5b800;
                          font-weight: 600; }
  table.vp tr.cp-row td:first-child { border-left: 4px solid #f5b800; }
  @keyframes turnPulse { 0%,100% { box-shadow: 0 0 6px 1px #ffd633; }
                          50% { box-shadow: 0 0 14px 3px #ffe97a; } }
  .seat-chip.cp { outline: 2px solid #ffd633; outline-offset: 1px;
                  animation: turnPulse 1.6s ease-in-out infinite; }
  .swatch { display: inline-block; width: 12px; height: 12px; vertical-align: middle;
            margin-right: 6px; border: 1px solid #333; border-radius: 2px; }
  .key-hint { color: #888; font-size: 11px; }
  select { padding: 4px 6px; font-size: 13px; }
  .badge { display: inline-block; padding: 1px 5px; margin-right: 3px; border-radius: 3px;
           font-size: 11px; background: #f0f0f0; border: 1px solid #ccc; }
  .badge.lr { background: #d4f7d4; border-color: #6c6; }
  .badge.la { background: #f7d4d4; border-color: #c66; }
  .seat-strip { display: flex; gap: 4px; margin: 0 0 6px 0; font-size: 11px; }
  .seat-chip {
    flex: 1; padding: 4px 6px; border-radius: 4px;
    color: white; font-weight: 600; display: flex; justify-content: space-between;
    align-items: center; min-width: 0;
  }
  .seat-chip .vp { background: rgba(255,255,255,0.25); padding: 1px 5px; border-radius: 3px; font-size: 10px; }
  .seat-chip .ttl { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dim { color: #999; }
</style>
</head>
<body>
<h1>Catan v2 replay — seed {{SEED}}, {{N_STEPS}} steps</h1>
<div id="narration">(loading)</div>
<div class="controls">
  <button id="first">⏮</button>
  <button id="prevBig">−10</button>
  <button id="prev">◀</button>
  <button id="play">▶ play</button>
  <button id="next">▶</button>
  <button id="nextBig">+10</button>
  <button id="last">⏭</button>
  <label style="font-size:12px; margin-left:8px;">speed
    <select id="speed">
      <option value="100">very fast (0.1s)</option>
      <option value="200">fast (0.2s)</option>
      <option value="500" selected>normal (0.5s)</option>
      <option value="1000">slow (1.0s)</option>
      <option value="2000">very slow (2.0s)</option>
    </select>
  </label>
  <input type="range" id="slider" min="0" max="0" value="0" step="1">
  <span id="status">step 0</span>
  <span class="key-hint">←/→ ±1 · Shift+←/→ ±10 · Home/End · Space play/pause</span>
</div>
<div class="row">
  <div class="board-col panel">
    <div id="boardWrap">
      <img id="board" src="data:image/png;base64,{{BOARD_B64}}" alt="board">
      <svg id="overlay" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
  </div>
  <div class="players-col panel">
    <h3 style="margin:0 0 6px; font-size:14px;">Players</h3>
    <div class="seat-strip" id="seatStrip"></div>
    <table class="vp" id="vpTable"></table>
    <p style="font-size:11px; color:#666; margin:8px 0 0;">
      v2: every value is read directly from the engine — hand breakdowns,
      bank, dev cards, longest road / largest army, building inventory,
      and ports. Highlighted row = current player. <span class="badge lr">LR</span> = longest
      road holder, <span class="badge la">LA</span> = largest army holder.
    </p>
  </div>
</div>

<script>
const PLAYER_COLORS = ["#cc3333", "#3366cc", "#33aa55", "#cc8833"];
const PLAYER_COLORS_DARK = ["#5a1414", "#1a3370", "#1a5a2c", "#5a3a14"];
const SEAT_NAMES = {{SEAT_NAMES}};
const RESOURCES = ['🪵', '🧱', '🐑', '🌾', '⛰️'];
const DEV_NAMES = ['Knight', 'Road Building', 'Monopoly', 'Year of Plenty', 'Victory Point'];
const DEV_EMOJI = ['⚔️', '🛣️', '📜', '🌽', '⭐'];
const PORT_NAMES = ['3:1', 'Wd 2:1', 'Bk 2:1', 'Sh 2:1', 'Wh 2:1', 'Or 2:1'];
let layout = null, overlays = null, cur = 0, playing = false, playTimer = null;

const img = document.getElementById('board');
const svg = document.getElementById('overlay');
const narr = document.getElementById('narration');
const status = document.getElementById('status');
const slider = document.getElementById('slider');
const vpTable = document.getElementById('vpTable');
const playBtn = document.getElementById('play');

function dataToPx(x, y) {
  const w = img.clientWidth, h = img.clientHeight;
  const [x0, x1] = layout.xlim;
  const [y0, y1] = layout.ylim;
  const px = ((x - x0) / (x1 - x0)) * w;
  const py = h - ((y - y0) / (y1 - y0)) * h;
  return [px, py];
}

function fmtBreakdown(arr) {
  const parts = [];
  for (let r = 0; r < 5; r++) if (arr[r] > 0) parts.push(`${RESOURCES[r]}${arr[r]}`);
  return parts.length ? parts.join(' ') : '<span class="dim">empty</span>';
}

function fmtDev(arr) {
  const parts = [];
  for (let k = 0; k < 5; k++) {
    if (arr[k] > 0) {
      parts.push(`<span title="${DEV_NAMES[k]}">${DEV_EMOJI[k]}×${arr[k]}</span>`);
    }
  }
  return parts.length ? parts.join(' ') : '<span class="dim">none</span>';
}

function fmtPorts(arr) {
  const parts = [];
  for (let i = 0; i < 6; i++) if (arr[i]) parts.push(PORT_NAMES[i]);
  return parts.length ? parts.join(', ') : '<span class="dim">none</span>';
}

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function formatNarration(s) {
  let m = s.match(/^CHANCE: dice → (\d+)$/);
  if (m) return `<span class="dice-chip">🎲 ${m[1]}</span>dice rolled`;
  m = s.match(/^CHANCE: steal p(\d) card(\d+)$/);
  if (m) return `<span class="dice-chip">🥷</span>steal from <span class="seat-tag-${m[1]}">P${m[1]}</span> (card ${m[2]})`;
  m = s.match(/^P(\d) (.*)$/);
  if (m) return `<span class="seat-tag-${m[1]}">P${m[1]}</span> ${escapeHtml(m[2])}`;
  return escapeHtml(s);
}

function renderState() {
  if (!overlays || !layout) return;
  const st = overlays.states[cur];
  if (!st) return;
  svg.setAttribute('width', img.clientWidth);
  svg.setAttribute('height', img.clientHeight);
  svg.setAttribute('viewBox', `0 0 ${img.clientWidth} ${img.clientHeight}`);

  // Diff against the previous state to find "newly built this step" assets.
  // Each of (s, c, r) is a list of [id, owner] pairs. We treat a vertex/edge
  // as newly built if it didn't appear with the same owner in prev.s/.c/.r
  // (e.g. a settlement upgraded to a city counts as a new city, not a new
  // settlement, so the glow follows the upgrade).
  const prev = (cur > 0) ? overlays.states[cur - 1] : null;
  function asKeySet(arr) { const s = new Set(); for (const [id, o] of arr) s.add(`${id}:${o}`); return s; }
  const prevSettleKeys = prev ? asKeySet(prev.s) : new Set();
  const prevCityKeys   = prev ? asKeySet(prev.c) : new Set();
  const prevRoadKeys   = prev ? asKeySet(prev.r) : new Set();
  const isNewSettle = ([vid, owner]) => prev && !prevSettleKeys.has(`${vid}:${owner}`);
  const isNewCity   = ([vid, owner]) => prev && !prevCityKeys.has(`${vid}:${owner}`);
  const isNewRoad   = ([eid, owner]) => prev && !prevRoadKeys.has(`${eid}:${owner}`);

  // SVG filter definitions: gold glow for newly-built objects this step,
  // and a per-player soft glow used to highlight ALL of the current player's
  // assets while it's their turn. The player-color glow uses CSS variables
  // applied via filter URL fragments per render.
  let body = `<defs>
    <filter id="goldGlow" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feFlood flood-color="#ffd633" flood-opacity="0.95" result="gold"/>
      <feComposite in="gold" in2="blur" operator="in" result="goldBlur"/>
      <feMerge>
        <feMergeNode in="goldBlur"/>
        <feMergeNode in="goldBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <filter id="cpGlow" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="1.5" result="b1"/>
      <feFlood flood-color="#ffffff" flood-opacity="0.55" result="white"/>
      <feComposite in="white" in2="b1" operator="in" result="bw"/>
      <feMerge>
        <feMergeNode in="bw"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>`;

  if (st.rh >= 0) {
    const [hx, hy] = layout.hex_centers[st.rh];
    const [px, py] = dataToPx(hx, hy);
    body += `<g style="filter: drop-shadow(0 2px 3px rgba(0,0,0,0.6))">`;
    body += `<ellipse cx="${px}" cy="${py + 13}" rx="14" ry="3.5" fill="rgba(0,0,0,0.4)"/>`;
    body += `<path d="M ${px - 11} ${py + 9} ` +
            `L ${px - 11} ${py - 5} ` +
            `Q ${px - 11} ${py - 14} ${px} ${py - 16} ` +
            `Q ${px + 11} ${py - 14} ${px + 11} ${py - 5} ` +
            `L ${px + 11} ${py + 9} Z" ` +
            `fill="#222" stroke="white" stroke-width="1.5"/>`;
    body += `<circle cx="${px}" cy="${py - 11}" r="5.5" fill="#222" stroke="white" stroke-width="1.5"/>`;
    body += `</g>`;
  }

  // Helper: pick the visual emphasis filter for an asset.
  //   - "newly built this step" wins over everything → gold glow
  //   - else if the owner is the current player → soft white glow
  //   - else → no filter
  function pickFilter(isNew, owner) {
    if (isNew) return ' filter="url(#goldGlow)"';
    if (st.cp === owner) return ' filter="url(#cpGlow)"';
    return '';
  }
  // Bold the stroke for current-player assets so it pops even at small sizes.
  function pickStrokeWidth(owner, base) {
    return st.cp === owner ? (base + 0.9) : base;
  }

  for (const pair of st.r) {
    const [eid, owner] = pair;
    const e = layout.edges[eid];
    const [x1, y1] = dataToPx(e[0], e[1]);
    const [x2, y2] = dataToPx(e[2], e[3]);
    const fAttr = pickFilter(isNewRoad(pair), owner);
    body += `<g${fAttr}>`;
    body += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="white" stroke-width="7" stroke-linecap="round"/>`;
    body += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${PLAYER_COLORS[owner]}" stroke-width="4.5" stroke-linecap="round"/>`;
    body += `</g>`;
  }

  for (const pair of st.s) {
    const [vid, owner] = pair;
    const v = layout.vertices[String(vid)];
    const [px, py] = dataToPx(v[0], v[1]);
    const sz = 11;
    const path = `M ${px - sz} ${py + sz} L ${px + sz} ${py + sz} L ${px + sz} ${py - sz/3} L ${px} ${py - sz} L ${px - sz} ${py - sz/3} Z`;
    const sw = pickStrokeWidth(owner, 1.8);
    const fAttr = pickFilter(isNewSettle(pair), owner);
    body += `<g${fAttr}>`;
    body += `<path d="${path}" fill="${PLAYER_COLORS[owner]}" stroke="${PLAYER_COLORS_DARK[owner]}" stroke-width="${sw}" stroke-linejoin="round"/>`;
    body += `<rect x="${px - 2.5}" y="${py + sz/3}" width="5" height="${sz - sz/3 - 1}" fill="${PLAYER_COLORS_DARK[owner]}"/>`;
    body += `</g>`;
  }

  for (const pair of st.c) {
    const [vid, owner] = pair;
    const v = layout.vertices[String(vid)];
    const [px, py] = dataToPx(v[0], v[1]);
    const sz = 14;
    const dark = PLAYER_COLORS_DARK[owner];
    const base = `M ${px - sz} ${py + sz} L ${px + sz} ${py + sz} L ${px + sz} ${py - sz/2} L ${px} ${py - sz - 3} L ${px - sz} ${py - sz/2} Z`;
    const sw = pickStrokeWidth(owner, 1.8);
    const fAttr = pickFilter(isNewCity(pair), owner);
    body += `<g${fAttr}>`;
    body += `<path d="${base}" fill="${PLAYER_COLORS[owner]}" stroke="${dark}" stroke-width="${sw}" stroke-linejoin="round"/>`;
    body += `<rect x="${px + sz/2}" y="${py - sz - 1}" width="4" height="7" fill="${PLAYER_COLORS[owner]}" stroke="${dark}" stroke-width="1.2"/>`;
    body += `<rect x="${px - sz + 3}" y="${py - 1}" width="${sz * 2 - 6}" height="3.5" fill="${dark}" opacity="0.55"/>`;
    body += `<rect x="${px - 3}" y="${py + sz/3}" width="6" height="${sz - sz/3}" fill="${dark}" opacity="0.7"/>`;
    body += `</g>`;
  }

  svg.innerHTML = body;
  narr.innerHTML = formatNarration(st.n);
  let cpStr = st.cp >= 0 ? `cp=${st.cp} (${SEAT_NAMES[st.cp]})` : 'terminal';
  status.textContent = `step ${cur} / ${overlays.n_steps - 1}  |  ${cpStr}  |  phase=${st.phase}`;
  slider.value = cur;

  // Locked column widths so the table doesn't reflow as hand contents change.
  // Hand gets the widest slot since it grows the most; ports/dev are tighter.
  let html = '<colgroup>' +
             '<col style="width:90px">' +    // seat (wide enough for "P0 LR LA" badges)
             '<col style="width:32px">' +    // VP
             '<col style="width:200px">' +   // hand
             '<col style="width:108px">' +   // dev cards
             '<col style="width:50px">' +    // built S/C/R
             '<col style="width:54px">' +    // LR / knights
             '<col style="width:84px">' +    // ports
             '</colgroup>';
  html += '<tr><th>seat</th><th>VP</th><th>hand</th><th>dev cards</th><th>built (S/C/R)</th><th>LR / Knights</th><th>ports</th></tr>';
  for (let i = 0; i < 4; i++) {
    const cpClass = (st.cp === i) ? 'cp-row' : '';
    const h = st.hands[i];
    let badges = '';
    if (st.lr_holder === i) badges += '<span class="badge lr">LR</span>';
    if (st.la_holder === i) badges += '<span class="badge la">LA</span>';
    const turnIcon = (st.cp === i) ? '<span title="Current turn" style="margin-right:3px">▶</span>' : '';
    const handStr = `${fmtBreakdown(h.breakdown)} <span class="dim">(${h.total})</span>`;
    let devStr = fmtDev(st.dev_held[i]);
    const vpcPlayed = (st.vp_played && st.vp_played[i]) || 0;
    if (vpcPlayed > 0) {
      const playedStr = `<span title="Victory Point cards already cashed in" class="dim">⭐×${vpcPlayed} played</span>`;
      devStr = (devStr.includes('none') ? playedStr : `${devStr} · ${playedStr}`);
    }
    const built = st.built[i];
    const builtStr = `${built.settle}/${built.city}/${built.road}`;
    const lrK = `${st.lr_len[i]} / ${st.knights[i]}`;
    const portStr = fmtPorts(st.ports[i]);
    html += `<tr class="${cpClass}">` +
            `<td style="white-space:nowrap">${turnIcon}<span class="swatch" style="background:${PLAYER_COLORS[i]}"></span>${i} ${badges}</td>` +
            `<td style="text-align:right;font-weight:bold">${st.vp[i]}</td>` +
            `<td>${handStr}</td>` +
            `<td>${devStr}</td>` +
            `<td>${builtStr}</td>` +
            `<td>${lrK}</td>` +
            `<td style="font-size:11px">${portStr}</td>` +
            `</tr>`;
  }
  if (st.bank) {
    const bankTotal = st.bank.reduce((a, b) => a + b, 0);
    html += `<tr style="background:#f0f0f0;border-top:2px solid #aaa">` +
            `<td colspan="2" style="font-style:italic;color:#555">🏦 bank (${bankTotal})</td>` +
            `<td colspan="5">${fmtBreakdown(st.bank)}</td></tr>`;
  }
  let stripHtml = '';
  for (let i = 0; i < 4; i++) {
    const isCp = (st.cp === i);
    let badges = '';
    if (st.la_holder === i) badges += ' ⚔️';
    if (st.lr_holder === i) badges += ' 🛣️';
    const vpc = (st.vp_played && st.vp_played[i]) || 0;
    if (vpc > 0) badges += ` <span title="VP cards played">⭐×${vpc}</span>`;
    // Emit literal `class="seat-chip"` so static-HTML scanners can match the
    // base class; the `cp` modifier is added via classList below.
    stripHtml += `<div class="seat-chip" style="background:${PLAYER_COLORS[i]}">` +
                 `<span class="ttl">P${i}${badges}</span>` +
                 `<span class="vp">${st.vp[i]} VP</span>` +
                 `</div>`;
  }
  const seatStrip = document.getElementById('seatStrip');
  seatStrip.innerHTML = stripHtml;
  for (let i = 0; i < 4; i++) {
    if (st.cp === i) seatStrip.children[i].classList.add('cp');
  }
  vpTable.innerHTML = html;
}

function go(delta) { cur = Math.max(0, Math.min(overlays.n_steps - 1, cur + delta)); renderState(); }
function jump(idx) { cur = Math.max(0, Math.min(overlays.n_steps - 1, idx)); renderState(); }

document.getElementById('first').onclick   = () => jump(0);
document.getElementById('prevBig').onclick = () => go(-10);
document.getElementById('prev').onclick    = () => go(-1);
document.getElementById('next').onclick    = () => go(1);
document.getElementById('nextBig').onclick = () => go(10);
document.getElementById('last').onclick    = () => jump(overlays.n_steps - 1);
slider.oninput = (e) => jump(parseInt(e.target.value, 10));
playBtn.onclick = togglePlay;

function getSpeedMs() { return parseInt(document.getElementById('speed').value, 10) || 500; }

function startPlay() {
  if (playTimer) clearInterval(playTimer);
  playTimer = setInterval(() => {
    if (cur >= overlays.n_steps - 1) { togglePlay(); return; }
    go(1);
  }, getSpeedMs());
}

function togglePlay() {
  playing = !playing;
  playBtn.textContent = playing ? '⏸ pause' : '▶ play';
  playBtn.classList.toggle('active', playing);
  if (playing) startPlay();
  else clearInterval(playTimer);
}

document.getElementById('speed').addEventListener('change', () => { if (playing) startPlay(); });

window.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft')  { e.shiftKey ? go(-10) : go(-1); e.preventDefault(); }
  if (e.key === 'ArrowRight') { e.shiftKey ? go(10)  : go(1);  e.preventDefault(); }
  if (e.key === 'Home')       { jump(0); e.preventDefault(); }
  if (e.key === 'End')        { jump(overlays.n_steps - 1); e.preventDefault(); }
  if (e.key === ' ')          { togglePlay(); e.preventDefault(); }
});

img.addEventListener('load', renderState);
window.addEventListener('resize', renderState);

layout = {{LAYOUT_JSON}};
overlays = {{OVERLAYS_JSON}};
slider.max = overlays.n_steps - 1;
if (img.complete) renderState();
</script>
</body>
</html>
"""


def render(run_dir: Path, seed: int, out_dir: Path | None = None) -> Path:
    """Replay seed from run_dir's parquet and emit a self-contained HTML viewer.

    Returns the path to index.html."""
    if out_dir is None:
        out_dir = run_dir / f"playback_seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  reading action_history for seed={seed}...", flush=True)
    history, winner, final_vp = _read_action_history(run_dir, seed)
    print(f"  history length = {len(history)}, winner=p{winner}, final_vp={final_vp}", flush=True)

    print("  building layout...", flush=True)
    vertex_xy, edges, hex_centers = _build_layout()

    print("  rendering board PNG...", flush=True)
    board_png = out_dir / "board.png"
    _render_static_board_png(seed, board_png, vertex_xy=vertex_xy)
    layout = {
        "xlim": list(XLIM),
        "ylim": list(YLIM),
        "vertices": {str(v): list(xy) for v, xy in vertex_xy.items()},
        "edges": [list(e) for e in edges],
        "hex_centers": [list(c) for c in hex_centers],
    }

    print("  replaying engine...", flush=True)
    states = _replay_to_states(seed, history)
    print(f"  captured {len(states)} states", flush=True)

    overlays = {
        "seed": seed,
        "n_steps": len(states),
        "winner": winner,
        "final_vp": final_vp,
        "states": states,
    }

    seat_names = [f"P{i}" for i in range(4)]
    board_b64 = base64.b64encode(board_png.read_bytes()).decode("ascii")
    html = (INDEX_HTML
            .replace("{{SEED}}", str(seed))
            .replace("{{SEAT_NAMES}}", json.dumps(seat_names))
            .replace("{{N_STEPS}}", str(len(states)))
            .replace("{{BOARD_B64}}", board_b64)
            .replace("{{LAYOUT_JSON}}", json.dumps(layout))
            .replace("{{OVERLAYS_JSON}}", json.dumps(overlays)))
    out_path = out_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    size_kb = out_path.stat().st_size // 1024
    print(f"DONE. Open: {out_path}  ({size_kb} KB self-contained)", flush=True)
    return out_path


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path, help="path to a v2 run dir (containing games/moves parquet)")
    p.add_argument("seed", type=int, help="seed of the game to render")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="destination dir (default: <run_dir>/playback_seed_<seed>)")
    args = p.parse_args()
    render(args.run_dir, args.seed, args.out_dir)


if __name__ == "__main__":
    cli_main()
