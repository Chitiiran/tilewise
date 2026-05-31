"""Board geometry + static PNG renderer, shared by the live web server and
the offline replay viewer (catan_mcts.playback).

Moved verbatim out of catan_mcts.playback during the Phase-1 extraction.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def build_layout():
    """Public: (vertex_xy: dict[int,(x,y)], edges: list[(x1,y1,x2,y2)], hex_centers: list[(x,y)])."""
    return _build_layout()


def layout_dict() -> dict:
    """Public: JSON-ready layout for the frontend (same shape playback emits)."""
    vertex_xy, edges, hex_centers = _build_layout()
    return {
        "xlim": list(XLIM),
        "ylim": list(YLIM),
        "vertices": {str(v): list(xy) for v, xy in vertex_xy.items()},
        "edges": [list(e) for e in edges],
        "hex_centers": [list(c) for c in hex_centers],
    }


def render_board_png(seed: int, out_path: Path, vertex_xy: dict | None = None) -> None:
    """Public: render the static board PNG for `seed`."""
    if vertex_xy is None:
        vertex_xy, _, _ = _build_layout()
    _render_static_board_png(seed, out_path, vertex_xy=vertex_xy)
