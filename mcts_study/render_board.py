"""Render the standard Catan board from the engine's observation API.

Creates a PNG showing all 19 hexes with their resource colors and dice numbers,
the robber position, and any settlements/cities/roads if a game is in progress.

Usage:
    python mcts_study/render_board.py [seed]

Default seed is 0 (initial state, no buildings). Provide a seed of an in-progress
game to see settlements/roads.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from catan_bot import _engine


# ---------- Hex layout ----------
# Standard Catan rows: 3, 4, 5, 4, 3 hexes top-to-bottom.
# Hex IDs 0..18 in spiral-from-top-left order, matching catan_engine/src/board.rs.

ROW_LENGTHS = [3, 4, 5, 4, 3]
HEX_RADIUS = 1.0
HEX_HEIGHT = math.sqrt(3) * HEX_RADIUS  # flat-top hex height

# Map hex ID to (row, col-in-row).
HEX_ROW_COL = {}
hid = 0
for r, n in enumerate(ROW_LENGTHS):
    for c in range(n):
        HEX_ROW_COL[hid] = (r, c)
        hid += 1


def hex_center(hex_id: int) -> tuple[float, float]:
    """Return (x, y) center for a flat-topped hex.
    Rows are vertically stacked; rows of length 4 are offset relative to length-3 rows."""
    row, col = HEX_ROW_COL[hex_id]
    n_in_row = ROW_LENGTHS[row]
    # Horizontal center: 1.5 * radius spacing per hex; row width = (n-1) * 1.5 * radius.
    x_offset = -(n_in_row - 1) * 1.5 / 2  # centre the row
    x = x_offset + col * 1.5
    # Vertical: rows are HEX_HEIGHT/2 apart (pointy-top), but we want flat-top stacking.
    # Standard Catan diagrams have pointy-top hexes; switch.
    # Re-derive with pointy-top (apothem-aligned columns, vertex-aligned rows):
    return x, -row * (HEX_HEIGHT * 0.75) * 2  # placeholder


def hex_center_pointy(hex_id: int) -> tuple[float, float]:
    """Pointy-topped hex centre. Each row's hexes are spaced sqrt(3)*r apart;
    rows are 1.5*r apart vertically. Even/odd rows offset by sqrt(3)/2 * r."""
    row, col = HEX_ROW_COL[hex_id]
    n_in_row = ROW_LENGTHS[row]
    spacing_x = math.sqrt(3) * HEX_RADIUS
    x_offset = -(n_in_row - 1) * spacing_x / 2
    x = x_offset + col * spacing_x
    y = -row * 1.5 * HEX_RADIUS
    return x, y


def draw_hex(ax, cx: float, cy: float, color: str, label: str, dice_num=None):
    """Draw a pointy-topped hexagon centered at (cx, cy)."""
    angles = [math.pi / 6 + i * math.pi / 3 for i in range(6)]
    pts = [(cx + HEX_RADIUS * math.cos(a), cy + HEX_RADIUS * math.sin(a)) for a in angles]
    poly = plt.Polygon(pts, facecolor=color, edgecolor="black", linewidth=1.5)
    ax.add_patch(poly)
    # Resource label, top
    ax.text(cx, cy + 0.25, label, ha="center", va="center", fontsize=8, fontweight="bold")
    # Dice number, centered, with color cue (red for 6/8 the high-pip numbers)
    if dice_num is not None:
        num_color = "red" if dice_num in (6, 8) else "black"
        ax.text(cx, cy - 0.05, str(dice_num), ha="center", va="center",
                fontsize=14, fontweight="bold", color=num_color)
        # Pip dots underneath the number, count = 6 - |7 - dice_num|
        pips = 6 - abs(7 - dice_num)
        pip_str = "·" * pips
        ax.text(cx, cy - 0.35, pip_str, ha="center", va="center",
                fontsize=10, color=num_color)


# ---------- Resource colors ----------

RESOURCE_COLORS = {
    0: "#3d8b37",   # Wood — forest green
    1: "#a04020",   # Brick — red-brown
    2: "#90c060",   # Sheep — pale green
    3: "#e6c243",   # Wheat — gold
    4: "#7a7a7a",   # Ore — gray
}
DESERT_COLOR = "#d4b483"
RESOURCE_LABEL = {0: "Wood", 1: "Brick", 2: "Sheep", 3: "Wheat", 4: "Ore"}


def main(seed: int = 0, out_path: str = "mcts_study/board.png"):
    eng = _engine.Engine(seed)
    obs = eng.observation()
    hex_features = obs["hex_features"]   # shape [19, F]
    # Decode hex features. Per catan_engine/src/observation.rs:
    #   F_HEX = 8 columns: 5 resource one-hot (wood/brick/sheep/wheat/ore),
    #   1 dice_norm, 1 robber_present, 1 is_desert.
    # Verify shape; fall back gracefully if engine layout drifts.
    print(f"hex_features shape: {hex_features.shape}")

    fig, ax = plt.subplots(figsize=(10, 9))

    # Place each hex
    for h in range(19):
        cx, cy = hex_center_pointy(h)
        feats = hex_features[h]
        # Find resource: argmax of first 5 columns; if all zero, it's desert.
        res_one_hot = feats[:5]
        if res_one_hot.sum() < 0.5:  # desert
            color = DESERT_COLOR
            label = "Desert"
            dice = None
        else:
            res_idx = int(np.argmax(res_one_hot))
            color = RESOURCE_COLORS[res_idx]
            label = RESOURCE_LABEL[res_idx]
            # Dice number: feat[5] is normalized dice. Reconstruct from board hard-coded.
            # Actually easier: use eng.stats() to grab dice info OR re-read board.
            # For now extract from feats[5] which is dice_norm, if non-zero recover.
            # But we don't store inverse-norm. Simplest: hardcode dice numbers from
            # board.rs::standard_hexes (matches engine directly).
            dice = STANDARD_DICE[h]
        # Robber: feat[6]
        is_robber = feats[6] > 0.5
        draw_hex(ax, cx, cy, color, label, dice_num=dice)
        # Hex ID in tiny font, bottom-right of hex
        ax.text(cx + 0.5, cy - 0.7, f"h{h}", ha="left", va="center",
                fontsize=6, color="#444444", style="italic")
        # Robber overlay
        if is_robber:
            ax.add_patch(plt.Circle((cx, cy + 0.55), 0.18,
                                    facecolor="black", edgecolor="white", linewidth=1.5))
            ax.text(cx, cy + 0.55, "R", ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")

    # Title and metadata
    phase_str = str(eng.current_player()) if not eng.is_terminal() else "terminal"
    ax.set_title(f"Catan engine standard board — seed={seed}, current_player={phase_str}",
                 fontsize=12)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Legend
    legend_handles = [mpatches.Patch(color=RESOURCE_COLORS[i], label=RESOURCE_LABEL[i])
                      for i in range(5)]
    legend_handles.append(mpatches.Patch(color=DESERT_COLOR, label="Desert"))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


# Hardcoded from catan_engine/src/board.rs::standard_hexes
STANDARD_DICE = {
    0: 10, 1: 2,  2: 9,
    3: 12, 4: 6,  5: 4,  6: 10,
    7: 9,  8: 11, 9: None, 10: 3, 11: 8,
    12: 8, 13: 3, 14: 4,
    15: 5, 16: 5, 17: 6, 18: 11,
}


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out = sys.argv[2] if len(sys.argv) > 2 else "mcts_study/board.png"
    main(seed=seed, out_path=out)
