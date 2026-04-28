"""Render the standard Catan board topology (hex IDs + resources only).
No live state, no dice numbers — just the structural layout."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Standard 19-hex Catan layout: rows of 3, 4, 5, 4, 3 hexes.
ROW_LENGTHS = [3, 4, 5, 4, 3]
HEX_RADIUS = 1.0

# Hex ID -> (row, col-in-row)
HEX_ROW_COL = {}
hid = 0
for r, n in enumerate(ROW_LENGTHS):
    for c in range(n):
        HEX_ROW_COL[hid] = (r, c)
        hid += 1


def hex_center_pointy(hex_id: int) -> tuple[float, float]:
    """Pointy-topped hex centre."""
    row, col = HEX_ROW_COL[hex_id]
    n_in_row = ROW_LENGTHS[row]
    spacing_x = math.sqrt(3) * HEX_RADIUS
    x_offset = -(n_in_row - 1) * spacing_x / 2
    x = x_offset + col * spacing_x
    y = -row * 1.5 * HEX_RADIUS
    return x, y


def draw_hex(ax, cx, cy, color, label, hex_id):
    angles = [math.pi / 6 + i * math.pi / 3 for i in range(6)]
    pts = [(cx + HEX_RADIUS * math.cos(a), cy + HEX_RADIUS * math.sin(a)) for a in angles]
    ax.add_patch(plt.Polygon(pts, facecolor=color, edgecolor="black", linewidth=1.5))
    ax.text(cx, cy + 0.15, label, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(cx, cy - 0.25, f"h{hex_id}", ha="center", va="center",
            fontsize=9, style="italic", color="#333333")


# Resources from catan_engine/src/board.rs::standard_hexes
RESOURCES = [
    "Ore",   "Sheep", "Wood",
    "Wheat", "Brick", "Sheep", "Brick",
    "Wheat", "Wood",  "Desert", "Wood", "Ore",
    "Wood",  "Ore",   "Wheat",
    "Sheep", "Brick", "Wheat", "Sheep",
]
COLORS = {
    "Wood":   "#3d8b37",
    "Brick":  "#a04020",
    "Sheep":  "#90c060",
    "Wheat":  "#e6c243",
    "Ore":    "#7a7a7a",
    "Desert": "#d4b483",
}


def main(out_path: str = "mcts_study/board_layout.png"):
    fig, ax = plt.subplots(figsize=(9, 8))
    for h in range(19):
        cx, cy = hex_center_pointy(h)
        res = RESOURCES[h]
        draw_hex(ax, cx, cy, COLORS[res], res, h)

    ax.set_title("Catan engine — standard board layout (19 hexes, fixed)",
                 fontsize=12)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-7, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    legend_handles = [
        mpatches.Patch(color=COLORS[r], label=r)
        for r in ("Wood", "Brick", "Sheep", "Wheat", "Ore", "Desert")
    ]
    ax.legend(handles=legend_handles, loc="lower center", ncol=6,
              fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
