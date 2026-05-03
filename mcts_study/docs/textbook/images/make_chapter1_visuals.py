"""Generate visuals for textbook Chapter 1 — Input Projection.

Outputs (all written to this directory):
  ch1_fig1_state_breakdown.png      — the 4 input streams (hexes/verts/edges/scalars)
  ch1_fig2_proj_layer_shared.png    — same matrix applied to all 19 hexes (weight sharing)
  ch1_fig3_forward_pass.png         — forward pass: raw -> projection -> 32-dim features
  ch1_fig4_backward_pass.png        — backward pass: 19 grad contributions summed into one update
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent


# ---------- Figure 1: state breakdown ----------
def fig1_state_breakdown():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 95, "The Catan game state, broken into 4 streams",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(50, 90, "Each stream has its own raw width and its own count.",
            ha="center", fontsize=11, color="#555")

    streams = [
        # (y, name, count, raw, color, edge, blurb)
        (75, "Hexes",    19, 8,  "#ff9933", "#a86b00",
         "resource one-hot, dice token, robber flag, …"),
        (58, "Vertices", 54, 13, "#3399cc", "#1a4f7a",
         "settlement / city flags, port info, who-owns, …"),
        (41, "Edges",    72, 6,  "#aa66cc", "#5a1a8a",
         "road flags, who-owns, …"),
        (24, "Scalars",  1,  59, "#888888", "#333",
         "current player, VP per player, dev cards, longest road, …"),
    ]
    for y, name, count, raw, fc, ec, blurb in streams:
        # Box on the left with the count and raw width
        ax.add_patch(mpatches.FancyBboxPatch((4, y - 5.5), 22, 11,
                                              boxstyle="round,pad=0.3",
                                              edgecolor=ec, facecolor=fc, alpha=0.25, linewidth=2))
        ax.text(15, y + 2.5, name, ha="center", fontsize=13,
                fontweight="bold", color=ec)
        ax.text(15, y - 0.2, f"count = {count}", ha="center", fontsize=10,
                family="monospace")
        ax.text(15, y - 2.8, f"raw width = {raw}", ha="center", fontsize=10,
                family="monospace")

        # Arrow + blurb on the right
        ax.annotate("", xy=(46, y), xytext=(27, y),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))
        ax.text(48, y + 1.2, f"shape: [{count} × {raw}]", fontsize=10,
                family="monospace", color=ec, fontweight="bold")
        ax.text(48, y - 1.6, blurb, fontsize=10, color="#444", style="italic")

    fig.savefig(OUT / "ch1_fig1_state_breakdown.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote ch1_fig1_state_breakdown.png")


# ---------- Figure 2: shared projection matrix across 19 hexes ----------
def fig2_proj_layer_shared():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 95, "proj_hex: one shared 8 × 32 matrix, applied to every hex",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(50, 90, "Same weights for hex 0, hex 1, …, hex 18. Only the inputs differ.",
            ha="center", fontsize=11, color="#555")

    # Draw a single shared W matrix in the middle
    cx, cy = 50, 55
    ax.add_patch(mpatches.Rectangle((cx - 6, cy - 9), 12, 18,
                                     facecolor="#ffe9b0", edgecolor="#a86b00",
                                     linewidth=2.5))
    ax.text(cx, cy + 11, "proj_hex W", ha="center", fontsize=12,
            fontweight="bold", color="#a86b00", family="monospace")
    ax.text(cx, cy + 0.5, "8 × 32", ha="center", fontsize=14,
            family="monospace")
    ax.text(cx, cy - 3, "(256 weights)", ha="center", fontsize=9,
            color="#555")
    ax.text(cx, cy - 11.5, "ONE matrix\n(shared across all 19 hexes)",
            ha="center", fontsize=10, color="#a86b00", fontweight="bold")

    # Draw raw inputs on the left and projected outputs on the right.
    # We render 5 representative hexes (0, 1, 2, …, 18).
    sample_indices = [0, 1, 2, 17, 18]
    label_strings = ["hex 0", "hex 1", "hex 2", "...", "hex 18"]
    n = len(sample_indices)
    y_top, y_bot = 80, 25
    ys = np.linspace(y_top, y_bot, n)

    for i, (yi, lab) in enumerate(zip(ys, label_strings)):
        # Input box
        ax.add_patch(mpatches.Rectangle((6, yi - 2), 14, 4,
                                         facecolor="#ffd089", edgecolor="#a86b00",
                                         linewidth=1.2))
        ax.text(13, yi, f"{lab}: [8 raw]", ha="center", va="center",
                fontsize=9, family="monospace")

        # Output box
        ax.add_patch(mpatches.Rectangle((78, yi - 2), 16, 4,
                                         facecolor="#ffeec5", edgecolor="#a86b00",
                                         linewidth=1.2))
        ax.text(86, yi, f"out_{sample_indices[i]}: [32]",
                ha="center", va="center", fontsize=9, family="monospace")

        # Arrow into the matrix
        ax.annotate("", xy=(cx - 6, cy + (yi - cy) * 0.25),
                    xytext=(20, yi),
                    arrowprops=dict(arrowstyle="->", color="#a86b00",
                                    lw=1.0, alpha=0.7))
        # Arrow out of the matrix
        ax.annotate("", xy=(78, yi),
                    xytext=(cx + 6, cy + (yi - cy) * 0.25),
                    arrowprops=dict(arrowstyle="->", color="#a86b00",
                                    lw=1.0, alpha=0.7))

    # Bottom equation
    ax.text(50, 8, r"out$_i$ = raw$_i$ @ W + b   (for every hex i)",
            ha="center", fontsize=12, family="monospace", color="#222")
    ax.text(50, 4, "→ stacked together: hex_x = [19 × 32]",
            ha="center", fontsize=11, color="#555", style="italic")

    fig.savefig(OUT / "ch1_fig2_proj_layer_shared.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote ch1_fig2_proj_layer_shared.png")


# ---------- Figure 3: forward pass ----------
def fig3_forward_pass():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 95, "Forward pass — one batched matmul per stream",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(50, 90, "Forward direction →", ha="center", fontsize=11,
            color="#0a7d0a", fontweight="bold")

    streams = [
        # (y, name, count, raw, color, edge)
        (74, "hex_raw",    19, 8,  "#ff9933", "#a86b00", "proj_hex"),
        (56, "vert_raw",   54, 13, "#3399cc", "#1a4f7a", "proj_vertex"),
        (38, "edge_raw",   72, 6,  "#aa66cc", "#5a1a8a", "proj_edge"),
        (20, "scalars_raw", 1, 59, "#888888", "#333",   "proj_scalars"),
    ]
    for y, name, count, raw, fc, ec, w_name in streams:
        # Raw input
        ax.add_patch(mpatches.Rectangle((4, y - 3), 18, 6,
                                         facecolor=fc, edgecolor=ec, alpha=0.3, linewidth=1.5))
        ax.text(13, y + 0.5, name, ha="center", fontsize=10,
                fontweight="bold", color=ec, family="monospace")
        ax.text(13, y - 1.8, f"[{count} × {raw}]",
                ha="center", fontsize=9, family="monospace")

        # Matrix in middle
        ax.add_patch(mpatches.Rectangle((36, y - 3), 22, 6,
                                         facecolor="white", edgecolor=ec, linewidth=2))
        ax.text(47, y + 0.5, f"{w_name}", ha="center", fontsize=10,
                fontweight="bold", color=ec, family="monospace")
        ax.text(47, y - 1.8, f"W: [{raw} × 32]   b: [32]",
                ha="center", fontsize=9, family="monospace")

        # Output
        ax.add_patch(mpatches.Rectangle((72, y - 3), 22, 6,
                                         facecolor=fc, edgecolor=ec, alpha=0.5, linewidth=1.5))
        out_name = name.replace("_raw", "_x") if "scalars" not in name else "scalars_x"
        ax.text(83, y + 0.5, out_name, ha="center", fontsize=10,
                fontweight="bold", color=ec, family="monospace")
        ax.text(83, y - 1.8, f"[{count} × 32]",
                ha="center", fontsize=9, family="monospace")

        # Arrows (forward direction, green)
        ax.annotate("", xy=(36, y), xytext=(22, y),
                    arrowprops=dict(arrowstyle="->", color="#0a7d0a", lw=2))
        ax.annotate("", xy=(72, y), xytext=(58, y),
                    arrowprops=dict(arrowstyle="->", color="#0a7d0a", lw=2))

    # Equation footer
    ax.text(50, 6, "out = raw @ W + b   (one batched matmul, all elements at once)",
            ha="center", fontsize=11, family="monospace", color="#222")

    fig.savefig(OUT / "ch1_fig3_forward_pass.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote ch1_fig3_forward_pass.png")


# ---------- Figure 4: backward pass ----------
def fig4_backward_pass():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 95,
            "Backward pass — 19 gradient contributions, summed into ONE update",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(50, 90, "← Backward direction (gradients flow upstream)",
            ha="center", fontsize=11, color="#b00", fontweight="bold")

    # Right side: 19 incoming gradient vectors (we draw 5 reps + ellipsis)
    rep_indices = [0, 1, 2, 17, 18]
    rep_labels  = ["hex 0", "hex 1", "hex 2", "...", "hex 18"]
    ys = np.linspace(82, 22, len(rep_indices))
    for i, (yi, lab, idx) in enumerate(zip(ys, rep_labels, rep_indices)):
        ax.add_patch(mpatches.Rectangle((78, yi - 2.2), 18, 4.4,
                                         facecolor="#ffe1e1", edgecolor="#b00",
                                         linewidth=1.2))
        ax.text(87, yi + 0.4, f"dL/d(out_{idx})",
                ha="center", fontsize=9, family="monospace", color="#b00")
        ax.text(87, yi - 1.4, "[32]",
                ha="center", fontsize=8, family="monospace", color="#555")

    # Middle: each contributes raw_i.T @ dL/d(out_i)
    cx, cy = 50, 55
    ax.add_patch(mpatches.FancyBboxPatch((cx - 14, cy - 16), 28, 32,
                                          boxstyle="round,pad=0.3",
                                          facecolor="#fff5f5", edgecolor="#b00", linewidth=2))
    ax.text(cx, cy + 13.5, "Gradient accumulation",
            ha="center", fontsize=12, fontweight="bold", color="#b00")
    ax.text(cx, cy + 9, "for each hex i:", ha="center", fontsize=10,
            family="monospace", color="#222")
    ax.text(cx, cy + 5.5,
            "dW_i = raw_i.T @ dL/d(out_i)",
            ha="center", fontsize=10, family="monospace", color="#222")
    ax.text(cx, cy + 2.5,
            "(shape: 8 × 32)", ha="center", fontsize=9,
            family="monospace", color="#555")
    ax.text(cx, cy - 2, "then sum over all 19 hexes:",
            ha="center", fontsize=10, family="monospace", color="#222")
    ax.text(cx, cy - 6.5,
            r"dL/dW  =  $\Sigma_i$  dW$_i$",
            ha="center", fontsize=14, color="#222")
    ax.text(cx, cy - 11, "(shape: 8 × 32)",
            ha="center", fontsize=9, family="monospace", color="#555")
    ax.text(cx, cy - 13.5, "ONE update per step",
            ha="center", fontsize=10, color="#b00", fontweight="bold")

    # Left side: shared W after update
    ax.add_patch(mpatches.Rectangle((6, cy - 10), 18, 20,
                                     facecolor="#ffe9b0", edgecolor="#a86b00",
                                     linewidth=2.5))
    ax.text(15, cy + 11.5, "proj_hex W", ha="center", fontsize=11,
            fontweight="bold", color="#a86b00", family="monospace")
    ax.text(15, cy + 1, "8 × 32", ha="center", fontsize=14,
            family="monospace")
    ax.text(15, cy - 12.5, "W ← W − lr × dL/dW",
            ha="center", fontsize=10, family="monospace", color="#a86b00",
            fontweight="bold")

    # Arrows: gradient inflow (right -> middle)
    for yi in ys:
        ax.annotate("", xy=(cx + 14, cy + (yi - cy) * 0.3),
                    xytext=(78, yi),
                    arrowprops=dict(arrowstyle="->", color="#b00",
                                    lw=1.0, alpha=0.7))
    # Arrow: middle -> matrix (one update, big arrow)
    ax.annotate("", xy=(24, cy), xytext=(cx - 14, cy),
                arrowprops=dict(arrowstyle="->", color="#b00", lw=3))
    ax.text((24 + cx - 14) / 2, cy + 2.5, "applied once",
            ha="center", fontsize=10, color="#b00", fontweight="bold")

    fig.savefig(OUT / "ch1_fig4_backward_pass.png", dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote ch1_fig4_backward_pass.png")


def main():
    fig1_state_breakdown()
    fig2_proj_layer_shared()
    fig3_forward_pass()
    fig4_backward_pass()
    print(f"\nAll visuals in {OUT}")


if __name__ == "__main__":
    main()
