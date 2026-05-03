"""Two visualizations for the v3 GNN.

1. visualize_architecture.png — boxes-and-arrows of the model layers.
2. visualize_graph.png — the heterogeneous game graph (19 hexes, 54 verts,
   72 edges) with message-passing edges colored by direction.

Both use matplotlib so we can serve PNGs from the dashboard or the
playback HTTP server.

Usage:
  python scripts/visualize_gnn.py [--out-dir DIR] [--hidden-dim N] [--num-layers N]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_architecture(out_path: Path, hidden_dim: int, num_layers: int) -> None:
    """Boxes-and-arrows of the model layers, with widths proportional to
    parameter counts so you can SEE the bottleneck visually."""
    from catan_gnn.gnn_model import GnnModel
    m = GnnModel(hidden_dim=hidden_dim, num_layers=num_layers)
    m.eval()

    # Group params by region and compute group-totals.
    groups = {
        "Input proj": 0,
        "SAGEConv layers": 0,
        "Body final": 0,
        "Value head": 0,
        "Policy head": 0,
    }
    for name, p in m.named_parameters():
        n = p.numel()
        if "proj_" in name: groups["Input proj"] += n
        elif "convs" in name: groups["SAGEConv layers"] += n
        elif "body.final" in name: groups["Body final"] += n
        elif "value_head" in name: groups["Value head"] += n
        elif "policy_head" in name: groups["Policy head"] += n
    total = sum(groups.values())

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    title = f"v3 GNN architecture — h{hidden_dim} × l{num_layers} = {total:,} params"
    ax.text(50, 96, title, ha="center", fontsize=15, fontweight="bold")

    # Three columns: input, body, heads.
    # Column 1: inputs
    inputs = [
        ("Hex features\n[19 × 8]", "#ffd089"),
        ("Vertex features\n[54 × 13]", "#a4d4f7"),
        ("Edge features\n[72 × 6]", "#d4a8ff"),
        ("Scalars\n[59]", "#cccccc"),
    ]
    for i, (label, color) in enumerate(inputs):
        y = 75 - i * 14
        rect = mpatches.FancyBboxPatch((3, y - 4), 16, 8,
                                        boxstyle="round,pad=0.3",
                                        edgecolor="black", facecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(11, y, label, ha="center", va="center", fontsize=9)

    # Column 2: input projection block
    proj_h = 60 / 4  # 4 box width per group
    proj_x = 27
    ax.add_patch(mpatches.FancyBboxPatch((proj_x, 12), 14, 70,
                                          boxstyle="round,pad=0.3",
                                          edgecolor="#444", facecolor="#fff5e0", linewidth=1.5))
    ax.text(proj_x + 7, 78, "Input projections", ha="center", fontsize=10, fontweight="bold")
    ax.text(proj_x + 7, 73, f"4 × Linear(F → {hidden_dim})", ha="center", fontsize=8)
    ax.text(proj_x + 7, 50, f"Each input modality\nprojects to a\n{hidden_dim}-dim embedding", ha="center", fontsize=9, color="#666")
    ax.text(proj_x + 7, 18, f"{groups['Input proj']:,} params", ha="center", fontsize=9, color="#444", style="italic")

    # Column 3: SAGEConv block (size grows with layer count and hidden_dim)
    sage_x = 47
    sage_w = 14
    ax.add_patch(mpatches.FancyBboxPatch((sage_x, 12), sage_w, 70,
                                          boxstyle="round,pad=0.3",
                                          edgecolor="#444",
                                          facecolor="#e0f5e0",
                                          linewidth=1.5))
    ax.text(sage_x + sage_w/2, 78, "SAGEConv body", ha="center", fontsize=10, fontweight="bold")
    ax.text(sage_x + sage_w/2, 73, f"{num_layers} × HeteroConv blocks", ha="center", fontsize=8)
    # Show each conv layer
    inner_layers_top = 67
    inner_layers_bot = 25
    inner_h = (inner_layers_top - inner_layers_bot) / max(num_layers, 1)
    for i in range(num_layers):
        y = inner_layers_top - (i + 0.5) * inner_h
        ax.add_patch(mpatches.FancyBboxPatch((sage_x + 1.5, y - inner_h/3),
                                              sage_w - 3, inner_h * 0.6,
                                              boxstyle="round,pad=0.1",
                                              edgecolor="#2a8", facecolor="#c4e8c4"))
        ax.text(sage_x + sage_w/2, y, f"layer {i+1}\n{hidden_dim}→{hidden_dim}",
                ha="center", va="center", fontsize=8)
    ax.text(sage_x + sage_w/2, 18, f"{groups['SAGEConv layers']:,} params",
            ha="center", fontsize=9, color="#444", style="italic")

    # Column 4: body final
    bf_x = 67
    bf_w = 8
    ax.add_patch(mpatches.FancyBboxPatch((bf_x, 30), bf_w, 30,
                                          boxstyle="round,pad=0.3",
                                          edgecolor="#444", facecolor="#f0e0ff", linewidth=1.5))
    ax.text(bf_x + bf_w/2, 56, "Body\nfinal", ha="center", fontsize=10, fontweight="bold")
    ax.text(bf_x + bf_w/2, 47, f"({4*hidden_dim}+27)\n→ 128", ha="center", fontsize=8)
    ax.text(bf_x + bf_w/2, 35, f"{groups['Body final']:,}\nparams", ha="center", fontsize=8, style="italic")

    # Column 5: heads
    val_x = 82
    val_w = 14
    # value head
    ax.add_patch(mpatches.FancyBboxPatch((val_x, 50), val_w, 18,
                                          boxstyle="round,pad=0.3",
                                          edgecolor="#444", facecolor="#ffe0e0", linewidth=1.5))
    ax.text(val_x + val_w/2, 64, "Value head", ha="center", fontsize=10, fontweight="bold")
    ax.text(val_x + val_w/2, 59, "128 → 64 → 4", ha="center", fontsize=9)
    ax.text(val_x + val_w/2, 53, f"{groups['Value head']:,} params", ha="center", fontsize=8, style="italic")

    # policy head
    ax.add_patch(mpatches.FancyBboxPatch((val_x, 22), val_w, 22,
                                          boxstyle="round,pad=0.3",
                                          edgecolor="#444", facecolor="#fff0c4", linewidth=2))
    ax.text(val_x + val_w/2, 39, "Policy head", ha="center", fontsize=10, fontweight="bold")
    ax.text(val_x + val_w/2, 34, "128 → 280", ha="center", fontsize=9)
    ax.text(val_x + val_w/2, 28, f"{groups['Policy head']:,} params", ha="center", fontsize=8, style="italic")
    ax.text(val_x + val_w/2, 24, "← bottleneck", ha="center", fontsize=8, color="#a44", fontweight="bold")

    # Arrows between columns
    arrows = [
        (19, 50, 27, 50),      # inputs → proj
        (41, 50, 47, 50),      # proj → sage
        (61, 50, 67, 50),      # sage → body
        (75, 50, 82, 60),      # body → value head
        (75, 50, 82, 33),      # body → policy head
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))

    # Legend / param share bar at bottom
    bar_y = 4
    bar_x = 5
    bar_w = 90
    bar_h = 4
    cum = 0
    colors = {"Input proj": "#fff5e0", "SAGEConv layers": "#e0f5e0",
              "Body final": "#f0e0ff", "Value head": "#ffe0e0", "Policy head": "#fff0c4"}
    for label in ["Input proj", "SAGEConv layers", "Body final", "Value head", "Policy head"]:
        frac = groups[label] / total
        w = bar_w * frac
        ax.add_patch(mpatches.Rectangle((bar_x + cum, bar_y), w, bar_h,
                                         facecolor=colors[label], edgecolor="black"))
        if frac > 0.05:
            ax.text(bar_x + cum + w/2, bar_y + bar_h/2,
                    f"{label}\n{frac*100:.0f}%",
                    ha="center", va="center", fontsize=8)
        cum += w
    ax.text(50, 0, "← parameter share by component →", ha="center", fontsize=9, color="#666")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_message_passing_graph(out_path: Path) -> None:
    """Visualize the actual heterogeneous Catan graph the GNN operates on:
    19 hex nodes (orange), 54 vertex nodes (blue), 72 edge nodes (purple)
    with the message-passing connections."""
    import networkx as nx
    # Hard-coded coordinates from the playback layout.
    # Hex centers: 19 hexes in 5 rows.
    hex_rows = [3, 4, 5, 4, 3]
    hex_centers = []
    y0 = 4
    for r, n in enumerate(hex_rows):
        y = y0 - r
        x_offset = -(n - 1) / 2.0
        for i in range(n):
            hex_centers.append((x_offset + i, y))
    assert len(hex_centers) == 19
    # Vertex positions: 54 vertices roughly distributed around hexes.
    # Use a hex-vertex arrangement: each hex has 6 corners. We'll synthesize
    # vertex positions as the average of adjacent hex centers, approximately.
    # For simplicity: place vertices in 6 rows.
    vert_rows = [7, 8, 9, 10, 9, 8, 7] if False else None  # complicated
    # Simpler: arrange vertices in a circle around each hex (visually approximate).
    import math
    vertex_xy = {}
    for v in range(54):
        # Spread vertices radially.
        angle = 2 * math.pi * v / 54
        r = 5.5
        vertex_xy[v] = (r * math.cos(angle), r * math.sin(angle))
    edge_xy = {}
    for ed in range(72):
        angle = 2 * math.pi * ed / 72
        r = 7
        edge_xy[ed] = (r * math.cos(angle), r * math.sin(angle))

    # Read observation features to get actual hex/vertex/edge incidence.
    # To keep this simple, we'll just show the NODE TYPES with colored dots
    # and connect them with light edges showing message-passing direction.
    G = nx.Graph()
    pos = {}
    colors = []
    sizes = []
    labels = {}

    for h, (x, y) in enumerate(hex_centers):
        node = f"H{h}"
        G.add_node(node)
        pos[node] = (x * 1.5, y * 1.5)
        colors.append("#ff9933")
        sizes.append(220)
        labels[node] = str(h)

    # Use a more structured hex-corner layout for vertices.
    # For each hex, 6 corner positions at 60-degree increments.
    v_count = 0
    seen_vertices = {}
    for h, (cx, cy) in enumerate(hex_centers):
        for k in range(6):
            ang = math.pi/6 + k * math.pi/3
            vx = cx * 1.5 + 0.7 * math.cos(ang)
            vy = cy * 1.5 + 0.7 * math.sin(ang)
            # Round to combine shared corners
            key = (round(vx, 1), round(vy, 1))
            if key not in seen_vertices and v_count < 54:
                seen_vertices[key] = v_count
                node = f"V{v_count}"
                G.add_node(node)
                pos[node] = (vx, vy)
                colors.append("#3399cc")
                sizes.append(60)
                labels[node] = ""
                v_count += 1

    # Edges as midpoints between adjacent vertices in each hex.
    e_count = 0
    seen_edges = {}
    for h, (cx, cy) in enumerate(hex_centers):
        for k in range(6):
            ang1 = math.pi/6 + k * math.pi/3
            ang2 = math.pi/6 + ((k+1) % 6) * math.pi/3
            ex = cx * 1.5 + 0.7 * (math.cos(ang1) + math.cos(ang2)) / 2
            ey = cy * 1.5 + 0.7 * (math.sin(ang1) + math.sin(ang2)) / 2
            key = (round(ex, 1), round(ey, 1))
            if key not in seen_edges and e_count < 72:
                seen_edges[key] = e_count
                node = f"E{e_count}"
                G.add_node(node)
                pos[node] = (ex, ey)
                colors.append("#aa66cc")
                sizes.append(30)
                labels[node] = ""
                e_count += 1

    # Add representative message-passing edges (just hex-vert and vert-edge).
    # Pick one example pattern.
    g_edges = []
    for h_idx, (cx, cy) in enumerate(hex_centers):
        for k in range(6):
            ang = math.pi/6 + k * math.pi/3
            vx = cx * 1.5 + 0.7 * math.cos(ang)
            vy = cy * 1.5 + 0.7 * math.sin(ang)
            key = (round(vx, 1), round(vy, 1))
            if key in seen_vertices:
                v_idx = seen_vertices[key]
                G.add_edge(f"H{h_idx}", f"V{v_idx}")

    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color="#888", width=0.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, ax=ax,
                            edgecolors="black", linewidths=0.6)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

    legend_handles = [
        mpatches.Patch(color="#ff9933", label=f"Hex node (19) — F=8"),
        mpatches.Patch(color="#3399cc", label=f"Vertex node (54) — F=13"),
        mpatches.Patch(color="#aa66cc", label=f"Edge node (72) — F=6"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=11, frameon=True)

    ax.set_title("Heterogeneous Catan graph the GNN operates on\n"
                 "(message passing: hex↔vertex, vertex↔edge)",
                 fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_input_routing(out_path: Path, hidden_dim: int = 32) -> None:
    """Detailed view of game-state → input projections only.

    For each input modality, show:
    - The raw observation it comes from (engine field name + shape)
    - What each feature dimension means (resource one-hot, ports, etc.)
    - The Linear projection that takes it to hidden_dim
    - Where it goes next in the GNN

    No SAGEConv body, no heads — just the data routing layer.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    title = f"Game state → Input projections (hidden_dim={hidden_dim})"
    subtitle = "What each feature is and where it goes BEFORE the GNN body"
    ax.text(50, 96, title, ha="center", fontsize=15, fontweight="bold")
    ax.text(50, 92, subtitle, ha="center", fontsize=11, color="#666")

    # 4 input rows: hex / vertex / edge / scalars
    # Each row: [raw observation field] → [feature breakdown] → [Linear] → [hidden_dim output]

    rows = [
        {
            "y": 75,
            "color": "#ffd089",
            "stroke": "#a86b00",
            "name": "HEX",
            "shape": "[19 × 8]",
            "src": "obs.hex_features",
            "engine_src": "build_observation()\nin observation.rs",
            "feature_breakdown": [
                ("dim 0-4", "resource one-hot (Wd/Br/Sh/Wh/Or)"),
                ("dim 5",   "dice-roll number (normalized 0-1)"),
                ("dim 6",   "robber bit (1 if robber here)"),
                ("dim 7",   "desert bit"),
            ],
            "input_dim": 8,
            "n_nodes": 19,
        },
        {
            "y": 53,
            "color": "#a4d4f7",
            "stroke": "#1a4f7a",
            "name": "VERTEX",
            "shape": "[54 × 13]",
            "src": "obs.vertex_features",
            "engine_src": "(perspective-rotated\nper player)",
            "feature_breakdown": [
                ("dim 0",   "empty / not-built"),
                ("dim 1",   "settlement"),
                ("dim 2",   "city"),
                ("dim 3-6", "owner one-hot (P0..P3, perspective-rotated)"),
                ("dim 7-12", "port flags (3:1, Wd, Br, Sh, Wh, Or)"),
            ],
            "input_dim": 13,
            "n_nodes": 54,
        },
        {
            "y": 31,
            "color": "#d4a8ff",
            "stroke": "#5a1a8a",
            "name": "EDGE",
            "shape": "[72 × 6]",
            "src": "obs.edge_features",
            "engine_src": "(perspective-rotated)",
            "feature_breakdown": [
                ("dim 0",   "empty / no road"),
                ("dim 1",   "road built"),
                ("dim 2-5", "owner one-hot (P0..P3, perspective-rotated)"),
            ],
            "input_dim": 6,
            "n_nodes": 72,
        },
        {
            "y": 9,
            "color": "#cccccc",
            "stroke": "#444",
            "name": "SCALARS",
            "shape": "[59]",
            "src": "obs.scalars",
            "engine_src": "(see observation.rs § Scalar layout)",
            "feature_breakdown": [
                ("dim 0-3",   "VP per player (perspective-rotated)"),
                ("dim 4-23",  "hands per player [4×5] (perspective)"),
                ("dim 24-25", "longest-road / largest-army holders"),
                ("dim 26-29", "longest_road_length per player"),
                ("dim 30-34", "knights_played per player"),
                ("dim 35-49", "settlements/cities/roads built [3×5]"),
                ("dim 50-55", "current player one-hot + phase one-hot"),
                ("dim 56-58", "v3 flags + dice roll history"),
            ],
            "input_dim": 59,
            "n_nodes": 1,
        },
    ]

    for row in rows:
        y = row["y"]
        # Box 1: source observation
        bx, bw = 2, 17
        ax.add_patch(mpatches.FancyBboxPatch((bx, y - 6), bw, 12,
                                              boxstyle="round,pad=0.4",
                                              edgecolor=row["stroke"],
                                              facecolor=row["color"],
                                              linewidth=2))
        ax.text(bx + bw/2, y + 4, row["name"], ha="center", fontsize=12, fontweight="bold")
        ax.text(bx + bw/2, y + 1.3, row["shape"], ha="center", fontsize=10, family="monospace")
        ax.text(bx + bw/2, y - 1.3, row["src"], ha="center", fontsize=8, color="#444", family="monospace")
        ax.text(bx + bw/2, y - 4, row["engine_src"], ha="center", fontsize=7, color="#777", style="italic")

        # Box 2: feature breakdown
        bx2, bw2 = 24, 38
        ax.add_patch(mpatches.FancyBboxPatch((bx2, y - 7), bw2, 14,
                                              boxstyle="round,pad=0.3",
                                              edgecolor="#999", facecolor="white",
                                              linewidth=1))
        ax.text(bx2 + 1, y + 5, "Per-node feature meaning:", fontsize=8.5,
                fontweight="bold", color="#444")
        n_lines = len(row["feature_breakdown"])
        line_h = 11 / max(n_lines, 1)
        for i, (rng, desc) in enumerate(row["feature_breakdown"]):
            yy = y + 4 - (i + 0.5) * line_h
            ax.text(bx2 + 1, yy, rng, fontsize=8, family="monospace", color="#555")
            ax.text(bx2 + 8, yy, desc, fontsize=8, color="#222")

        # Box 3: Linear projection
        bx3, bw3 = 65, 14
        ax.add_patch(mpatches.FancyBboxPatch((bx3, y - 5), bw3, 10,
                                              boxstyle="round,pad=0.3",
                                              edgecolor="#444", facecolor="#fff5e0",
                                              linewidth=2))
        ax.text(bx3 + bw3/2, y + 2.5, f"Linear", ha="center", fontsize=10, fontweight="bold")
        ax.text(bx3 + bw3/2, y + 0.2, f"{row['input_dim']} → {hidden_dim}",
                ha="center", fontsize=10, family="monospace", color="#a44")
        n_params = (row["input_dim"] + 1) * hidden_dim  # +1 for bias
        ax.text(bx3 + bw3/2, y - 2.5, f"{n_params:,} params",
                ha="center", fontsize=8, style="italic", color="#666")

        # Box 4: hidden_dim output
        bx4, bw4 = 82, 14
        ax.add_patch(mpatches.FancyBboxPatch((bx4, y - 5), bw4, 10,
                                              boxstyle="round,pad=0.3",
                                              edgecolor="#2a8", facecolor="#d4f5d4",
                                              linewidth=2))
        ax.text(bx4 + bw4/2, y + 2.5,
                f"[{row['n_nodes']} × {hidden_dim}]" if row["n_nodes"] > 1 else f"[{hidden_dim}]",
                ha="center", fontsize=10, family="monospace", fontweight="bold")
        ax.text(bx4 + bw4/2, y - 0.2, f"→ SAGEConv\n   body" if row["name"] != "SCALARS" else "→ concat\n   into body.final",
                ha="center", fontsize=8, color="#444")

        # Arrows
        ax.annotate("", xy=(bx2, y), xytext=(bx + bw, y),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))
        ax.annotate("", xy=(bx3, y), xytext=(bx2 + bw2, y),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))
        ax.annotate("", xy=(bx4, y), xytext=(bx3 + bw3, y),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))

    ax.text(11, 88, "Raw observation",
            ha="center", fontsize=10, fontweight="bold", color="#444")
    ax.text(43, 88, "What each feature dim means",
            ha="center", fontsize=10, fontweight="bold", color="#444")
    ax.text(72, 88, "Projection",
            ha="center", fontsize=10, fontweight="bold", color="#444")
    ax.text(89, 88, "Output (to GNN body)",
            ha="center", fontsize=10, fontweight="bold", color="#444")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_sageconv_zoom(out_path: Path, hidden_dim: int = 32) -> None:
    """Three-panel zoom-in/zoom-out diagram for SAGEConv intuition.

    Panel 1 (top, zoom-in):
      - Show one source node's 32 neurons (as a column of dots)
      - Show its 32×32 W_neigh matrix as a coloured grid
      - Show how source → target via mean aggregation + matmul

    Panel 2 (middle):
      - The 4 heterogeneous message-passing directions on a small toy graph
        (one hex, its 6 vertices, the connecting edges)

    Panel 3 (bottom, zoom-out):
      - Full pipeline: input → projection → 2× SAGEConv → body → heads
        compressed into a single horizontal flow
    """
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.0, 0.5], hspace=0.25)
    ax_top = fig.add_subplot(gs[0])
    ax_mid = fig.add_subplot(gs[1])
    ax_bot = fig.add_subplot(gs[2])

    # ============================================================
    # PANEL 1 — ZOOM-IN: one SAGEConv computation
    # ============================================================
    ax = ax_top
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 96, "ZOOM IN — One SAGEConv operation: how one VERTEX node updates from its HEX neighbors",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(50, 92, "(applies the same way for every other direction: vertex↔hex, vertex↔edge, edge↔vertex)",
            ha="center", fontsize=10, color="#666", style="italic")

    H = hidden_dim  # for cleaner formula display

    # Layout: 3 hex-source nodes on left, 1 vertex-target on right
    # Each "node" is shown as a column of N=32 dots representing its 32 features.

    def draw_neuron_column(cx, cy, n=32, color="#3399cc", edge="#1a4f7a", height=42, label=""):
        """Draw a vertical column of n dots representing a node's feature vector."""
        spacing = height / n
        # Background card
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - 2.2, cy - height/2 - 2), 4.4, height + 4,
            boxstyle="round,pad=0.2",
            edgecolor=edge, facecolor="white", linewidth=1.5))
        for i in range(n):
            y = cy + height/2 - (i + 0.5) * spacing
            circle = mpatches.Circle((cx, y), 0.5, facecolor=color,
                                      edgecolor=edge, linewidth=0.4)
            ax.add_patch(circle)
        if label:
            ax.text(cx, cy + height/2 + 4, label, ha="center", fontsize=10, fontweight="bold")
        ax.text(cx, cy - height/2 - 5, f"{n} dims", ha="center", fontsize=8, color="#666")

    # 3 source hex nodes
    src_xs = [10, 10, 10]
    src_ys = [70, 50, 30]
    src_labels = ["Hex H_a\n(neighbor 1)", "Hex H_b\n(neighbor 2)", "Hex H_c\n(neighbor 3)"]
    for cx, cy, lbl in zip(src_xs, src_ys, src_labels):
        draw_neuron_column(cx, cy, n=H, color="#ff9933", edge="#a86b00", height=32, label=lbl)

    # Aggregation step: mean
    agg_x = 28
    agg_y = 50
    ax.add_patch(mpatches.FancyBboxPatch((agg_x - 5, agg_y - 22), 10, 44,
                                          boxstyle="round,pad=0.4",
                                          edgecolor="#444", facecolor="#fff5e0", linewidth=1.5))
    ax.text(agg_x, agg_y + 18, "MEAN", ha="center", fontsize=11, fontweight="bold")
    ax.text(agg_x, agg_y + 14, "(aggregator)", ha="center", fontsize=8, color="#666")
    draw_neuron_column(agg_x, agg_y - 3, n=H, color="#ffcc66", edge="#a86b00", height=24,
                        label="")
    ax.text(agg_x, agg_y - 17, "agg = avg of\nall neighbors",
            ha="center", fontsize=8, color="#444")

    # Arrows from each src to agg
    for cy in src_ys:
        ax.annotate("", xy=(agg_x - 5, agg_y), xytext=(11.5, cy),
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.8))

    # W_neigh matrix
    wn_x = 47
    wn_y = 60
    wn_size = 16
    ax.add_patch(mpatches.FancyBboxPatch((wn_x - wn_size/2, wn_y - wn_size/2), wn_size, wn_size,
                                          boxstyle="round,pad=0.2",
                                          edgecolor="#444", facecolor="white", linewidth=2))
    # Draw a stylized matrix grid
    grid_n = 8  # show 8x8 representative cells (real is 32x32)
    cell_size = (wn_size - 2) / grid_n
    np.random.seed(0)
    grid_vals = np.random.randn(grid_n, grid_n) * 0.5
    for i in range(grid_n):
        for j in range(grid_n):
            v = grid_vals[i, j]
            color_intensity = max(0, min(1, (v + 1.5) / 3))
            color = (color_intensity * 0.6 + 0.3, color_intensity * 0.7 + 0.2, 0.8)
            ax.add_patch(mpatches.Rectangle(
                (wn_x - wn_size/2 + 1 + j * cell_size,
                 wn_y - wn_size/2 + 1 + i * cell_size),
                cell_size, cell_size,
                facecolor=color, edgecolor="white", linewidth=0.3))
    ax.text(wn_x, wn_y + wn_size/2 + 2, "W_neigh", ha="center", fontsize=11, fontweight="bold")
    ax.text(wn_x, wn_y - wn_size/2 - 2, f"{H}×{H} matrix\n({H*H} weights)", ha="center", fontsize=8, color="#444")

    # Arrow from agg to W_neigh
    ax.annotate("", xy=(wn_x - wn_size/2, wn_y), xytext=(agg_x + 5, agg_y - 3),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))
    ax.text((agg_x + wn_x) / 2, 56, "matmul", ha="center", fontsize=8,
            color="#444", style="italic", rotation=20)

    # The vertex's OWN features
    self_x = 28
    self_y = 84
    draw_neuron_column(self_x, self_y, n=H, color="#3399cc", edge="#1a4f7a", height=22,
                        label="VERTEX V (target)\n— its OWN features")

    # W_self matrix
    ws_x = 47
    ws_y = 84
    ws_size = 12
    ax.add_patch(mpatches.FancyBboxPatch((ws_x - ws_size/2, ws_y - ws_size/2), ws_size, ws_size,
                                          boxstyle="round,pad=0.2",
                                          edgecolor="#444", facecolor="white", linewidth=2))
    np.random.seed(1)
    grid_vals2 = np.random.randn(grid_n, grid_n) * 0.5
    cell2 = (ws_size - 2) / grid_n
    for i in range(grid_n):
        for j in range(grid_n):
            v = grid_vals2[i, j]
            color_intensity = max(0, min(1, (v + 1.5) / 3))
            color = (0.8, color_intensity * 0.7 + 0.2, color_intensity * 0.6 + 0.3)
            ax.add_patch(mpatches.Rectangle(
                (ws_x - ws_size/2 + 1 + j * cell2,
                 ws_y - ws_size/2 + 1 + i * cell2),
                cell2, cell2,
                facecolor=color, edgecolor="white", linewidth=0.3))
    ax.text(ws_x, ws_y + ws_size/2 + 2, "W_self", ha="center", fontsize=11, fontweight="bold")
    ax.text(ws_x, ws_y - ws_size/2 - 2, f"{H}×{H}", ha="center", fontsize=8, color="#444")

    ax.annotate("", xy=(ws_x - ws_size/2, ws_y), xytext=(self_x + 2.5, self_y),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))

    # Sum + activation
    sum_x = 64
    sum_y = 72
    ax.add_patch(mpatches.Circle((sum_x, sum_y), 4, facecolor="#ffe9b3", edgecolor="#a86b00", linewidth=2))
    ax.text(sum_x, sum_y, "+", ha="center", va="center", fontsize=20, fontweight="bold")
    ax.text(sum_x, sum_y + 6, "sum + ReLU", ha="center", fontsize=9, color="#444")

    # Arrows into sum
    ax.annotate("", xy=(sum_x - 3, sum_y + 2), xytext=(ws_x + ws_size/2, ws_y),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))
    ax.annotate("", xy=(sum_x - 3, sum_y - 2), xytext=(wn_x + wn_size/2, wn_y),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))

    # Output vertex
    out_x = 84
    out_y = 72
    draw_neuron_column(out_x, out_y, n=H, color="#66cc88", edge="#1a5a3a", height=32,
                        label="VERTEX V (updated)\n— next-layer features")

    # Arrow from sum to output
    ax.annotate("", xy=(out_x - 2.5, out_y), xytext=(sum_x + 4, sum_y),
                arrowprops=dict(arrowstyle="->", color="#666", lw=2))

    # Formula (mathtext-safe)
    ax.text(50, 8, r"$v_{new} = \mathrm{ReLU}( W_{self} \cdot v_{old} + W_{neigh} \cdot \mathrm{mean}(\mathrm{neighbors}) )$",
            ha="center", fontsize=14)
    ax.text(50, 4, f"Total weights for ONE direction: 2 × ({H}×{H}) + bias = {2*H*H + H:,} learnable params",
            ha="center", fontsize=10, color="#444")

    # ============================================================
    # PANEL 2 — MIDDLE: 4 message-passing directions on a toy graph
    # ============================================================
    ax = ax_mid
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 95, "ZOOM OUT (a bit) — All 4 message-passing directions on a tiny toy graph",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(50, 90, "Each colored arrow is a SEPARATE 32×32 W_neigh matrix (4 directions × num_layers = 8 matrices for L=2)",
            ha="center", fontsize=10, color="#666", style="italic")

    # Tiny toy graph: 1 hex at center, 6 vertices around, 6 edges connecting them
    import math
    hex_cx, hex_cy = 50, 50
    n_corners = 6
    R = 14
    vert_pos = []
    for k in range(n_corners):
        ang = math.pi / 2 + k * math.pi / 3
        vx = hex_cx + R * math.cos(ang)
        vy = hex_cy + R * math.sin(ang)
        vert_pos.append((vx, vy))
    edge_pos = []
    for k in range(n_corners):
        v1 = vert_pos[k]
        v2 = vert_pos[(k + 1) % n_corners]
        edge_pos.append(((v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2))

    # Draw the static graph (light)
    # hex
    ax.add_patch(mpatches.Circle((hex_cx, hex_cy), 3.5, facecolor="#ff9933",
                                  edgecolor="#a86b00", linewidth=2, zorder=5))
    ax.text(hex_cx, hex_cy, "H", ha="center", va="center", fontsize=12,
            fontweight="bold", zorder=6)
    # vertices
    for i, (vx, vy) in enumerate(vert_pos):
        ax.add_patch(mpatches.Circle((vx, vy), 2.5, facecolor="#3399cc",
                                      edgecolor="#1a4f7a", linewidth=1.5, zorder=5))
        ax.text(vx, vy, f"V{i}", ha="center", va="center", fontsize=8, color="white",
                fontweight="bold", zorder=6)
    # edges
    for i, (ex, ey) in enumerate(edge_pos):
        ax.add_patch(mpatches.Circle((ex, ey), 1.6, facecolor="#aa66cc",
                                      edgecolor="#5a1a8a", linewidth=1, zorder=5))

    # Show the 4 directions as colored arrow patches around the graph
    direction_colors = {
        "hex → vertex": "#d62728",
        "vertex → hex": "#1f77b4",
        "vertex → edge": "#2ca02c",
        "edge → vertex": "#9467bd",
    }

    # Direction 1: hex → vertex (red, from H to each V)
    for vx, vy in vert_pos:
        ax.annotate("", xy=(vx - 1.5 * (vx-hex_cx)/R, vy - 1.5 * (vy-hex_cy)/R),
                    xytext=(hex_cx + 4 * (vx-hex_cx)/R, hex_cy + 4 * (vy-hex_cy)/R),
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5, alpha=0.6))

    # Direction 2: vertex → edge (green, only show 2)
    for k in [0, 2]:
        v1 = vert_pos[k]
        e_pos = edge_pos[k]
        ax.annotate("", xy=(e_pos[0] - 0.7 * (e_pos[0]-v1[0]),
                             e_pos[1] - 0.7 * (e_pos[1]-v1[1])),
                    xytext=(v1[0] + 0.5 * (e_pos[0]-v1[0]),
                             v1[1] + 0.5 * (e_pos[1]-v1[1])),
                    arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5, alpha=0.6))

    # Legend
    leg_x, leg_y = 5, 55
    for i, (label, color) in enumerate(direction_colors.items()):
        yy = leg_y - i * 7
        ax.add_patch(mpatches.Rectangle((leg_x, yy - 1), 6, 2, facecolor=color, alpha=0.7))
        ax.text(leg_x + 8, yy, label, fontsize=10, va="center")
    ax.text(leg_x, leg_y + 7, "4 direction matrices:", fontsize=11, fontweight="bold")
    ax.text(leg_x, leg_y - 30, f"Each = {H}×{H} weights", fontsize=9, color="#444",
            style="italic")
    ax.text(leg_x, leg_y - 33, f"× 2 layers = 8 total", fontsize=9, color="#444",
            style="italic")
    ax.text(leg_x, leg_y - 36, f"≈ {8 * H * H:,} weights\nfor SAGEConv body",
            fontsize=9, color="#444", style="italic")

    # Side note about scale
    ax.text(72, 50, "Toy graph: 1 hex + 6 verts + 6 edges\n"
                    "Real graph: 19 + 54 + 72 nodes\n"
                    "(150× more node count)\n\n"
                    "But weights are SHARED across\n"
                    "all nodes of same type — so the\n"
                    "8 matrices are the WHOLE body.",
            fontsize=10, color="#444", style="italic", va="center")

    # ============================================================
    # PANEL 3 — BOTTOM: full pipeline zoom-out
    # ============================================================
    ax = ax_bot
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 95, "ZOOM OUT (full) — Where SAGEConv sits in the full pipeline",
            ha="center", fontsize=13, fontweight="bold")

    boxes = [
        (4, "Game state\n[19 hex + 54 vert\n+ 72 edge + 59 scalar]", "#cccccc"),
        (22, "Input\nProjections\n4× Linear", "#fff5e0"),
        (40, "SAGEConv\n× 2 layers\n8 matrices", "#e0f5e0"),
        (58, "Body final\nLinear\n→ 128", "#f0e0ff"),
        (76, "Value head\n→ 4", "#ffe0e0"),
    ]
    for cx, label, color in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((cx, 30), 14, 40,
                                              boxstyle="round,pad=0.3",
                                              edgecolor="#444", facecolor=color,
                                              linewidth=1.5))
        ax.text(cx + 7, 50, label, ha="center", va="center", fontsize=10)

    ax.add_patch(mpatches.FancyBboxPatch((76, 8), 14, 18,
                                          boxstyle="round,pad=0.3",
                                          edgecolor="#444", facecolor="#fff0c4",
                                          linewidth=2))
    ax.text(83, 17, "Policy head\n→ 280", ha="center", va="center", fontsize=10, fontweight="bold")

    # arrows
    for cx in [4, 22, 40, 58]:
        ax.annotate("", xy=(cx + 16, 50), xytext=(cx + 14.5, 50),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))
    # body → policy head (down arrow)
    ax.annotate("", xy=(83, 26), xytext=(72 + 14, 30),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))

    # Highlight where the zoom-in panel lives
    ax.add_patch(mpatches.Rectangle((40, 30), 14, 40, facecolor="none",
                                     edgecolor="#d62728", linewidth=2.5, linestyle="--"))
    ax.text(47, 76, "← we just zoomed into here ↑",
            ha="center", fontsize=10, color="#d62728", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_input_to_sageconv(out_path: Path, hidden_dim: int = 32) -> None:
    """Show ONLY the handoff from input projections to SAGEConv body.

    NOT what's inside SAGEConv. Just: where do the projection outputs
    physically connect into the graph that SAGEConv operates on.

    The point: each row of the output tensor becomes one node's starting
    feature vector. No additional learned weights between them.
    """
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    H = hidden_dim

    ax.text(50, 96, f"Input projections → SAGEConv body — the handoff",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(50, 92, f"Each row of the projected tensor becomes ONE node's starting features. "
                    "No extra learned weights between these.",
            ha="center", fontsize=11, color="#555")

    # ============== LEFT COLUMN: projection output tensors ==============
    # Show each of the 4 tensors as a labeled rectangle annotated with shape.
    tensors = [
        {"y": 78, "name": "hex_x", "rows": 19, "color": "#ff9933", "edge": "#a86b00",
         "node_letter": "H", "n_nodes": 19, "type_label": "HEX"},
        {"y": 60, "name": "vert_x", "rows": 54, "color": "#3399cc", "edge": "#1a4f7a",
         "node_letter": "V", "n_nodes": 54, "type_label": "VERTEX"},
        {"y": 42, "name": "edge_x", "rows": 72, "color": "#aa66cc", "edge": "#5a1a8a",
         "node_letter": "E", "n_nodes": 72, "type_label": "EDGE"},
        {"y": 22, "name": "scalars", "rows": 1, "color": "#888888", "edge": "#444",
         "node_letter": "S", "n_nodes": 1, "type_label": "SCALARS"},
    ]

    for t in tensors:
        y = t["y"]

        # Tensor box (left side)
        tx = 4
        tw = 18
        # We can't draw 19/54/72 individual rows so render a stylized
        # "tensor with N rows" — title row + 3 representative rows + ellipsis + last row.
        ax.add_patch(mpatches.FancyBboxPatch((tx, y - 6.5), tw, 13,
                                              boxstyle="round,pad=0.3",
                                              edgecolor=t["edge"],
                                              facecolor="white", linewidth=2))
        ax.text(tx + tw/2, y + 5, f"{t['name']}", ha="center", fontsize=11,
                fontweight="bold", color=t["edge"], family="monospace")
        ax.text(tx + tw/2, y + 2.5, f"shape: [{t['rows']}, {H}]",
                ha="center", fontsize=10, family="monospace")
        ax.text(tx + tw/2, y - 1, f"({t['rows']} rows, each\n{H}-dim feature vector)",
                ha="center", fontsize=8, color="#555")

        # Show a stylized few rows — dots for the first 3 + last row
        rows_to_show = min(3, t["rows"])
        if t["rows"] > 1:
            for ri in range(rows_to_show):
                ry = y - 3.5 - ri * 0.6
                # row label
                ax.text(tx + 1, ry, f"row {ri}:", fontsize=6, color="#888",
                        family="monospace", va="center")
                # dots representing the row's features (just show 12 of 32)
                for di in range(12):
                    cx_dot = tx + 5 + di * 1
                    ax.scatter([cx_dot], [ry], s=4, color=t["color"], alpha=0.7,
                               edgecolors=t["edge"], linewidths=0.3)

        # ====== ARROW: tensor → graph node ======
        # The arrow says "row r → node X_r"
        arrow_y = y
        arrow_start_x = tx + tw + 0.5
        arrow_end_x = 50
        ax.annotate("",
                    xy=(arrow_end_x, arrow_y),
                    xytext=(arrow_start_x, arrow_y),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=2))
        # Label the arrow
        mid_x = (arrow_start_x + arrow_end_x) / 2
        ax.text(mid_x, arrow_y + 1.8, "row i → node " + t["node_letter"] + "_i",
                ha="center", fontsize=9, color="#444", family="monospace",
                fontweight="bold")
        ax.text(mid_x, arrow_y - 1.8, "(direct assignment — no extra weights)",
                ha="center", fontsize=8, color="#888", style="italic")

        # ====== RIGHT SIDE: stylized node group ======
        rx = 53
        rw = 18
        ax.add_patch(mpatches.FancyBboxPatch((rx, y - 7), rw, 14,
                                              boxstyle="round,pad=0.3",
                                              edgecolor=t["edge"],
                                              facecolor=t["color"],
                                              alpha=0.18, linewidth=2))
        ax.text(rx + rw/2, y + 5, f"{t['n_nodes']} {t['type_label']} nodes",
                ha="center", fontsize=11, fontweight="bold", color=t["edge"])

        # Draw a few representative nodes as colored dots
        n_to_draw = min(7, t["n_nodes"])
        for ni in range(n_to_draw):
            nx = rx + 2 + ni * (rw - 4) / max(n_to_draw - 1, 1)
            ax.scatter([nx], [y - 0.5], s=70, color=t["color"],
                       edgecolors=t["edge"], linewidths=1, zorder=5)
            ax.text(nx, y - 0.5, f"{t['node_letter']}{ni}", ha="center", va="center",
                    fontsize=6, fontweight="bold", color="white", zorder=6)
        if t["n_nodes"] > n_to_draw:
            ax.text(rx + rw - 2, y - 0.5, "...", ha="center", va="center",
                    fontsize=10, color=t["edge"])

        ax.text(rx + rw/2, y - 4.5,
                f"Each node starts with a {H}-dim feature vector",
                ha="center", fontsize=8, color="#444", style="italic")

        # ====== FAR RIGHT: shape note ======
        ex = 75
        ax.text(ex, y + 2,
                f"= {t['n_nodes']} × {H} tensor",
                fontsize=10, family="monospace", color=t["edge"],
                fontweight="bold")
        ax.text(ex, y - 1,
                "fed into SAGEConv\nas this node-type's\nstarting features",
                fontsize=9, color="#444")

    # ============== BOTTOM PANEL: the connectivity (edge_index) ==============
    bot_y = 8
    ax.add_patch(mpatches.FancyBboxPatch((4, bot_y - 5), 92, 8,
                                          boxstyle="round,pad=0.3",
                                          edgecolor="#444",
                                          facecolor="#fffae0", linewidth=1.5))
    ax.text(50, bot_y + 1.5, "PLUS: edge_index (which-node-is-connected-to-which)",
            ha="center", fontsize=11, fontweight="bold", color="#a86b00")
    ax.text(50, bot_y - 1, "FIXED by Catan board geometry (NOT learned). "
                              "Tells SAGEConv: hex H_i is connected to vertices V_a, V_b, V_c, ... etc.",
            ha="center", fontsize=9, color="#555")
    ax.text(50, bot_y - 3.5, "→ SAGEConv = (node features tensor) + (edge_index) → updated node features",
            ha="center", fontsize=9, color="#444", style="italic")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("runs/v3/dashboard"))
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Plotting architecture (h{}, l{})...".format(args.hidden_dim, args.num_layers))
    plot_architecture(args.out_dir / f"viz_arch_h{args.hidden_dim}_l{args.num_layers}.png",
                      args.hidden_dim, args.num_layers)

    # Also plot the largest cell for comparison.
    print("Plotting architecture (h128, l4)...")
    plot_architecture(args.out_dir / "viz_arch_h128_l4.png", 128, 4)

    print("Plotting message-passing graph...")
    plot_message_passing_graph(args.out_dir / "viz_graph.png")

    print("Plotting input routing (state → projections)...")
    plot_input_routing(args.out_dir / f"viz_input_h{args.hidden_dim}.png",
                        hidden_dim=args.hidden_dim)

    print("Plotting SAGEConv zoom-in/out diagram...")
    plot_sageconv_zoom(args.out_dir / f"viz_sageconv_h{args.hidden_dim}.png",
                        hidden_dim=args.hidden_dim)

    print("Plotting input -> sageconv handoff...")
    plot_input_to_sageconv(args.out_dir / f"viz_input_to_sageconv_h{args.hidden_dim}.png",
                            hidden_dim=args.hidden_dim)

    print()
    print("Done. Files in", args.out_dir.absolute())


if __name__ == "__main__":
    main()
