"""Plot the 4-quadrant rules x opponents winrate matrix.

Visualizes how each cell's winrate changes across:
  - Rule set: v3 (5 VP, no bonuses) vs Full Catan (10 VP, bonuses)
  - Opponents: with LookV3 vs without LookV3

Numbers from the four 1200-game tournaments documented in companion journals.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = Path("/mnt/c/dojo/catan_bot/.claude/worktrees/v3/docs/superpowers/journals/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

CELL_COLORS = {
    "Cell0-vanilla":   "#888888",
    "Cell1-cand8+10":  "#1f77b4",
    "Cell5v2-cand11":  "#2ca02c",
    "Cell6-stack":     "#d62728",
    "LookaheadV3":     "#000000",
}

# Four contexts (n=1200 each, cited journals):
# 1. v3 rules + LookV3 (2026-05-26 head-to-head)
# 2. v3 rules + no LookV3 (2026-05-27 4-PureGnn)
# 3. Full Catan + no LookV3 (2026-05-27 full-Catan 4-PureGnn)
# 4. Full Catan + LookV3 (2026-05-28 — this run)

# (cell -> winrate %) per context
contexts = {
    "v3 + LookV3":         {"Cell0-vanilla": 7.92, "Cell1-cand8+10": 8.83, "Cell5v2-cand11": 14.67, "Cell6-stack": 8.92, "LookaheadV3": 67.58},
    "v3 + no-LookV3":      {"Cell0-vanilla": 17.33, "Cell1-cand8+10": 24.83, "Cell5v2-cand11": 30.92, "Cell6-stack": 26.92, "LookaheadV3": None},
    "Full Catan + no-LookV3": {"Cell0-vanilla": 1.08, "Cell1-cand8+10": 42.92, "Cell5v2-cand11": 0.75, "Cell6-stack": 54.33, "LookaheadV3": None},
    "Full Catan + LookV3": {"Cell0-vanilla": None, "Cell1-cand8+10": 10.83, "Cell5v2-cand11": 0.25, "Cell6-stack": 19.00, "LookaheadV3": 69.92},
}

cells = ["Cell0-vanilla", "Cell1-cand8+10", "Cell5v2-cand11", "Cell6-stack", "LookaheadV3"]
ctx_names = list(contexts.keys())

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, ctx in enumerate(ctx_names):
    ax = axes[i]
    cell_winrates = contexts[ctx]
    cells_present = [c for c in cells if cell_winrates.get(c) is not None]
    vals = [cell_winrates[c] for c in cells_present]
    colors = [CELL_COLORS[c] for c in cells_present]
    bars = ax.bar(cells_present, vals, color=colors, edgecolor="white")
    ax.set_title(f"{ctx} (n=1200)", fontsize=13, fontweight="bold")
    ax.set_ylabel("winrate %")
    ax.set_ylim(0, 80)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticklabels(cells_present, rotation=15, ha="right", fontsize=9)
    # Annotate values
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}",
                ha="center", fontsize=10, fontweight="bold")
    # 25% reference line for the no-LookV3 contexts (where 25% is uniform)
    if cell_winrates.get("LookaheadV3") is None:
        ax.axhline(25, color="black", linestyle=":", alpha=0.4, label="uniform-random (25%)")
        ax.legend(loc="upper left", fontsize=9)

fig.suptitle("Cumulative-best is RULE-CONDITIONAL\nSame 4 GNN cells, 4 tournament contexts, n=1200 each",
             fontsize=14, fontweight="bold")
fig.tight_layout()
p = FIG_DIR / "rules_opponents_matrix.png"
fig.savefig(p, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"saved {p}")

# Also produce a "cell rank by context" summary
fig, ax = plt.subplots(figsize=(12, 6))
ctx_short = ["v3+LookV3", "v3+noLook", "fullCatan+noLook", "fullCatan+LookV3"]
x = np.arange(len(ctx_short))
w = 0.18
gnn_cells = ["Cell0-vanilla", "Cell1-cand8+10", "Cell5v2-cand11", "Cell6-stack"]

for j, cell in enumerate(gnn_cells):
    vals = [contexts[ctx].get(cell, 0) or 0 for ctx in ctx_names]
    ax.bar(x + (j - 1.5) * w, vals, w, label=cell, color=CELL_COLORS[cell], edgecolor="white")
    for xi, v in enumerate(vals):
        if v > 0:
            ax.text(xi + (j - 1.5) * w, v + 0.5, f"{v:.0f}", ha="center", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(ctx_short)
ax.set_ylabel("winrate %")
ax.set_title("GNN winrate by context — same 4 cells, 4 tournament setups")
ax.legend(loc="upper left", ncol=4, fontsize=9)
ax.grid(axis="y", alpha=0.3)
p2 = FIG_DIR / "cell_rank_by_context.png"
fig.savefig(p2, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"saved {p2}")
