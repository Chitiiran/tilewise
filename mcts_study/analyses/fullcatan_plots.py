"""Generate plots for the full-Catan deep behavioral analysis.

Re-runs the analysis from scratch_fullcatan_deep_analysis (importing
its analyze_game function) and produces matplotlib figures saved to
docs/superpowers/journals/figures/.

Plots:
  1. bonus_holding.png — bar chart of %LR_held and %LA_held per cell
     (overall + wins side-by-side).
  2. knights_vs_roads.png — scatter of mean roads vs mean knights per
     game, color-coded by cell.
  3. game_length_dist.png — boxplot of game-length distribution per
     winning cell.
  4. winrate_by_rules.png — bar chart of v3-rules vs full-Catan winrate
     per cell (the inversion plot).
  5. port_resource_specialization.png — stacked bar of settlement
     hex-resource distribution per cell.
  6. bonus_contribution_to_wins.png — for winners only, stacked bar:
     % wins with LR / LA / both / neither.

Total runtime: ~25 min (replay 1200 games) + ~30s plotting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import json

# Import the analyze_game function from the deep-analysis script
sys.path.insert(0, str(Path(__file__).parent))
from scratch_fullcatan_deep_analysis import analyze_game, TR, V2H

FIG_DIR = Path("/mnt/c/dojo/catan_bot/.claude/worktrees/v3/docs/superpowers/journals/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Cell colors (consistent across plots)
CELL_COLORS = {
    "Cell0-vanilla":   "#888888",
    "Cell1-cand8+10":  "#1f77b4",
    "Cell5v2-cand11":  "#2ca02c",
    "Cell6-stack":     "#d62728",
}


def replay_all():
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
    role_to_short = {role: short[labels[role]] for role in seating}

    rows = []
    for w in sorted(TR.glob("worker*")):
        for parq in sorted(w.glob("games.rot=*.parquet")):
            rot = int(parq.name.split(".")[1].split("=")[1])
            df = pq.read_table(str(parq)).to_pandas()
            df["rot"] = rot
            rows.append(df)
    g = pd.concat(rows, ignore_index=True)
    print(f"Replaying {len(g)} games for plot data...")

    games = []
    for i, (_, row) in enumerate(g.iterrows()):
        r = analyze_game(int(row["seed"]), row["action_history"], seating,
                         int(row["rot"]), int(row["winner"]), row["final_vp"])
        if r is not None:
            games.append(r)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(g)}")
    print(f"  done. {len(games)} games replayed.")
    return games, seating, role_to_short


def plot_bonus_holding(games, seating, role_to_short):
    """Bar chart: %LR_held and %LA_held per cell, overall + in_wins."""
    cells = list(role_to_short.values())
    role_by_short = {v: k for k, v in role_to_short.items()}

    pct_lr_overall = []
    pct_la_overall = []
    pct_lr_wins = []
    pct_la_wins = []
    for sname in cells:
        role = role_by_short[sname]
        n_all = len(games)
        n_lr = sum(1 for g in games if g[role].get("lr_holder", False))
        n_la = sum(1 for g in games if g[role].get("la_holder", False))
        pct_lr_overall.append(100 * n_lr / n_all)
        pct_la_overall.append(100 * n_la / n_all)
        wins = [g for g in games if g.get("_winner_role") == role]
        if wins:
            n_lr_w = sum(1 for g in wins if g[role].get("lr_holder", False))
            n_la_w = sum(1 for g in wins if g[role].get("la_holder", False))
            pct_lr_wins.append(100 * n_lr_w / len(wins))
            pct_la_wins.append(100 * n_la_w / len(wins))
        else:
            pct_lr_wins.append(0); pct_la_wins.append(0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(cells))
    w = 0.35

    axes[0].bar(x - w/2, pct_lr_overall, w, label="LR held", color="#1f77b4")
    axes[0].bar(x + w/2, pct_la_overall, w, label="LA held", color="#ff7f0e")
    axes[0].set_xticks(x); axes[0].set_xticklabels(cells, rotation=15, ha="right")
    axes[0].set_ylabel("% games")
    axes[0].set_title("Bonus holding — ALL 1200 games")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    for i, (lr, la) in enumerate(zip(pct_lr_overall, pct_la_overall)):
        axes[0].text(i - w/2, lr + 1, f"{lr:.0f}", ha="center", fontsize=9)
        axes[0].text(i + w/2, la + 1, f"{la:.0f}", ha="center", fontsize=9)

    axes[1].bar(x - w/2, pct_lr_wins, w, label="LR held", color="#1f77b4")
    axes[1].bar(x + w/2, pct_la_wins, w, label="LA held", color="#ff7f0e")
    axes[1].set_xticks(x); axes[1].set_xticklabels(cells, rotation=15, ha="right")
    axes[1].set_ylabel("% of winning games")
    axes[1].set_title("Bonus holding in WINS only")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    for i, (lr, la) in enumerate(zip(pct_lr_wins, pct_la_wins)):
        axes[1].text(i - w/2, lr + 1, f"{lr:.0f}", ha="center", fontsize=9)
        axes[1].text(i + w/2, la + 1, f"{la:.0f}", ha="center", fontsize=9)

    fig.suptitle("Longest Road / Largest Army holding — Full Catan tournament (n=1200)")
    fig.tight_layout()
    p = FIG_DIR / "bonus_holding.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def plot_knights_vs_roads(games, seating, role_to_short):
    """Scatter: per-game knights vs roads, colored by cell.
    Marker size = final VP (winners big, losers small)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    role_by_short = {v: k for k, v in role_to_short.items()}
    for sname, color in CELL_COLORS.items():
        role = role_by_short[sname]
        roads = [g[role].get("n_road_built", 0) for g in games]
        knights = [g[role].get("knights_played", 0) for g in games]
        is_winner = [g.get("_winner_role") == role for g in games]
        # Plot losers small/transparent, winners full
        roads_l = [r for r, w in zip(roads, is_winner) if not w]
        knights_l = [k for k, w in zip(knights, is_winner) if not w]
        roads_w = [r for r, w in zip(roads, is_winner) if w]
        knights_w = [k for k, w in zip(knights, is_winner) if w]
        ax.scatter(roads_l, knights_l, s=8, color=color, alpha=0.1)
        ax.scatter(roads_w, knights_w, s=30, color=color, alpha=0.6,
                   label=f"{sname} ({sum(is_winner)} wins)", edgecolor="white", linewidth=0.5)
    ax.axhline(3, color="red", linestyle="--", alpha=0.5, label="LA threshold (3 knights)")
    ax.axvline(5, color="blue", linestyle="--", alpha=0.5, label="LR threshold (5 roads)")
    ax.set_xlabel("Roads built per game")
    ax.set_ylabel("Knights played per game")
    ax.set_title("Per-game knights vs roads (n=1200, full Catan)\nSmall faded dots = losses; large dots = wins")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    p = FIG_DIR / "knights_vs_roads.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def plot_game_length_dist(games, seating, role_to_short):
    """Boxplot of game lengths for each WINNING cell."""
    fig, ax = plt.subplots(figsize=(10, 6))
    role_by_short = {v: k for k, v in role_to_short.items()}
    data = []
    labels = []
    for sname in role_to_short.values():
        role = role_by_short[sname]
        lens = [g["_game_length"] for g in games if g.get("_winner_role") == role]
        if not lens:
            continue
        data.append(lens)
        labels.append(f"{sname}\n({len(lens)} wins)")
    bp = ax.boxplot(data, labels=labels, showfliers=True, patch_artist=True,
                     medianprops=dict(color="black", linewidth=2),
                     flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch, sname in zip(bp["boxes"], role_to_short.values()):
        patch.set_facecolor(CELL_COLORS[sname])
        patch.set_alpha(0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Game length (moves, log scale)")
    ax.set_title("Game-length distribution per winning cell (n=1189 wins, full Catan)")
    ax.grid(axis="y", alpha=0.3)
    p = FIG_DIR / "game_length_dist.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def plot_winrate_by_rules(games, seating, role_to_short):
    """Bar chart: v3 vs full-Catan winrate per cell. Hardcoded v3 numbers
    from prior tournament (`2026-05-27-4puregnn-no-lookahead-tournament.md`)."""
    v3_winrates = {
        "Cell0-vanilla":  17.33,
        "Cell1-cand8+10": 24.83,
        "Cell5v2-cand11": 30.92,
        "Cell6-stack":    26.92,
    }
    role_by_short = {v: k for k, v in role_to_short.items()}
    full_winrates = {}
    n_total = len(games)
    for sname in role_to_short.values():
        role = role_by_short[sname]
        wins = sum(1 for g in games if g.get("_winner_role") == role)
        full_winrates[sname] = 100 * wins / n_total

    cells = list(role_to_short.values())
    v3 = [v3_winrates[c] for c in cells]
    fc = [full_winrates[c] for c in cells]
    x = np.arange(len(cells))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w/2, v3, w, label="v3 rules (vp=5, no bonuses)", color="#888888")
    bars2 = ax.bar(x + w/2, fc, w, label="Full Catan (vp=10, bonuses)",
                    color=[CELL_COLORS[c] for c in cells])
    ax.axhline(25, color="black", linestyle=":", alpha=0.5, label="uniform-random (25%)")
    ax.set_xticks(x); ax.set_xticklabels(cells, rotation=15, ha="right")
    ax.set_ylabel("winrate %")
    ax.set_title("Same models, different rules: complete ranking inversion\n4-PureGnn tournaments, n=1200 each")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for i, (a, b) in enumerate(zip(v3, fc)):
        ax.text(i - w/2, a + 0.5, f"{a:.1f}", ha="center", fontsize=9)
        ax.text(i + w/2, b + 0.5, f"{b:.1f}", ha="center", fontsize=9, fontweight="bold")
    p = FIG_DIR / "winrate_by_rules.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def plot_resource_specialization(games, seating, role_to_short):
    """Stacked bar of settlement-vertex hex resource distribution per cell."""
    from catan_bot import _engine
    eng_static = _engine.Engine.with_rules(0, 10, True)
    obs_static = eng_static.observation_for(0)
    hex_feat_s = obs_static["hex_features"]
    hex_resource_static = []
    for h in range(19):
        if float(hex_feat_s[h, 7]) >= 0.5:
            hex_resource_static.append(-1)
        else:
            for r in range(5):
                if float(hex_feat_s[h, r]) >= 0.5:
                    hex_resource_static.append(r); break
            else:
                hex_resource_static.append(-1)

    role_by_short = {v: k for k, v in role_to_short.items()}
    RES_LABELS = ["wood", "brick", "sheep", "wheat", "ore", "desert"]
    RES_COLORS = ["#2d5016", "#8b3a3a", "#7fb069", "#dab94a", "#5a4a3a", "#bdb5a0"]

    cells = list(role_to_short.values())
    pcts = []
    for sname in cells:
        role = role_by_short[sname]
        counts = [0]*6  # 5 resources + desert
        for g in games:
            for v in g[role].get("settle_vertices", []):
                for h in V2H[v]:
                    r = hex_resource_static[h]
                    if r < 0:
                        counts[5] += 1
                    else:
                        counts[r] += 1
        total = sum(counts)
        pcts.append([100*c/total for c in counts] if total else [0]*6)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(cells))
    for ri, (label, color) in enumerate(zip(RES_LABELS, RES_COLORS)):
        vals = [p[ri] for p in pcts]
        ax.bar(cells, vals, label=label, color=color, bottom=bottom, edgecolor="white")
        # Annotate
        for ci, v in enumerate(vals):
            if v > 4:
                ax.text(ci, bottom[ci] + v/2, f"{v:.0f}", ha="center", va="center",
                        color="white" if ri in (0, 1, 4) else "black", fontsize=9)
        bottom += np.array(vals)
    ax.set_ylabel("% of settlement-adjacent hexes")
    ax.set_title("Resource specialization — full Catan tournament settlements (n=1200)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    p = FIG_DIR / "resource_specialization.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def plot_bonus_contribution_to_wins(games, seating, role_to_short):
    """For winners: stacked bar showing which bonus(es) accompanied each win."""
    role_by_short = {v: k for k, v in role_to_short.items()}
    cells = list(role_to_short.values())
    # 4 categories: both, LR-only, LA-only, neither
    breakdowns = {sname: [0, 0, 0, 0] for sname in cells}
    for sname in cells:
        role = role_by_short[sname]
        for g in games:
            if g.get("_winner_role") != role:
                continue
            lr = g[role].get("lr_holder", False)
            la = g[role].get("la_holder", False)
            if lr and la: breakdowns[sname][0] += 1
            elif lr:      breakdowns[sname][1] += 1
            elif la:      breakdowns[sname][2] += 1
            else:         breakdowns[sname][3] += 1

    fig, ax = plt.subplots(figsize=(10, 6))
    cats = ["both LR + LA", "LR only", "LA only", "neither"]
    cat_colors = ["#9b59b6", "#3498db", "#e67e22", "#bdc3c7"]
    bottom = np.zeros(len(cells))
    for ci, (cat, color) in enumerate(zip(cats, cat_colors)):
        vals = [breakdowns[c][ci] for c in cells]
        ax.bar(cells, vals, label=cat, color=color, bottom=bottom, edgecolor="white")
        for xi, v in enumerate(vals):
            if v > 5:
                ax.text(xi, bottom[xi] + v/2, str(v), ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")
        bottom += np.array(vals)
    # Annotate total wins per cell
    for xi, c in enumerate(cells):
        total = sum(breakdowns[c])
        ax.text(xi, bottom[xi] + 8, f"n={total}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("number of wins")
    ax.set_title("Bonus composition of wins — full Catan tournament\nWhich bonus(es) did the winning cell hold?")
    ax.legend(loc="upper left")
    p = FIG_DIR / "bonus_contribution_to_wins.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p}")


def main():
    games, seating, role_to_short = replay_all()
    print(f"\nGenerating plots into {FIG_DIR}...")
    plot_bonus_holding(games, seating, role_to_short)
    plot_knights_vs_roads(games, seating, role_to_short)
    plot_game_length_dist(games, seating, role_to_short)
    plot_winrate_by_rules(games, seating, role_to_short)
    plot_resource_specialization(games, seating, role_to_short)
    plot_bonus_contribution_to_wins(games, seating, role_to_short)
    print("Done.")


if __name__ == "__main__":
    main()
