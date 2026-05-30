"""Starting-pip, ending-pip, and conversion-to-victory analysis on the
1200-game e10c head-to-head tournament.

Three deliverables per role:
  1. Mean & median STARTING pip (sum of pip on 2 setup settlements).
     Reuses opening_quality.opening_pip_per_seat.
  2. Mean & median ENDING pip (sum of pip on all settled+city vertices
     at game end; city = 2 x pip per Catan rules — cities double the
     resource yield). Replays full action_history through the engine.
  3. Conversion-to-victory: P(win | this role had strict highest starting
     pip) and P(win | this role had strict highest ending pip).

Cites:
  - hex_features[h, 5] = (dice - 7) / 5; hex_features[h, 7] = desert flag
    (observation.rs:75-86).
  - vertex_features[v, 0/1/2] = empty/settle/city; [v, 3..7] = owner one-hot
    perspective-rotated (observation.rs:89-99).
  - PIP_BY_DICE from opening_quality.py.
  - role rotation: SEATING[(seat + rot) % 4] matches e10_triple_gnn.py:83.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from catan_bot import _engine
from catan_gnn.analysis.opening_quality import (
    opening_pip_per_seat,
    PIP_BY_DICE,
    CHANCE_BIT,
)

ROOT = Path("/mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study")
TR = ROOT / "runs/v3/e10c_4way_1200_2026_05_26/2026-05-26T15-59-e10c_triple_gnn"


def role_for(seat: int, rot: int, seating: list[str]) -> str:
    return seating[(seat + rot) % 4]


def _build_vertex_to_hexes() -> dict[int, list[int]]:
    from catan_gnn.adjacency import HEX_TO_VERTICES
    v2h: dict[int, list[int]] = {v: [] for v in range(54)}
    for h, vs in enumerate(HEX_TO_VERTICES):
        for v in vs:
            v2h[int(v)].append(h)
    return v2h


def ending_pip_per_seat(seed: int, action_history) -> list[float]:
    """Replay the full game; return per-seat ending-pip (settlements 1×,
    cities 2× per Catan yield rules). Returns [-1]*4 on replay failure.

    Reads vertex_features from a viewer=0 observation: col 0/1/2 are
    empty/settle/city flags (board topology — viewer-independent), col
    3..7 are owner one-hot in viewer's perspective. Since viewer=0,
    perspective_idx(p, viewer=0) = p, so col 3+p is True iff player p
    owns the vertex.
    """
    v2h = _build_vertex_to_hexes()
    eng = _engine.Engine(int(seed))
    # Decode hex dice from the initial observation (board is static).
    obs0 = eng.observation_for(0)
    hex_feat = obs0["hex_features"]
    hex_dice = []
    for h in range(19):
        if float(hex_feat[h, 7]) >= 0.5:
            hex_dice.append(0)
        else:
            n = round(float(hex_feat[h, 5]) * 5.0 + 7.0)
            hex_dice.append(int(n))

    # Replay all actions (including chance points encoded with CHANCE_BIT).
    for a in action_history:
        a = int(a)
        if a & CHANCE_BIT:
            outcome = a & ~CHANCE_BIT
            try:
                eng.apply_chance_outcome(outcome)
            except Exception:
                return [-1, -1, -1, -1]
        else:
            try:
                eng.step(a)
            except Exception:
                return [-1, -1, -1, -1]

    # Read final vertex_features in viewer=0 perspective.
    obs_end = eng.observation_for(0)
    vf = obs_end["vertex_features"]
    pips = [0.0, 0.0, 0.0, 0.0]
    for v in range(54):
        is_settle = float(vf[v, 1]) >= 0.5
        is_city = float(vf[v, 2]) >= 0.5
        if not (is_settle or is_city):
            continue
        # Owner: cols 3..7 in viewer=0 perspective; col (3+p) for player p
        owner = -1
        for p in range(4):
            if float(vf[v, 3 + p]) >= 0.5:
                owner = p
                break
        if owner < 0:
            continue
        v_pip = sum(PIP_BY_DICE.get(hex_dice[h], 0) for h in v2h[v])
        weight = 2.0 if is_city else 1.0
        pips[owner] += weight * v_pip
    return pips


def load_all_with_labels():
    cfg = json.loads((TR / "worker0" / "config.json").read_text())
    seating = cfg["seating"]
    label_map = {
        "PureGnnA": cfg["label_a"],
        "PureGnnB": cfg["label_b"],
        "PureGnnC": cfg["label_c"],
        "LookaheadMctsV3": "LookaheadMctsV3",
    }
    rows = []
    for w in sorted(TR.glob("worker*")):
        for parq in sorted(w.glob("games.rot=*.parquet")):
            rot = int(parq.name.split(".")[1].split("=")[1])
            df = pq.read_table(str(parq)).to_pandas()
            df["rot"] = rot
            rows.append(df)
    return pd.concat(rows, ignore_index=True), seating, label_map


def summarize(role_values: dict[str, list[float]], labels: list[str], unit: str = "") -> None:
    print(f"  {'role':<28s} {'mean':>8s} {'median':>8s} {'min':>6s} {'max':>6s} {'n':>5s}")
    for role in role_values:
        xs = role_values[role]
        if not xs:
            continue
        xs_sorted = sorted(xs)
        mean = sum(xs) / len(xs)
        median = xs_sorted[len(xs) // 2]
        lbl = labels[role] if isinstance(labels, dict) else role
        print(f"  {lbl:<28s} {mean:>8.2f} {median:>8.2f} {xs_sorted[0]:>6.1f} {xs_sorted[-1]:>6.1f} {len(xs):>5d}")


def main():
    g, seating, label_map = load_all_with_labels()
    n_total = len(g)
    print(f"Loaded {n_total} games from {TR.name}")
    print(f"Seating: {seating}")
    print(f"Labels:")
    for k, v in label_map.items():
        print(f"  {k:<20s} = {v}")
    print()

    # ----------- 1. STARTING PIP -----------
    print("=== 1. Starting pip (sum of pips on 2 setup settlements) ===")
    print("Computing starting pip for all games...")
    starting_pips: list[list[int]] = []
    for _, row in g.iterrows():
        try:
            p = opening_pip_per_seat(int(row["seed"]), row["action_history"])
        except Exception:
            p = [-1, -1, -1, -1]
        starting_pips.append(p)
    g["start_pips"] = starting_pips
    valid_start = g[g["start_pips"].apply(lambda p: all(x >= 0 for x in p))].copy()
    print(f"  successful: {len(valid_start)}/{n_total}")

    role_start: dict[str, list[float]] = {r: [] for r in seating}
    for _, row in valid_start.iterrows():
        rot = int(row["rot"])
        for seat in range(4):
            role = role_for(seat, rot, seating)
            role_start[role].append(float(row["start_pips"][seat]))
    summarize(role_start, label_map)

    # ----------- 2. ENDING PIP -----------
    print()
    print("=== 2. Ending pip (settlements x1 + cities x2 of pips on owned vertices) ===")
    print("Computing ending pip for all games (replaying full action_history)...")
    ending_pips: list[list[float]] = []
    failures = 0
    for i, (_, row) in enumerate(g.iterrows()):
        try:
            p = ending_pip_per_seat(int(row["seed"]), row["action_history"])
        except Exception:
            p = [-1.0, -1.0, -1.0, -1.0]
        if p[0] < 0:
            failures += 1
        ending_pips.append(p)
        if (i + 1) % 200 == 0:
            print(f"  progress: {i+1}/{n_total} (failures so far: {failures})")
    g["end_pips"] = ending_pips
    valid_end = g[g["end_pips"].apply(lambda p: all(x >= 0 for x in p))].copy()
    print(f"  successful: {len(valid_end)}/{n_total}")

    role_end: dict[str, list[float]] = {r: [] for r in seating}
    for _, row in valid_end.iterrows():
        rot = int(row["rot"])
        for seat in range(4):
            role = role_for(seat, rot, seating)
            role_end[role].append(float(row["end_pips"][seat]))
    summarize(role_end, label_map)

    # ----------- 3. CONVERSION TO VICTORY -----------
    print()
    print("=== 3. Conversion to victory ===")

    # 3a. P(win | strict highest STARTING pip)
    print()
    print("  3a. When this role had STRICT highest starting pip:")
    print(f"  {'role':<28s} {'n_top':>6s} {'wins':>6s} {'conv%':>8s}")
    for role in seating:
        n_top = 0
        n_wins = 0
        for _, row in valid_start.iterrows():
            rot = int(row["rot"])
            pips = row["start_pips"]
            max_p = max(pips)
            top_seats = [s for s, p in enumerate(pips) if p == max_p]
            if len(top_seats) != 1:
                continue
            top_seat = top_seats[0]
            if role_for(top_seat, rot, seating) != role:
                continue
            n_top += 1
            if int(row["winner"]) == top_seat:
                n_wins += 1
        conv = (100 * n_wins / n_top) if n_top else 0
        lbl = label_map[role]
        print(f"  {lbl:<28s} {n_top:>6d} {n_wins:>6d} {conv:>7.1f}%")

    # 3b. P(win | strict highest ENDING pip)
    print()
    print("  3b. When this role had STRICT highest ending pip:")
    print(f"  {'role':<28s} {'n_top':>6s} {'wins':>6s} {'conv%':>8s}")
    for role in seating:
        n_top = 0
        n_wins = 0
        for _, row in valid_end.iterrows():
            rot = int(row["rot"])
            pips = row["end_pips"]
            max_p = max(pips)
            top_seats = [s for s, p in enumerate(pips) if p == max_p]
            if len(top_seats) != 1:
                continue
            top_seat = top_seats[0]
            if role_for(top_seat, rot, seating) != role:
                continue
            n_top += 1
            if int(row["winner"]) == top_seat:
                n_wins += 1
        conv = (100 * n_wins / n_top) if n_top else 0
        lbl = label_map[role]
        print(f"  {lbl:<28s} {n_top:>6d} {n_wins:>6d} {conv:>7.1f}%")

    # 3c. Delta-pip: ending - starting (resource economy growth)
    print()
    print("=== 4. Pip growth (ending - starting) — economy expansion proxy ===")
    role_delta: dict[str, list[float]] = {r: [] for r in seating}
    common = g.index.intersection(valid_start.index).intersection(valid_end.index)
    valid_both = g.loc[common]
    for _, row in valid_both.iterrows():
        rot = int(row["rot"])
        for seat in range(4):
            role = role_for(seat, rot, seating)
            start = float(row["start_pips"][seat])
            end = float(row["end_pips"][seat])
            role_delta[role].append(end - start)
    summarize(role_delta, label_map, unit="pip")

    # ----------- 5. Pip vs winrate context -----------
    print()
    print("=== 5. Role winrates (from this 1200-game run, for context) ===")
    role_wins = {r: 0 for r in seating}
    role_games = {r: 0 for r in seating}
    for _, row in g.iterrows():
        rot = int(row["rot"])
        winner = int(row["winner"])
        for seat in range(4):
            role = role_for(seat, rot, seating)
            role_games[role] += 1
            if winner == seat:
                role_wins[role] += 1
    print(f"  {'role':<28s} {'wins':>6s} {'games':>7s} {'%':>8s}")
    for role in seating:
        pct = (100 * role_wins[role] / role_games[role]) if role_games[role] else 0
        print(f"  {label_map[role]:<28s} {role_wins[role]:>6d} {role_games[role]:>7d} {pct:>7.2f}%")


if __name__ == "__main__":
    sys.exit(main() or 0)
