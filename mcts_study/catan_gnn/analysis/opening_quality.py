"""Opening-placement quality analysis (mean pip + win conversion).

Re-usable module variant of scratch_opening_analysis.py. For each
tournament game:
  - Replay setup (action_history[0..15] = 4 settlements interleaved with
    4 roads × 2 rounds in snake-draft).
  - Per-seat pip count = sum over the seat's 2 settlements of the pip
    values of adjacent hex tiles.
  - Higher pips = more expected resource production (cited Catan rules).

Then aggregate by role and ask:
  - What's each role's mean opening pip count?
  - What fraction of games did the role with strict highest opening pips
    actually win that game?

Usage as module:
    from catan_gnn.analysis.opening_quality import analyze
    result = analyze(tournament_dir, seating)

Usage as CLI:
    python -m catan_gnn.analysis.opening_quality \\
        --run-dir runs/v3/loss_aug/.../<tournament_dir> \\
        --seating "PureGnnA,PureGnnB,LookaheadMctsV3,Random"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from catan_bot import _engine


CHANCE_BIT = 0x80000000

# Standard Catan pip values (dots on the hex tile per dice number).
PIP_BY_DICE = {
    0: 0,   # desert / no number
    2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
    8: 5, 9: 4, 10: 3, 11: 2, 12: 1,
}


def _role_for(seat: int, rot: int, seating: list[str]) -> str:
    return seating[(seat + rot) % 4]


def opening_pip_per_seat(seed: int, action_history) -> list[int]:
    """Replay 16 setup steps; return list of 4 per-seat pip totals from
    each seat's 2 settlements. Returns [-1]*4 on replay failure.

    Cited: hex_features[h, 5] = (dice_num - 7) / 5 per observation.rs:83.
    BuildSettlement action_id = vertex_id (catan_gnn.adjacency.HEX_TO_VERTICES).
    """
    from catan_gnn.adjacency import HEX_TO_VERTICES
    eng = _engine.Engine(int(seed))
    obs = eng.observation_for(0)
    hex_feat = obs["hex_features"]
    hex_dice = []
    for h in range(19):
        if float(hex_feat[h, 7]) >= 0.5:
            hex_dice.append(0)  # desert
        else:
            n = round(float(hex_feat[h, 5]) * 5.0 + 7.0)
            hex_dice.append(int(n))
    v2h: dict[int, list[int]] = {v: [] for v in range(54)}
    for h, vs in enumerate(HEX_TO_VERTICES):
        for v in vs:
            v2h[int(v)].append(h)

    settle_pips = [0, 0, 0, 0]
    for i in range(16):
        a = int(action_history[i])
        if a & CHANCE_BIT:
            return [-1, -1, -1, -1]
        cp = int(eng.current_player())
        if i % 2 == 0:
            # settlement: vertex id = a
            for h in v2h[a]:
                settle_pips[cp] += PIP_BY_DICE.get(hex_dice[h], 0)
        eng.step(a)
    return settle_pips


def load_games(tournament_dir: Path) -> pd.DataFrame:
    rows = []
    for parq in sorted(tournament_dir.rglob("games.rot=*.parquet")):
        rot = int(parq.name.split(".")[1].split("=")[1])
        df = pq.read_table(str(parq)).to_pandas()
        df["rot"] = rot
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No games.rot=*.parquet under {tournament_dir}")
    return pd.concat(rows, ignore_index=True)


def analyze(tournament_dir: Path, seating: list[str], verbose: bool = False) -> dict:
    """Compute mean opening pips per role and win-conversion when a role
    had strictly highest opening pips."""
    g = load_games(tournament_dir)
    if verbose:
        print(f"loaded {len(g)} games from {tournament_dir.name}")

    # Per-game opening pips
    pips_per_game: list[list[int]] = []
    for _, row in g.iterrows():
        try:
            p = opening_pip_per_seat(int(row["seed"]), row["action_history"])
        except Exception:
            p = [-1, -1, -1, -1]
        pips_per_game.append(p)
    g["seat_pips"] = pips_per_game

    valid = g[g["seat_pips"].apply(lambda p: all(x >= 0 for x in p))].copy()

    # Per-role mean opening pips
    role_pips: dict[str, list[int]] = {r: [] for r in seating}
    for _, row in valid.iterrows():
        rot = int(row["rot"])
        for seat in range(4):
            role = _role_for(seat, rot, seating)
            role_pips[role].append(row["seat_pips"][seat])
    mean_pips = {r: (sum(role_pips[r]) / len(role_pips[r]) if role_pips[r] else 0.0)
                 for r in seating}
    median_pips = {r: (sorted(role_pips[r])[len(role_pips[r]) // 2]
                       if role_pips[r] else 0) for r in seating}

    # Win-conversion when role had strict highest opening
    role_win_when_top: dict[str, tuple[int, int]] = {}  # (n_games_with_top, wins)
    for role in seating:
        n_top = 0
        n_wins = 0
        for _, row in valid.iterrows():
            rot = int(row["rot"])
            pips = row["seat_pips"]
            max_p = max(pips)
            top_seats = [s for s, p in enumerate(pips) if p == max_p]
            if len(top_seats) != 1:
                continue
            top_seat = top_seats[0]
            if _role_for(top_seat, rot, seating) != role:
                continue
            n_top += 1
            if int(row["winner"]) == top_seat:
                n_wins += 1
        role_win_when_top[role] = (n_top, n_wins)

    return {
        "n_games_total": len(g),
        "n_games_valid": len(valid),
        "seating": seating,
        "mean_pips": mean_pips,
        "median_pips": median_pips,
        "role_pips_n": {r: len(role_pips[r]) for r in seating},
        "role_win_when_top": role_win_when_top,
    }


def print_report(result: dict, label: str = "") -> None:
    print(f"\n=== Opening-placement quality: {label} ===")
    print(f"games scanned: {result['n_games_total']} ({result['n_games_valid']} valid)\n")
    seating = result["seating"]

    print(f"  Mean opening pip count per role:")
    print(f"    {'role':<22s} {'mean':>8s} {'median':>8s} {'n':>6s}")
    for r in seating:
        m = result["mean_pips"][r]
        md = result["median_pips"][r]
        n = result["role_pips_n"][r]
        print(f"    {r:<22s} {m:>8.2f} {md:>8d} {n:>6d}")

    print(f"\n  Win conversion when role had STRICT highest opening pips:")
    print(f"    {'role':<22s} {'n_top':>6s} {'wins':>6s} {'conv%':>8s}")
    for r in seating:
        n_top, n_wins = result["role_win_when_top"][r]
        conv = (100 * n_wins / n_top) if n_top else 0.0
        print(f"    {r:<22s} {n_top:>6d} {n_wins:>6d} {conv:>7.1f}%")


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--seating", type=str, required=True,
                   help="Comma-separated role names (e.g. PureGnnA,PureGnnB,LookaheadMctsV3,Random)")
    p.add_argument("--label", type=str, default="")
    args = p.parse_args()
    seating = [s.strip() for s in args.seating.split(",")]
    result = analyze(args.run_dir, seating, verbose=True)
    print_report(result, label=args.label)


if __name__ == "__main__":
    cli_main()
