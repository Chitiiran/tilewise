"""Score an e10f_valueq_async run: win% per ROLE (ValueQ / RawPureGnn / LookV3).

Each game records the winner as an absolute SEAT (0-3). Seating rotates per
game by rot_idx = (seed // 10_000) % 4 over the base order
["ValueQA","RawPureGnnB","RawPureGnnC","LookaheadMctsV3"], i.e.
seating = base[rot:] + base[:rot]. Map winner-seat -> role and tally.

Usage: python -m analyses.score_e10f <run_dir>
"""
import sys, glob
import pandas as pd

BASE = ["ValueQA", "RawPureGnnB", "RawPureGnnC", "LookaheadMctsV3"]
# collapse the two raw-PureGnn filler seats into one reported player
DISPLAY = {"ValueQA": "ValueQ", "RawPureGnnB": "RawPureGnn",
           "RawPureGnnC": "RawPureGnn", "LookaheadMctsV3": "LookV3"}


def main():
    run_dir = sys.argv[1]
    gf = glob.glob(run_dir + "/*/games*.parquet") or glob.glob(run_dir + "/games*.parquet")
    g = pd.concat([pd.read_parquet(f) for f in gf], ignore_index=True)
    total = len(g)
    # skipped/timeout games have winner == -1; count separately
    skipped = int((g["winner"] < 0).sum())

    # appearances per display-player (every game seats all 4 base roles once,
    # so ValueQ appears in every game; RawPureGnn appears twice per game)
    appear = {"ValueQ": 0, "RawPureGnn": 0, "LookV3": 0}
    wins = {"ValueQ": 0, "RawPureGnn": 0, "LookV3": 0}

    for _, row in g.iterrows():
        rot = (int(row["seed"]) // 10_000) % 4
        seating = BASE[rot:] + BASE[:rot]
        for seat, role in enumerate(seating):
            appear[DISPLAY[role]] += 1
        w = int(row["winner"])
        if w >= 0:
            wins[DISPLAY[seating[w]]] += 1

    print(f"e10f gate — {total} games ({skipped} skipped/timeout)\n")
    print(f"{'player':<12}{'wins':>6}{'appearances':>13}{'win%/appearance':>18}")
    for p in ("ValueQ", "RawPureGnn", "LookV3"):
        wr = 100.0 * wins[p] / appear[p] if appear[p] else 0.0
        print(f"{p:<12}{wins[p]:>6}{appear[p]:>13}{wr:>17.1f}%")
    # also report win% over GAMES (each finished game has one winner)
    finished = total - skipped
    print(f"\nover {finished} finished games (share of all wins):")
    for p in ("ValueQ", "RawPureGnn", "LookV3"):
        share = 100.0 * wins[p] / finished if finished else 0.0
        print(f"  {p:<12}{wins[p]:>4} wins  ({share:.1f}% of decided games)")
    print("\nGATE: ValueQ must beat RawPureGnn (same net, only deployment differs).")


if __name__ == "__main__":
    main()
