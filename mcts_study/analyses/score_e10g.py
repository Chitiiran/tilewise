"""Score e10g_cheapsearch_async runs: win% per role (GnnMcts / RawPureGnn / LookV3).

Same rotation-aware seat->role mapping as score_e10f. Pass one or more run dirs;
each is scored independently (so you can compare sims=8/16/32 side by side).

Usage: python -m analyses.score_e10g <run_dir> [<run_dir> ...]
"""
import sys, glob, json
import pandas as pd

BASE = ["GnnMctsA", "RawPureGnnB", "RawPureGnnC", "LookaheadMctsV3"]
DISPLAY = {"GnnMctsA": "GnnMcts", "RawPureGnnB": "RawPureGnn",
           "RawPureGnnC": "RawPureGnn", "LookaheadMctsV3": "LookV3"}


def score_one(run_dir):
    gf = glob.glob(run_dir + "/*/games*.parquet") or glob.glob(run_dir + "/games*.parquet")
    g = pd.concat([pd.read_parquet(f) for f in gf], ignore_index=True)
    cfgs = glob.glob(run_dir + "/*/config*.json") or glob.glob(run_dir + "/config*.json")
    sims = "?"
    if cfgs:
        try:
            sims = json.load(open(cfgs[0])).get("sims", "?")
        except Exception:
            pass
    total = len(g)
    skipped = int((g["winner"] < 0).sum())
    appear = {"GnnMcts": 0, "RawPureGnn": 0, "LookV3": 0}
    wins = {"GnnMcts": 0, "RawPureGnn": 0, "LookV3": 0}
    for _, row in g.iterrows():
        rot = (int(row["seed"]) // 10_000) % 4
        seating = BASE[rot:] + BASE[:rot]
        for role in seating:
            appear[DISPLAY[role]] += 1
        w = int(row["winner"])
        if w >= 0:
            wins[DISPLAY[seating[w]]] += 1
    print(f"\n=== e10g sims={sims} — {total} games ({skipped} skipped) ===")
    print(f"{'player':<12}{'wins':>6}{'appear':>8}{'win%/appearance':>18}")
    for p in ("GnnMcts", "RawPureGnn", "LookV3"):
        wr = 100.0 * wins[p] / appear[p] if appear[p] else 0.0
        print(f"{p:<12}{wins[p]:>6}{appear[p]:>8}{wr:>17.1f}%")
    return sims, wins, appear


def main():
    for rd in sys.argv[1:]:
        score_one(rd)


if __name__ == "__main__":
    main()
