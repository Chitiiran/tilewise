"""Aggregate e5 results across worker shards: winrate per (depth, sims) cell."""
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

RUNDIR = Path("/mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/runs/2026-04-29T10-32-e5_lookahead_depth")

shards_by_cell = defaultdict(list)
shard_re = re.compile(r"games\.depth=(\d+)-sims=(\d+)\.parquet")
for p in RUNDIR.glob("worker*/games.*.parquet"):
    m = shard_re.match(p.name)
    if not m:
        continue
    depth, sims = int(m.group(1)), int(m.group(2))
    shards_by_cell[(depth, sims)].append(p)

print(f"{'depth':>5} {'sims':>5}  {'n':>4}  {'wins':>5}  {'winrate':>8}  {'mean_vp_p0':>11}  {'avg_len':>8}")
print("-" * 64)

skip_total = 0
for (depth, sims) in sorted(shards_by_cell):
    games = []
    for shard in shards_by_cell[(depth, sims)]:
        tbl = pq.read_table(shard).to_pandas()
        games.append(tbl)
    import pandas as pd
    df = pd.concat(games, ignore_index=True)
    n = len(df)
    # Player 0 is the MCTS bot in e5.
    wins_p0 = int((df["winner"] == 0).sum())
    winrate = wins_p0 / n if n else 0.0
    # final_vp is a list per row -> p0 VP
    mean_vp = df["final_vp"].apply(lambda v: v[0]).mean()
    avg_len = df["length_in_moves"].mean()
    print(f"{depth:>5} {sims:>5}  {n:>4}  {wins_p0:>5}  {winrate*100:>7.1f}%  {mean_vp:>11.2f}  {avg_len:>8.0f}")

print()
# Skipped seeds total
for w in range(4):
    skip = RUNDIR / f"worker{w}" / "skipped.csv"
    if skip.exists():
        with skip.open() as f:
            lines = f.readlines()
        skip_total += max(0, len(lines) - 1)
print(f"Total timeouts (skipped.csv across workers): {skip_total}")
