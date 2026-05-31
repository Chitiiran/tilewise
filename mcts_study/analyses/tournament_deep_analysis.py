"""Generalized deep behavioral analysis over a Catan tournament.

Adapted from fullcatan_deep_analysis.py to run on ANY tournament dir, handling
both parquet layouts:
  - flat:        <run>/games.<hash>.parquet           (e10e_async)
  - worker-subdir: <run>/worker*/games*.parquet        (e10d_quad_gnn)

Replays each game's action_history through the engine and extracts a per-role
behavioral profile: build dynamics (roads/settlements/cities), dev cards,
knights (Largest Army), Longest Road length, LR/LA bonus holding, VP, trades,
ports, resource specialization, robber targeting. Stratified by win/loss.

Usage:
  python -m analyses.tournament_deep_analysis <run_dir> [seed_base]

Metrics + offsets cited from observation.rs (see fullcatan_deep_analysis.py).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from catan_bot import _engine

CHANCE_BIT = 0x80000000

SETTLE_RANGE = (0, 54)
CITY_RANGE = (54, 108)
ROAD_RANGE = (108, 180)
ROBBER_RANGE = (180, 199)
ENDTURN = 204
TRADE_BANK_RANGE = (206, 226)
BUY_DEV = 226
PLAY_KNIGHT = 227
PROPOSE_TRADE_RANGE = (260, 280)

RES_NAMES = ["wood", "brick", "sheep", "wheat", "ore"]

SCALAR_VP = 8
SCALAR_LR_LEN = 26
SCALAR_KNIGHTS = 30
SCALAR_LR_HOLDER = 52
SCALAR_LA_HOLDER = 53


def _categorize(a: int) -> str:
    if SETTLE_RANGE[0] <= a < SETTLE_RANGE[1]: return "settle"
    if CITY_RANGE[0] <= a < CITY_RANGE[1]: return "city"
    if ROAD_RANGE[0] <= a < ROAD_RANGE[1]: return "road"
    if ROBBER_RANGE[0] <= a < ROBBER_RANGE[1]: return "robber"
    if a == ENDTURN: return "endturn"
    if TRADE_BANK_RANGE[0] <= a < TRADE_BANK_RANGE[1]: return "trade_bank"
    if a == BUY_DEV: return "buy_dev"
    if a == PLAY_KNIGHT: return "play_knight"
    if PROPOSE_TRADE_RANGE[0] <= a < PROPOSE_TRADE_RANGE[1]: return "propose_trade"
    return "other"


def _build_vertex_to_hexes() -> dict[int, list[int]]:
    from catan_gnn.adjacency import HEX_TO_VERTICES
    v2h: dict[int, list[int]] = {v: [] for v in range(54)}
    for h, vs in enumerate(HEX_TO_VERTICES):
        for v in vs:
            v2h[int(v)].append(h)
    return v2h


V2H = _build_vertex_to_hexes()


def analyze_game(seed, action_history, seating, rot, winner_seat):
    eng = _engine.Engine.with_rules(int(seed), 10, True)
    obs0 = eng.observation_for(0)
    vf0 = obs0["vertex_features"]
    vertex_has_port = [
        any(float(vf0[v, 7 + k]) >= 0.5 for k in range(6)) for v in range(54)
    ]

    def _role_for(seat):
        return seating[(seat + rot) % 4]

    out = {r: {
        "n_settle_built": 0, "n_city_built": 0, "n_road_built": 0,
        "n_trade_bank": 0, "n_propose_trade": 0, "n_buy_dev": 0,
        "n_play_knight": 0, "settle_vertices": [], "settle_on_port": 0,
        "robber_moves": 0, "first_city_endturn": -1,
    } for r in seating}
    n_endturn_global = 0

    for i, a_raw in enumerate(action_history):
        a = int(a_raw)
        if a & CHANCE_BIT:
            try:
                eng.apply_chance_outcome(a & ~CHANCE_BIT)
            except Exception:
                return None
            continue
        try:
            cp = int(eng.current_player())
            role = _role_for(cp)
        except Exception:
            return None
        cat = _categorize(a)
        if i >= 16:
            if cat == "settle":
                out[role]["n_settle_built"] += 1
            elif cat == "city":
                out[role]["n_city_built"] += 1
                if out[role]["first_city_endturn"] < 0:
                    out[role]["first_city_endturn"] = n_endturn_global
            elif cat == "road":
                out[role]["n_road_built"] += 1
            elif cat == "trade_bank":
                out[role]["n_trade_bank"] += 1
            elif cat == "propose_trade":
                out[role]["n_propose_trade"] += 1
            elif cat == "buy_dev":
                out[role]["n_buy_dev"] += 1
            elif cat == "play_knight":
                out[role]["n_play_knight"] += 1
            elif cat == "robber":
                out[role]["robber_moves"] += 1
        if cat == "settle":
            out[role]["settle_vertices"].append(a)
            if vertex_has_port[a]:
                out[role]["settle_on_port"] += 1
        if cat == "endturn":
            n_endturn_global += 1
        try:
            eng.step(a)
        except Exception:
            return None

    for seat in range(4):
        role = _role_for(seat)
        try:
            scalars = eng.observation_for(seat)["scalars"]
            out[role]["lr_length"] = float(scalars[SCALAR_LR_LEN]) * 15.0
            out[role]["knights_played"] = float(scalars[SCALAR_KNIGHTS]) * 14.0
            out[role]["lr_holder"] = float(scalars[SCALAR_LR_HOLDER]) >= 0.5
            out[role]["la_holder"] = float(scalars[SCALAR_LA_HOLDER]) >= 0.5
            out[role]["final_vp"] = int(scalars[SCALAR_VP])
        except Exception:
            pass

    out["_winner_role"] = _role_for(winner_seat) if winner_seat >= 0 else None
    out["_game_length"] = len(action_history)
    return out


def load_tournament(run_dir: Path):
    """Returns (df, seating, labels, seed_base). Handles both layouts."""
    cfgs = list(run_dir.rglob("config.json"))
    cfg = json.loads(cfgs[0].read_text())
    seating = cfg["seating"]
    labels = {seating[0]: cfg.get("label_a", seating[0]),
              seating[1]: cfg.get("label_b", seating[1]),
              seating[2]: cfg.get("label_c", seating[2]),
              seating[3]: cfg.get("label_d", seating[3]) or seating[3]}
    seed_base = int(cfg.get("seed_base", 0))
    parqs = list(run_dir.rglob("games*.parquet"))
    df = pd.concat([pq.read_table(str(p)).to_pandas() for p in parqs],
                   ignore_index=True)
    return df, seating, labels, seed_base


def main():
    run_dir = Path(sys.argv[1])
    df, seating, labels, seed_base = load_tournament(run_dir)
    if len(sys.argv) > 2:
        seed_base = int(sys.argv[2])
    print(f"### {run_dir.name}")
    print(f"games={len(df)}  seating={seating}")
    print(f"labels={[labels[r] for r in seating]}  seed_base={seed_base}\n")

    per_game = []
    fails = 0
    for _, row in df.iterrows():
        rot = (int(row["seed"]) - seed_base) // 10000
        if not (0 <= rot < 4):
            rot = 0
        res = analyze_game(int(row["seed"]), row["action_history"], seating,
                           rot, int(row["winner"]))
        if res is None:
            fails += 1
        else:
            per_game.append(res)
    print(f"analyzed {len(per_game)}/{len(df)} ({fails} replay failures)\n")

    # Winrate
    wins = defaultdict(int)
    for g in per_game:
        if g["_winner_role"]:
            wins[g["_winner_role"]] += 1
    n = len(per_game)
    print("WINRATE")
    for r in seating:
        print(f"  {labels[r]:<22} {wins[r]:3} ({100*wins[r]/n:5.1f}%)")
    print()

    def bucket_games(role, which):
        if which == "overall":
            return per_game
        if which == "in_wins":
            return [g for g in per_game if g["_winner_role"] == role]
        return [g for g in per_game if g["_winner_role"] != role]

    def mean(role, key, which):
        gs = bucket_games(role, which)
        vals = [g[role].get(key, 0) for g in gs]
        return np.mean(vals) if vals else 0.0

    # 1. BUILD DYNAMICS
    print("1. BUILD DYNAMICS (per game, post-setup)")
    for which in ["overall", "in_wins"]:
        print(f"  --- {which} ---")
        print(f"    {'role':<22}{'roads/g':>8}{'settle/g':>9}{'cities/g':>9}{'dev/g':>7}{'knights/g':>10}")
        for r in seating:
            gs = bucket_games(r, which)
            if not gs:
                print(f"    {labels[r]:<22}  (no games)")
                continue
            print(f"    {labels[r]:<22}{mean(r,'n_road_built',which):>8.2f}"
                  f"{mean(r,'n_settle_built',which):>9.2f}"
                  f"{mean(r,'n_city_built',which):>9.2f}"
                  f"{mean(r,'n_buy_dev',which):>7.2f}"
                  f"{mean(r,'n_play_knight',which):>10.2f}")
    print()

    # 2. BONUS ECONOMY
    print("2. BONUS ECONOMY (final state)")
    print(f"  {'role':<22}{'mean_LR':>8}{'mean_knights':>13}{'%LR_held':>9}{'%LA_held':>9}{'mean_VP':>8}")
    for r in seating:
        pct_lr = 100 * sum(1 for g in per_game if g[r].get("lr_holder")) / n
        pct_la = 100 * sum(1 for g in per_game if g[r].get("la_holder")) / n
        print(f"  {labels[r]:<22}{mean(r,'lr_length','overall'):>8.2f}"
              f"{mean(r,'knights_played','overall'):>13.2f}"
              f"{pct_lr:>8.1f}%{pct_la:>8.1f}%{mean(r,'final_vp','overall'):>8.2f}")
    print()

    # 2b. Bonus holding IN WINS
    print("2b. BONUS in WINS (% of this role's wins that held the bonus)")
    print(f"  {'role':<22}{'wins':>5}{'%LR_in_wins':>13}{'%LA_in_wins':>13}{'mean_VP_win':>12}")
    for r in seating:
        ws = bucket_games(r, "in_wins")
        if not ws:
            print(f"  {labels[r]:<22}    0   (no wins)")
            continue
        plr = 100 * sum(1 for g in ws if g[r].get("lr_holder")) / len(ws)
        pla = 100 * sum(1 for g in ws if g[r].get("la_holder")) / len(ws)
        mvp = np.mean([g[r].get("final_vp", 0) for g in ws])
        print(f"  {labels[r]:<22}{len(ws):>5}{plr:>12.1f}%{pla:>12.1f}%{mvp:>12.2f}")
    print()

    # 3. TRADE
    print("3. TRADE DYNAMICS (per game)")
    print(f"  {'role':<22}{'bank/g':>8}{'propose/g':>11}")
    for r in seating:
        print(f"  {labels[r]:<22}{mean(r,'n_trade_bank','overall'):>8.2f}"
              f"{mean(r,'n_propose_trade','overall'):>11.2f}")
    print()

    # 4. CLOSEOUT
    print("4. CLOSEOUT")
    print(f"  {'role':<22}{'wins':>5}{'median_len_in_wins':>20}{'mean_1st_city_endturn':>24}")
    for r in seating:
        ws = bucket_games(r, "in_wins")
        lens = sorted(g["_game_length"] for g in ws)
        med = lens[len(lens)//2] if lens else 0
        firsts = [g[r]["first_city_endturn"] for g in per_game
                  if g[r]["first_city_endturn"] >= 0]
        mfc = np.mean(firsts) if firsts else -1
        print(f"  {labels[r]:<22}{len(ws):>5}{med:>20}{mfc:>24.1f}")
    print()

    # 5. PORTS
    print("5. PORT USAGE (% of settlements on port vertices, overall)")
    print(f"  {'role':<22}{'settle_total':>13}{'on_port':>9}{'%on_port':>9}")
    for r in seating:
        tot = sum(len(g[r]["settle_vertices"]) for g in per_game)
        op = sum(g[r]["settle_on_port"] for g in per_game)
        print(f"  {labels[r]:<22}{tot:>13}{op:>9}{100*op/max(tot,1):>8.1f}%")
    print()

    # 6. RESOURCE SPECIALIZATION
    eng_s = _engine.Engine.with_rules(0, 10, True)
    hf = eng_s.observation_for(0)["hex_features"]
    hexres = []
    for h in range(19):
        if float(hf[h, 7]) >= 0.5:
            hexres.append(-1)
        else:
            hexres.append(next((r for r in range(5) if float(hf[h, r]) >= 0.5), -1))
    print("6. RESOURCE SPECIALIZATION (settlement-adjacent hex types)")
    print(f"  {'role':<22}{'wood%':>7}{'brick%':>7}{'sheep%':>7}{'wheat%':>7}{'ore%':>6}{'desert%':>8}")
    for r in seating:
        rc = [0]*5
        dc = 0
        tot = 0
        for g in per_game:
            for v in g[r]["settle_vertices"]:
                for h in V2H[v]:
                    res = hexres[h]
                    tot += 1
                    if res < 0:
                        dc += 1
                    else:
                        rc[res] += 1
        if tot == 0:
            continue
        p = [100*c/tot for c in rc]
        print(f"  {labels[r]:<22}{p[0]:>6.1f}%{p[1]:>6.1f}%{p[2]:>6.1f}%{p[3]:>6.1f}%{p[4]:>5.1f}%{100*dc/tot:>7.1f}%")
    print()


if __name__ == "__main__":
    sys.exit(main() or 0)
