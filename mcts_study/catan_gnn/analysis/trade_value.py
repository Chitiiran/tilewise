"""Phase 0 — measure ProposeTrade value-added rate over the 100k corpus.

Replays each recorded game through the v3 engine, intercepts every
ProposeTrade action (IDs 260..279), and records:
  - accepted: did the proposer's hand actually change after the action
    (cited rules.rs:347-374 — engine resolves trade by iterating
    opponents in seat order; if none has the requested resource, the
    action is a silent no-op)
  - value_adding: was the trade accepted AND did the proposer build
    settlement / city / road or buy a dev card before their next EndTurn

Aggregates per-corpus and per-give-resource / per-get-resource (no
per-pair heatmap, per user). Output: summary.json + decision printout.

Decision rule (per the loss-augmentation roadmap):
  value_adding_rate >= 60%   -> drop Cand 4 (trades are productive)
  30%-60%                    -> drop Cand 4 (user tie-break: keep trades)
  < 30%                      -> keep Cand 4 (ban ProposeTrade in training)

Phase 0 outcome (2026-05-10): 29.8% value-adding on 95k games / 4.58M
trades, with 100% acceptance rate. User decision: drop Cand 4 entirely;
the new VP-comparison rule (Cand 10) handles trade behavior implicitly.
See docs/superpowers/journals/2026-05-10-phase0-trade-value-summary.md.

Usage (run as a module from mcts_study/):
  python -m catan_gnn.analysis.trade_value [--shards N] [--out DIR]

Default scans all 12 worker shards under runs/v3/2026-05-05*-100k_w12/.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Action ID layout (cited catan_engine/src/actions.rs:49-58, 121-127).
ENDTURN = 204
ROLLDICE = 205
PROPOSE_TRADE_BASE = 260
PROPOSE_TRADE_END = 279  # inclusive
BUYDEVCARD = 226
BUILD_SETTLEMENT_RANGE = (0, 53)
BUILD_CITY_RANGE = (54, 107)
BUILD_ROAD_RANGE = (108, 179)
CHANCE_BIT = 0x8000_0000

# v3 rules: vp_target=5, bonuses_enabled=False (cited 100k corpus config).
V3_VP_TARGET = 5
V3_BONUSES_ENABLED = False

# Resource index layout (cited catan_engine/src/board.rs Resource enum order):
# all_hands()[p] = [wood, brick, sheep, wheat, ore].
RESOURCE_NAMES = ["wood", "brick", "sheep", "wheat", "ore"]


def _decode_propose_trade(action_id: int) -> tuple[int, int]:
    """Return (give_idx, get_compact_idx) from a ProposeTrade action_id.

    Cited actions.rs:151 — encoding is BASE + give*4 + get_compact, where
    get_compact in [0..3] skips the give resource itself. We don't need
    the get index for aggregation — give index is enough for per-resource
    breakdown.
    """
    off = action_id - PROPOSE_TRADE_BASE
    return off // 4, off % 4


def _is_value_adding_action(action_id: int) -> bool:
    """Per user: roads count too, even though they have 0 direct VP in v3."""
    if BUILD_SETTLEMENT_RANGE[0] <= action_id <= BUILD_SETTLEMENT_RANGE[1]:
        return True
    if BUILD_CITY_RANGE[0] <= action_id <= BUILD_CITY_RANGE[1]:
        return True
    if BUILD_ROAD_RANGE[0] <= action_id <= BUILD_ROAD_RANGE[1]:
        return True
    if action_id == BUYDEVCARD:
        return True
    return False


def classify_trades_in_game(seed: int, action_history) -> list[dict]:
    """Replay engine through action_history; return one record per
    ProposeTrade action.

    Each record:
      {"proposer": int 0..3,
       "action_id": int 260..279,
       "give": str,            # resource name
       "accepted": bool,
       "value_adding": bool}
    """
    from catan_bot import _engine

    eng = _engine.Engine.with_rules(seed, V3_VP_TARGET, V3_BONUSES_ENABLED)

    # Pending trades waiting for their proposer's next EndTurn to resolve
    # the value_adding flag. Indexed by proposer (0..3); each entry is a
    # list of trade records currently open for that player.
    pending_by_proposer: dict[int, list[dict]] = {0: [], 1: [], 2: [], 3: []}
    finalized: list[dict] = []

    for raw in action_history:
        if eng.is_terminal():
            break
        a = int(raw)
        if a & CHANCE_BIT:
            eng.apply_chance_outcome(a & 0x7FFF_FFFF)
            continue

        # Player and pre-action hand snapshot for trade detection.
        actor = int(eng.current_player())

        if PROPOSE_TRADE_BASE <= a <= PROPOSE_TRADE_END:
            hand_before = eng.all_hands()[actor].copy()
            eng.step(a)
            hand_after = eng.all_hands()[actor].copy()
            accepted = bool(np.any(hand_before != hand_after))
            give_idx, _ = _decode_propose_trade(a)
            rec = {
                "proposer": actor,
                "action_id": a,
                "give": RESOURCE_NAMES[give_idx],
                "accepted": accepted,
                "value_adding": False,  # default; promoted below if Build/Buy follows
            }
            if accepted:
                pending_by_proposer[actor].append(rec)
            else:
                # Silent failure — no chance to be value-adding.
                finalized.append(rec)
            continue

        # Value-adding trigger: the proposer executes a Build/Buy *before*
        # their next EndTurn. The actor here is the player taking the
        # action; only that player's pending trades can be promoted.
        if _is_value_adding_action(a) and pending_by_proposer[actor]:
            for rec in pending_by_proposer[actor]:
                rec["value_adding"] = True
            # Don't finalize yet — keep them open; promotion is monotone.

        if a == ENDTURN:
            # Finalize this player's pending trades.
            finalized.extend(pending_by_proposer[actor])
            pending_by_proposer[actor] = []

        eng.step(a)

    # Anything still open at end-of-history: game ended mid-turn (terminal
    # before the proposer's next EndTurn). Finalize as-is — the trade's
    # value_adding flag reflects whatever Build/Buy happened, and the
    # game ending in their favor means it counts.
    for plist in pending_by_proposer.values():
        finalized.extend(plist)

    return finalized


def _process_shard(args: tuple[str, int | None]) -> dict:
    """Worker for multiprocessing pool. Read one games.parquet, classify
    every game, return per-shard aggregated stats + raw records."""
    shard_path, max_games = args
    df = pq.read_table(shard_path).to_pandas()
    if max_games is not None:
        df = df.iloc[:max_games]
    records: list[dict] = []
    n_games = 0
    for _, row in df.iterrows():
        seed = int(row["seed"])
        ah = row["action_history"]
        try:
            recs = classify_trades_in_game(seed, ah)
        except Exception as e:
            print(f"[shard {shard_path}] seed={seed} failed: {e}", file=sys.stderr)
            continue
        records.extend(recs)
        n_games += 1
    return {"shard": shard_path, "n_games": n_games, "records": records}


def _aggregate(records: list[dict]) -> dict:
    """Compute summary statistics from a flat list of trade records."""
    n = len(records)
    n_accepted = sum(1 for r in records if r["accepted"])
    n_value = sum(1 for r in records if r["value_adding"])
    by_give: dict[str, dict] = {}
    for r in records:
        g = r["give"]
        d = by_give.setdefault(
            g, {"n": 0, "accepted": 0, "value_adding": 0}
        )
        d["n"] += 1
        d["accepted"] += int(r["accepted"])
        d["value_adding"] += int(r["value_adding"])
    return {
        "n_proposes_total": n,
        "n_accepted": n_accepted,
        "n_value_adding": n_value,
        "acceptance_rate": (n_accepted / n) if n else 0.0,
        "value_adding_rate": (n_value / n) if n else 0.0,
        "by_give": by_give,
    }


def _decision(value_adding_rate: float) -> str:
    if value_adding_rate >= 0.60:
        return "DROP_CAND_4 (trades are productive; >=60% value-adding)"
    if value_adding_rate >= 0.30:
        return "DROP_CAND_4 (user tie-break default in 30-60% band)"
    return "KEEP_CAND_4 (ban ProposeTrade in training; <30% value-adding)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--corpus-root",
        default="runs/v3/2026-05-05T05-50-e9_v3_data_gen_100k_w12/2026-05-05T09-50-e9_v3_data_gen",
        help="Root containing worker0..worker11 subdirs",
    )
    p.add_argument(
        "--max-games-per-shard", type=int, default=None,
        help="Cap games per shard (smoke testing)",
    )
    p.add_argument(
        "--shards", default="all",
        help="Comma-separated worker indexes, or 'all' (default)",
    )
    p.add_argument(
        "--out", default="runs/v3/trade_value_analysis_2026-05-10",
        help="Output directory for summary.json",
    )
    p.add_argument(
        "--pool", type=int, default=12,
        help="multiprocessing pool size (default 12 for 12 shards)",
    )
    args = p.parse_args()

    root = Path(args.corpus_root)
    if args.shards == "all":
        shard_dirs = sorted(root.glob("worker*"))
    else:
        idxs = [int(x) for x in args.shards.split(",")]
        shard_dirs = [root / f"worker{i}" for i in idxs]

    shard_files = []
    for d in shard_dirs:
        for f in d.glob("games.*.parquet"):
            shard_files.append(str(f))
    if not shard_files:
        print(f"No games.*.parquet found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(shard_files)} shard files")
    for f in shard_files:
        print(f"  {f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pool_args = [(f, args.max_games_per_shard) for f in shard_files]
    if args.pool == 1 or len(shard_files) == 1:
        per_shard = [_process_shard(a) for a in pool_args]
    else:
        with mp.Pool(min(args.pool, len(shard_files))) as pool:
            per_shard = pool.map(_process_shard, pool_args)
    elapsed = time.time() - t0

    all_records: list[dict] = []
    n_games_total = 0
    for s in per_shard:
        all_records.extend(s["records"])
        n_games_total += s["n_games"]

    summary = _aggregate(all_records)
    summary["n_games_total"] = n_games_total
    summary["n_shards"] = len(shard_files)
    summary["elapsed_seconds"] = elapsed
    summary["decision"] = _decision(summary["value_adding_rate"])
    summary["corpus_root"] = str(root)
    summary["max_games_per_shard"] = args.max_games_per_shard

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")

    # Pretty printout
    print("\n=== Trade-value analysis ===")
    print(f"Games scanned:       {summary['n_games_total']:,}")
    print(f"ProposeTrade actions: {summary['n_proposes_total']:,}")
    print(f"Accepted:             {summary['n_accepted']:,} "
          f"({100 * summary['acceptance_rate']:.1f}%)")
    print(f"Value-adding:         {summary['n_value_adding']:,} "
          f"({100 * summary['value_adding_rate']:.1f}%)")
    print(f"\nPer give-resource:")
    print(f"  {'resource':<8} {'n':>8} {'accept%':>8} {'value%':>8}")
    for res in RESOURCE_NAMES:
        if res not in summary["by_give"]:
            continue
        d = summary["by_give"][res]
        n = d["n"]
        ar = (100 * d["accepted"] / n) if n else 0.0
        vr = (100 * d["value_adding"] / n) if n else 0.0
        print(f"  {res:<8} {n:>8} {ar:>7.1f}% {vr:>7.1f}%")
    print(f"\nElapsed: {elapsed:.1f} s ({n_games_total/elapsed:.0f} games/s)")
    print(f"\nDecision: {summary['decision']}")


if __name__ == "__main__":
    main()
