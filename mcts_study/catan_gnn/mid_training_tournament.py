"""Mid-training tournament hook for the loss-augmentation roadmap.

Every N epochs during training we save a checkpoint and run a 120-game
shared-seed tournament vs LookaheadV3 to get a real PureGnn winrate.
Used as the early-stopping signal in place of val_top1 (which goes
DOWN by design once the policy is pulled toward VP-economy targets and
away from raw MCTS visit counts; cited v3.6 production journal showed
winrate doubling while val_top1 stayed at 0.18-0.19).

This module is a thin wrapper around e10_v3_tournament.main that:
  - runs the tournament with parallel workers + GPU (see STANDARD_*
    defaults below for the values verified end-to-end 2026-05-11)
  - parses the resulting worker*/games.rot=*.parquet files
  - returns PureGnn winrate + auxiliary role counts
  - exposes a pure decision rule (should_stop_for_drop) for early-stop

Tested against a known 120-game fixture at
  runs/v3/grid_pass3_lastepoch/h32_l2/2026-05-04T21-53-e10_v3_tournament/
with verified counts PureGnn=7, LookaheadV3=110, GnnMcts=2, Random=1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.parquet as pq


# Cited from catan_mcts/experiments/e10_v3_tournament.py:48.
ROLES = ["GnnMcts", "PureGnn", "LookaheadMctsV3", "Random"]


# === STANDARD TOURNAMENT CONFIG (Phase 1 baseline & all subsequent runs) ===
# These constants encode the apples-to-apples comparison the loss-aug
# roadmap requires. ALL Phase 1 mid-tournaments and standalone validation
# runs MUST use these values. Cited 2026-05-11 from pass-3 ground truth:
#
#   sims=100, lookahead_depth=10, base_sims_v3=200, vp_target=5,
#   bonuses=False, num_games_per_seating=30 (4 rotations -> 120 games),
#   seed_base=19_000_000 (the canonical Phase 1 seed range; pass-3 used
#   this same range so per-seed comparisons are valid).
#
# Pass-3 cited baselines on these seeds:
#   h32_l2 epoch 20 (val_top1≈0.18):   PureGnn 7/120 = 5.83%
#   h128_l4 epoch 20 (val_top1≈0.18):  PureGnn 3/120 = 2.50%
#
# Worker config: workers=10 + device="cuda" verified to run 120 games
# in ~30-40 min on GTX 1650 4GB. Each spawn worker loads ~150 MB into
# VRAM; 10 workers x 150 MB = 1.5 GB, leaving 2.5 GB headroom in 4 GB.
STANDARD_NUM_GAMES_PER_SEATING = 30
STANDARD_SIMS = 100
STANDARD_LOOKAHEAD_DEPTH = 10
STANDARD_BASE_SIMS_V3 = 200
STANDARD_SEED_BASE = 19_000_000
STANDARD_WORKERS = 10
STANDARD_DEVICE = "cuda"
STANDARD_MAX_SECONDS = 600.0


def _seat_to_role(seat: int, rot: int) -> str:
    """Same convention as scratch_pick_matches.py / dashboard: seat s in
    rotation r plays ROLES[(s + r) % 4]."""
    return ROLES[(seat + rot) % 4]


def summarize_tournament_run_dir(run_dir: Path) -> dict:
    """Parse all worker*/games.rot=*.parquet files under run_dir, compute
    per-role win counts and PureGnn winrate.

    Returns a dict with keys:
        total_games, pure_gnn_wins, pure_gnn_winrate, gnn_mcts_wins,
        lookahead_v3_wins, random_wins, no_winner_games

    Raises FileNotFoundError if no parquet files are found.
    """
    run_dir = Path(run_dir)
    parquets = sorted(run_dir.rglob("games.rot=*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"No games.rot=*.parquet files under {run_dir}"
        )

    frames = []
    for parq in parquets:
        # File name: games.rot=<N>.parquet  (cited tournament writer
        # naming in catan_mcts.recorder.SelfPlayRecorder).
        try:
            rot_str = parq.name.split(".")[1]
            rot = int(rot_str.split("=")[1])
        except (IndexError, ValueError) as e:
            raise ValueError(f"Cannot parse rot from {parq.name}: {e}")
        df = pq.read_table(parq).to_pandas()
        df["rot"] = rot
        frames.append(df)
    g = pd.concat(frames, ignore_index=True)

    g["winner_role"] = [
        _seat_to_role(int(w), int(r)) if w >= 0 else "DRAW"
        for w, r in zip(g["winner"], g["rot"])
    ]

    counts = g["winner_role"].value_counts().to_dict()
    pure_gnn_wins = int(counts.get("PureGnn", 0))
    gnn_mcts_wins = int(counts.get("GnnMcts", 0))
    lookahead_v3_wins = int(counts.get("LookaheadMctsV3", 0))
    random_wins = int(counts.get("Random", 0))
    no_winner_games = int(counts.get("DRAW", 0))
    total_games = int(len(g))

    return {
        "total_games": total_games,
        "pure_gnn_wins": pure_gnn_wins,
        "pure_gnn_winrate": (pure_gnn_wins / total_games) if total_games else 0.0,
        "gnn_mcts_wins": gnn_mcts_wins,
        "lookahead_v3_wins": lookahead_v3_wins,
        "random_wins": random_wins,
        "no_winner_games": no_winner_games,
    }


def should_stop_for_drop(
    *,
    history: Sequence[int],
    current_wins: int,
    total_games: int,
    drop_threshold: int = 3,
) -> bool:
    """Decision rule: stop iff this tournament's PureGnn wins dropped by
    >= drop_threshold games compared to the IMMEDIATELY-PRECEDING
    tournament. First tournament (empty history) always continues —
    nothing to compare against yet.

    Args:
        history: list of past PureGnn win counts, oldest first. The
            most recent is `history[-1]`.
        current_wins: PureGnn win count from the latest tournament.
        total_games: total games in the latest tournament (used to
            sanity-check; not used in the decision itself since the
            threshold is in absolute games per the plan).
        drop_threshold: number of games drop required to stop. Default 3
            per the plan (">= 3-game drop, >= 2.5pp at 120 games").

    Returns:
        True iff caller should stop training; False to continue.
    """
    if not history:
        return False
    prev = history[-1]
    drop = prev - current_wins
    return drop >= drop_threshold


@dataclass
class MidTrainingTournamentResult:
    """Structured return from run_mid_training_tournament."""
    epoch: int
    run_dir: Path
    summary: dict          # output of summarize_tournament_run_dir
    elapsed_seconds: float


def run_mid_training_tournament(
    *,
    epoch: int,
    checkpoint_path: Path,
    out_root: Path,
    hidden_dim: int,
    num_layers: int,
    num_games_per_seating: int = STANDARD_NUM_GAMES_PER_SEATING,
    sims: int = STANDARD_SIMS,
    lookahead_depth: int = STANDARD_LOOKAHEAD_DEPTH,
    base_sims_v3: int = STANDARD_BASE_SIMS_V3,
    seed_base: int = STANDARD_SEED_BASE,
    max_seconds: float = STANDARD_MAX_SECONDS,
    device: str = STANDARD_DEVICE,
    workers: int = STANDARD_WORKERS,
) -> MidTrainingTournamentResult:
    """Run one mid-training tournament with the given checkpoint.

    Default args produce a 120-game tournament (30 × 4 rotations) using
    seed_base=20M (fresh range, per plan; doesn't overlap pass-3's 19M).

    workers default 8: e10_v3_tournament uses get_context("spawn"). Each
    worker is a fresh Python process; the training process's CachedDataset
    is NOT replicated (spawn re-imports modules from scratch). Each worker
    loads the model checkpoint (~3-7 MB) and runs MCTS on CPU/GPU
    independently. With sims=100 the bottleneck is PyO3 calls (CPU-bound),
    so CPU workers parallelize ~linearly with cores. Use device=cpu for
    max throughput at the mid-tournament. The training process itself is
    paused during the tournament, so GPU contention is not a concern.
    """
    import time as _time
    # Import here so test files that only exercise the parser don't
    # transitively need open_spiel / torch / GnnEvaluator.
    from catan_mcts.experiments.e10_v3_tournament import main as e10_main

    t0 = _time.perf_counter()
    out_dir = e10_main(
        out_root=out_root,
        checkpoint=checkpoint_path,
        num_games_per_seating=num_games_per_seating,
        sims=sims,
        lookahead_depth=lookahead_depth,
        base_sims_v3=base_sims_v3,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seed_base=seed_base,
        max_seconds=max_seconds,
        vp_target=5,
        bonuses=False,
        resume=False,
        workers=workers,
        device=device,
    )
    elapsed = _time.perf_counter() - t0
    summary = summarize_tournament_run_dir(out_dir)
    return MidTrainingTournamentResult(
        epoch=epoch, run_dir=out_dir, summary=summary,
        elapsed_seconds=elapsed,
    )
