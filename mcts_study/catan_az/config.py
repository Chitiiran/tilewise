"""AzConfig — every knob of the AZ loop, JSON round-trippable.

Defaults are spec §7 (2026-06-11-az-loop-design.md). from_json rejects
unknown keys so a typo'd config fails loudly instead of silently using a
default (the kind of silent mis-config that cost runs before).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AzConfig:
    # Self-play
    games_per_iter: int = 400          # ≈2.5-3h at measured ~150 games/h
    sims: int = 200
    dirichlet_alpha: float = 0.8
    dirichlet_eps: float = 0.25
    temp_moves: int = 30
    n_procs: int = 5                   # until the B1 spike verdict
    n_concurrent: int = 24             # coroutines per proc
    max_batch: int = 24
    # Buffer
    window_games: int = 1200           # ~3 iterations
    # Train
    lr: float = 2e-4
    max_epochs: int = 4
    early_stop: bool = True
    policy_sharpen: float = 1.0        # canonical targets; sharpening = flagged experiment
    batch_size: int = 64
    # Arena
    arena_games: int = 120             # 4 rotations x 30 shared seeds
    promote_threshold: float = 0.55    # strictly-greater promotes (AGZ gate)
    arena_timeout_rate_max: float = 0.05   # legacy; surfaced, no longer a gate
    # Validity guards under VP-leader tiebreak (2026-06-13): a timed-out game
    # is decided by VP leader, so the gate keys on genuine no-signal games
    # (VP ties = draws) and on having enough decisive games to trust the
    # winrate — not on how many stalled past the wall-clock cap.
    arena_max_draw_rate: float = 0.40      # too many VP ties -> untrustworthy
    arena_min_decisive: int = 40           # need >=N decided games for a verdict
    arena_sims: int = 200
    # Per-game wall-clock cap: bounds the rare pathological game that crawls
    # toward the 200k step cap and would otherwise hold the whole arena's
    # gather() hostage (2026-06-13 incident). 600s >> a normal ~2min game, so
    # it only ever fires on a genuine straggler. Counts as a timeout (the
    # <5% timeout-rate gate still guards verdict trustworthiness).
    arena_game_max_seconds: float = 600.0
    # Anchor (absolute calibration vs LookV3)
    anchor_every: int = 5
    anchor_games: int = 60
    # Rules — full Catan throughout (user decision; inversion lesson)
    vp_target: int = 10
    bonuses: bool = True
    # Net
    hidden_dim: int = 128
    num_layers: int = 4

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "AzConfig":
        data = json.loads(Path(path).read_text())
        # dataclass __init__ raises TypeError on unexpected kwargs — exactly
        # the typo guard we want; just pass everything through.
        return cls(**data)
