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
    games_per_iter: int = 1000         # candidate self-play, ALWAYS generated
                                       # (redesign 2026-06-14; ~7.4h/iter)
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
    arena_games: int = 300             # 65% bar needs the power: CI ±6.6pp
                                       # (vs ±11pp at 120). 4 rotations x 75.
    promote_threshold: float = 0.65    # strictly-greater; demand clear improvement
    max_iters_per_model: int = 10      # stop after 10 iters w/o promotion -> user
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
    # Per-WORKER self-play wall-clock cap. At ~142 games/hr across 6 workers,
    # 1000 games needs ~7h; the original 6h (21600s) cap raced the quota and the
    # M4 floor tripped at 731/1000 (2026-06-15). 9h leaves headroom — and
    # because games persist per-seed, a worker that still hits the cap just
    # resumes toward the quota on the next launch.
    selfplay_worker_max_seconds: int = 32400
    # Per-GAME wall-clock cap for self-play. A pathological game can churn for an
    # hour+ under the 200k-step cap, stalling its worker (24 concurrent games
    # finalize together) and the whole stage — production iter_6 needed a manual
    # straggler-kill (2026-06-15). MEASURED: normal games avg ~190s wall-clock at
    # concurrency=24; stragglers ran 20-60+ min. 600s (~3.2x mean, == the arena's
    # own cap) clears virtually all healthy games yet kills stragglers in 10min.
    # The cut game is recorded timed_out (data preserved, not silently dropped).
    selfplay_game_deadline_seconds: float = 600.0
    # Anchor (absolute calibration vs LookV3)
    anchor_every: int = 5
    anchor_games: int = 60
    # Rules — full Catan throughout (user decision; inversion lesson)
    vp_target: int = 10
    bonuses: bool = True
    # Net
    hidden_dim: int = 128
    num_layers: int = 4
    # --- daily runner (spec 2026-06-13) ---
    fresh_ratio: float = 0.70          # current-champion share of the window
    rules_id: str = "v3-full"          # engine-rules version tag on run dirs
    worker_procs_max: int = 7          # VRAM cap on the 4 GB GTX 1650 (GPU-only)
    per_proc_vram_mb: float = 535.0    # measured CUDA-context cost per proc
    worker_nice: int = 10              # low OS priority -> yields to foreground
    min_fast_gb: float = 10.0          # preflight: fast-disk free floor
    min_hdd_gb: float = 20.0           # preflight: HDD (archive) free floor
    stagnation_holds: int = 5          # N trailing holds -> surface + stop day
    archive_root: str = "/mnt/d/catan_az_archive"   # HDD archive target
    dashboard_port: int = 8099

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "AzConfig":
        data = json.loads(Path(path).read_text())
        # dataclass __init__ raises TypeError on unexpected kwargs — exactly
        # the typo guard we want; just pass everything through.
        return cls(**data)
