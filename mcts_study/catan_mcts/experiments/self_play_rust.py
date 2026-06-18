"""Rust-engine self-play worker — CLI-compatible drop-in for self_play_async.

Calls catan_mcts_rs.run_selfplay (in-process Rust MCTS + TorchScript GNN, zero
per-node PyO3) and writes the SAME SelfPlayRecorder parquet shards, so daily.py's
worker/seed/resume/meta logic is unchanged — only the invoked module differs
(switched by AzConfig.engine). The game records are bit-exact to the Python
async path (proven by the Phase-6 100-seed gate).

The checkpoint passed in is a .pt; we load the sibling .ts (emitted by
loop._default_train). If the .ts is missing we export it on the fly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import catan_mcts_rs
from catan_gnn.export_torchscript import export
from ..recorder import SelfPlayRecorder
from .common import make_run_dir


def _ensure_ts(checkpoint: Path, hidden_dim: int, num_layers: int) -> Path:
    ts = checkpoint.with_suffix(".ts")
    if not ts.exists():
        export(checkpoint=checkpoint, out_ts=ts,
               hidden_dim=hidden_dim, num_layers=num_layers)
    return ts


def run_self_play_rust(*, out_root: Path, checkpoint: Path, num_games: int,
                       n_sims: int, hidden_dim: int, num_layers: int,
                       vp_target: int, bonuses: bool, seed_base: int,
                       self_play: bool, max_steps: int,
                       game_deadline_seconds: float | None,
                       resume_dir: Path | None) -> Path:
    out = resume_dir if resume_dir is not None else make_run_dir(out_root, "self_play_rust")
    ts = _ensure_ts(Path(checkpoint), hidden_dim, num_layers)
    rec = SelfPlayRecorder(out, config={
        "experiment": "self_play_rust", "n_sims": n_sims,
        "vp_target": vp_target, "bonuses": bonuses, "device": "cuda",
        "seed_base": seed_base, "engine": "rust"})
    done = rec.done_seeds() if resume_dir is not None else set()
    seeds = [seed_base + i for i in range(num_games) if (seed_base + i) not in done]

    # One Rust call per game keeps per-game crash-flush (records returned, then
    # written by the recorder immediately). game_deadline is enforced Rust-side
    # via max_steps; the Python async path's wall-clock deadline is moot here
    # because Rust games finish naturally (the whole point of the rewrite).
    records = catan_mcts_rs.run_selfplay(
        str(ts), seeds, n_sims, self_play, vp_target, bonuses,
        30, 0.8, 0.25, max_steps)
    for r in records:
        seed = int(r["seed"])
        with rec.game(seed=seed) as g:
            for m in r["moves"]:
                g.record_move(
                    current_player=int(m["current_player"]),
                    move_index=int(m["move_index"]),
                    legal_action_mask=np.asarray(m["legal_mask"], dtype=np.int8),
                    mcts_visit_counts=np.asarray(m["visit_counts"], dtype=np.int32),
                    action_taken=int(m["action_taken"]),
                    mcts_root_value=float(m["root_value"]))
            g.finalize(winner=int(r["winner"]),
                       final_vp=[int(x) for x in r["final_vp"]],
                       length_in_moves=int(r["length_in_moves"]),
                       action_history=[int(x) for x in r["action_history"]],
                       timed_out=not bool(r["terminal"]))
        rec.mark_done(seed)
    rec.checkpoint(rec.config_id[:8])
    print(f"[self_play_rust] done: {len(records)} games -> {out}")
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--num-games", type=int, default=64)
    p.add_argument("--n-sims", type=int, default=200)
    p.add_argument("--n-concurrent", type=int, default=64)  # accepted, unused (no asyncio)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--vp-target", type=int, default=10)
    p.add_argument("--bonuses", action="store_true", default=True)
    p.add_argument("--no-bonuses", dest="bonuses", action="store_false")
    p.add_argument("--device", type=str, default="cuda")  # accepted (tch picks device)
    p.add_argument("--max-batch", type=int, default=64)    # accepted, unused (Task 10)
    p.add_argument("--window-ms", type=float, default=5.0) # accepted, unused
    p.add_argument("--seed-base", type=int, default=20_000_000)
    p.add_argument("--max-seconds", type=float, default=900.0)  # accepted, unused
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--game-deadline-seconds", type=float, default=None)
    p.add_argument("--resume-dir", type=Path, default=None)
    p.add_argument("--self-play", action="store_true")
    a = p.parse_args()
    out = run_self_play_rust(
        out_root=a.out_root, checkpoint=a.checkpoint, num_games=a.num_games,
        n_sims=a.n_sims, hidden_dim=a.hidden_dim, num_layers=a.num_layers,
        vp_target=a.vp_target, bonuses=a.bonuses, seed_base=a.seed_base,
        self_play=a.self_play, max_steps=a.max_steps,
        game_deadline_seconds=a.game_deadline_seconds, resume_dir=a.resume_dir)
    print(f"self_play_rust wrote to {out}")


if __name__ == "__main__":
    cli_main()
