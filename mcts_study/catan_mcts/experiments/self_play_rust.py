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

# NOTE: `catan_mcts_rs` (tch-linked) is imported LAZILY inside the run function,
# not at module top — importing it requires libtorch on LD_LIBRARY_PATH, which
# the daily worker env (and pytest_mctsrs.sh) provide but plain `import` does
# not. Lazy import keeps this module importable for unit tests (which stub the
# engine) and surfaces a clear error only when actually generating games.
from catan_gnn.export_torchscript import export, export_batched
from ..recorder import SelfPlayRecorder
from .common import make_run_dir

# Lazily-bound engine handle. Tests monkeypatch this with a fake; production
# code calls _engine() which imports the real extension on first use.
catan_mcts_rs = None


def _engine():
    global catan_mcts_rs
    if catan_mcts_rs is None:
        import catan_mcts_rs as _rs
        catan_mcts_rs = _rs
    return catan_mcts_rs

# Cross-game batch width (Task 10). The batched .ts is traced at this fixed
# B_MAX; the scheduler pools up to B_MAX concurrent games' leaves per GPU
# forward. 32 matched the measured throughput knee on the GTX 1650.
B_MAX = 32


def _infer_device() -> str:
    """The device the rust engine will run on (CUDA if available). The .ts MUST
    be traced on this device — trace bakes the device of internally-constructed
    tensors, so a CPU-traced .ts fails on CUDA with a scatter device-mismatch."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_ts(checkpoint: Path, hidden_dim: int, num_layers: int) -> Path:
    dev = _infer_device()
    ts = checkpoint.with_suffix(f".{dev}.ts")
    if not ts.exists():
        export(checkpoint=checkpoint, out_ts=ts,
               hidden_dim=hidden_dim, num_layers=num_layers, device=dev)
    return ts


def _ensure_batched_ts(checkpoint: Path, hidden_dim: int, num_layers: int,
                       b_max: int = B_MAX) -> Path:
    dev = _infer_device()
    # Encode b_max in the filename so a different B_MAX produces a DIFFERENT file
    # (re-export) rather than silently loading a wrong-width batched .ts. Must
    # match loop._default_train's naming for the export to be reused.
    bts = checkpoint.with_suffix(f".{dev}.b{b_max}.batch.ts")
    if not bts.exists():
        export_batched(checkpoint=checkpoint, out_ts=bts, hidden_dim=hidden_dim,
                       num_layers=num_layers, b_max=b_max, device=dev)
    return bts


def _paused(pause_dir) -> bool:
    """PAUSE sentinel: a `PAUSE` file in pause_dir (or its parent loop root)."""
    if pause_dir is None:
        return False
    p = Path(pause_dir)
    return (p / "PAUSE").exists() or (p.parent / "PAUSE").exists() \
        or (p.parent.parent / "PAUSE").exists()


def _write_record(rec, r) -> None:
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


def run_self_play_rust(*, out_root: Path, checkpoint: Path, num_games: int,
                       n_sims: int, hidden_dim: int, num_layers: int,
                       vp_target: int, bonuses: bool, seed_base: int,
                       self_play: bool, max_steps: int,
                       game_deadline_seconds: float | None,
                       resume_dir: Path | None,
                       n_concurrent: int = 256,
                       b_max: int = B_MAX,
                       pause_dir: Path | None = None) -> Path:
    out = resume_dir if resume_dir is not None else make_run_dir(out_root, "self_play_rust")
    bts = _ensure_batched_ts(Path(checkpoint), hidden_dim, num_layers, b_max=b_max)
    rec = SelfPlayRecorder(out, config={
        "experiment": "self_play_rust", "n_sims": n_sims,
        "vp_target": vp_target, "bonuses": bonuses, "device": "cuda",
        "seed_base": seed_base, "engine": "rust", "b_max": b_max,
        "n_concurrent": n_concurrent})
    done = rec.done_seeds()  # always resume-aware (skip already-done seeds)
    seeds = [seed_base + i for i in range(num_games) if (seed_base + i) not in done]

    # PAUSABLE + reproducible: process seeds in n_concurrent-sized CHUNKS. Each
    # chunk is one big batched run (full GPU fill within the chunk); records are
    # flushed + marked done after each chunk. A PAUSE sentinel between chunks
    # stops cleanly, losing AT MOST one chunk's in-flight games. Resume re-runs
    # with the remaining (not-done) seeds — byte-identical games because each
    # game's RNG is seeded ONLY by its own seed (chunking does not change any
    # game's record). pause_dir defaults to the run dir.
    pause_dir = pause_dir if pause_dir is not None else out
    total = 0
    for start in range(0, len(seeds), n_concurrent):
        if _paused(pause_dir):
            print(f"[self_play_rust] PAUSED after {total} games "
                  f"({len(seeds) - total} remaining) -> resume re-runs this dir")
            return out
        chunk = seeds[start:start + n_concurrent]
        # net_ts (plain B=1) is unused by the batched path — pass bts for both;
        # the batched_ts arg is what run_selfplay loads (L2: skip the dead B=1
        # export entirely for self-play).
        records = _engine().run_selfplay(
            str(bts), chunk, n_sims, self_play, vp_target, bonuses,
            30, 0.8, 0.25, max_steps, str(bts), b_max)
        for r in records:
            _write_record(rec, r)
        total += len(records)
        print(f"[self_play_rust] chunk done: {total}/{len(seeds)} games", flush=True)
    rec.checkpoint(rec.config_id[:8])
    print(f"[self_play_rust] done: {total} games -> {out}")
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--num-games", type=int, default=64)
    p.add_argument("--n-sims", type=int, default=200)
    p.add_argument("--n-concurrent", type=int, default=256)  # games per batched chunk (Task 10)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--vp-target", type=int, default=10)
    p.add_argument("--bonuses", action="store_true", default=True)
    p.add_argument("--no-bonuses", dest="bonuses", action="store_false")
    p.add_argument("--device", type=str, default="cuda")  # accepted (tch picks device)
    p.add_argument("--max-batch", type=int, default=32)    # rust batched B_MAX (GNN forward cap)
    p.add_argument("--window-ms", type=float, default=5.0) # accepted, unused
    p.add_argument("--seed-base", type=int, default=20_000_000)
    p.add_argument("--max-seconds", type=float, default=900.0)  # accepted, unused
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--game-deadline-seconds", type=float, default=None)
    p.add_argument("--resume-dir", type=Path, default=None)
    p.add_argument("--pause-dir", type=Path, default=None,
                   help="dir checked for a PAUSE sentinel between chunks")
    p.add_argument("--self-play", action="store_true")
    a = p.parse_args()
    out = run_self_play_rust(
        out_root=a.out_root, checkpoint=a.checkpoint, num_games=a.num_games,
        n_sims=a.n_sims, hidden_dim=a.hidden_dim, num_layers=a.num_layers,
        vp_target=a.vp_target, bonuses=a.bonuses, seed_base=a.seed_base,
        self_play=a.self_play, max_steps=a.max_steps,
        game_deadline_seconds=a.game_deadline_seconds, resume_dir=a.resume_dir,
        n_concurrent=a.n_concurrent, b_max=a.max_batch, pause_dir=a.pause_dir)
    print(f"self_play_rust wrote to {out}")


if __name__ == "__main__":
    cli_main()
