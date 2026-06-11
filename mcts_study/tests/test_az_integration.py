"""Micro-iteration through the REAL default stages (spec §6 integration).

Proves the plumbing — self_play_async -> window -> train_main -> run_arena
-> ladder/journal — not strength. Tiny everything: scratch h32/L2 net,
vp_target=5 (fast games; this is a plumbing test, the loop itself runs
full Catan), sims=4, CPU.
"""
from __future__ import annotations

import dataclasses

import pytest

pytestmark = pytest.mark.slow


def test_micro_iteration_end_to_end(tmp_path):
    import torch
    from catan_az.config import AzConfig
    from catan_az.ladder import Ladder
    from catan_az.loop import run_iteration
    from catan_gnn.gnn_model import GnnModel

    # Scratch champion checkpoint (random weights — plumbing, not strength).
    champ = tmp_path / "scratch_champ.pt"
    torch.save(GnnModel(hidden_dim=32, num_layers=2).state_dict(), champ)

    cfg = AzConfig(
        games_per_iter=4, sims=4, n_concurrent=4, max_batch=4,
        window_games=4, lr=1e-3, max_epochs=1, batch_size=32,
        arena_games=4, arena_sims=4, promote_threshold=0.55,
        vp_target=5, bonuses=False, hidden_dim=32, num_layers=2,
    )
    loop_root = tmp_path / "loop"
    loop_root.mkdir()
    Ladder(loop_root, champion_checkpoint=str(champ), champion_name="scratch")

    # CPU-device stage wrappers (defaults assume cuda).
    from catan_az import loop as loop_mod

    def cpu_selfplay(cfg, iter_dir, champion_ckpt):
        from catan_mcts.experiments.self_play_async import run_self_play
        out = run_self_play(
            out_root=iter_dir, checkpoint=champion_ckpt,
            num_games=cfg.games_per_iter, n_sims=cfg.sims,
            n_concurrent=cfg.n_concurrent, hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers, vp_target=cfg.vp_target,
            bonuses=cfg.bonuses, device="cpu", max_batch=cfg.max_batch,
            max_seconds=900, seed_base=77_000, self_play=True)
        return [out]

    def cpu_train(cfg, run_dirs, iter_dir, init_ckpt):
        from catan_gnn.train import train_main
        out_dir = iter_dir / "training"
        train_main(run_dirs=run_dirs, out_dir=out_dir,
                   hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
                   epochs=cfg.max_epochs, batch_size=cfg.batch_size,
                   lr=cfg.lr, init_from=init_ckpt, device="cpu")
        best = out_dir / "checkpoint_best.pt"
        assert best.exists()
        return best

    def cpu_arena(cfg, cand, champ_ckpt, iter_dir):
        from catan_az.arena import run_arena
        return run_arena(candidate_ckpt=cand, champion_ckpt=champ_ckpt,
                         cfg=cfg, out_dir=iter_dir / "arena",
                         seed_base=88_000, device="cpu", n_concurrent=4)

    verdict = run_iteration(cfg, loop_root, 1, selfplay_fn=cpu_selfplay,
                            train_fn=cpu_train, arena_fn=cpu_arena)

    assert verdict in ("promote", "hold", "invalid")
    assert (loop_root / "journal.csv").exists()
    assert (loop_root / "iter_1" / "SELFPLAY.done").exists()
    assert (loop_root / "iter_1" / "TRAIN.done").exists()
    shards = list((loop_root / "iter_1").rglob("moves*.parquet"))
    assert shards, "self-play wrote no move shards"
    if verdict == "promote":
        assert (loop_root / "checkpoints" / "az_iter_1.pt").exists()
