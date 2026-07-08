"""Isolate where pause/resume diverges: compare epoch-1 checkpoints (should be
identical) vs final."""
import tempfile
from pathlib import Path
import torch
from catan_gnn.train import train_main
from catan_mcts.experiments.e1_winrate_vs_random import main as e1


def w(ck):
    o = torch.load(ck, map_location="cpu", weights_only=False)
    return o["model_state"] if isinstance(o, dict) and "model_state" in o else o


def diff(a, b):
    return max(float((a[k] - b[k]).abs().max()) for k in a)


d = Path(tempfile.mkdtemp())
run = e1(out_root=d / "run", num_games=3, sims_per_move_grid=[2], seed_base=999, max_seconds=300.0)


def train(out, epochs, resume=None):
    train_main(run_dirs=[run], out_dir=out, hidden_dim=32, num_layers=2,
               epochs=epochs, batch_size=4, lr=1e-3, seed=0, resume_from=resume,
               val_frac=0.2)


# Two straight 1-epoch runs from scratch -> identical? (tests init+ep1 determinism)
a1 = d / "a1"; train(a1, 1)
b1 = d / "b1"; train(b1, 1)
print("two 1-epoch runs epoch1 diff:", diff(w(a1 / "checkpoint.pt"), w(b1 / "checkpoint.pt")))

# Straight 2 epochs vs (1 epoch then resume 1) -> identical?
s2 = d / "s2"; train(s2, 2)
r = d / "r"; train(r, 1); train(r, 2, resume=r / "checkpoint.pt")
print("straight-2 vs 1+resume diff:", diff(w(s2 / "checkpoint.pt"), w(r / "checkpoint.pt")))
