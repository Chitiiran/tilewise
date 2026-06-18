"""Isolate the effect of BATCH SIZE on GNN-forward throughput, in the
DETERMINISTIC mode the production engine uses. Sweeps B far past 32 to find
where states/s saturates (and where VRAM caps out on the 4GB GTX 1650).

states/s = forwards/s * B is the metric that matters (each leaf is one state).
If states/s keeps rising with B, bigger batches help; when it plateaus, the GPU
is saturated and larger B only adds latency.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import time
import torch
torch.use_deterministic_algorithms(True)
from torch_geometric.data import Batch

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg
from catan_mcts.adapter import CatanGame

HIDDEN, LAYERS = 128, 4
BATCHES = [1, 4, 8, 16, 32, 48, 64, 96, 128]


def states(n):
    import random
    out, s = [], 0
    while len(out) < n:
        g = CatanGame(); st = g.new_initial_state(); rng = random.Random(s); s += 1
        for _ in range(rng.randrange(1, 60)):
            if st.is_terminal(): break
            la = st.legal_actions(); st.apply_action(la[rng.randrange(len(la))])
        if not st.is_terminal(): out.append(st._engine.observation())
    return out


def bench(device):
    model = GnnModel(hidden_dim=HIDDEN, num_layers=LAYERS).to(device).eval()
    pool = [state_to_pyg(o) for o in states(max(BATCHES))]
    print(f"\n=== device={device}  (deterministic) ===")
    print(f"{'B':>4} {'fwd/s':>9} {'states/s':>10} {'ms/fwd':>8}")
    best = (0, 0.0)
    for B in BATCHES:
        try:
            batch = Batch.from_data_list(pool[:B]).to(device)
            with torch.no_grad():
                for _ in range(5):
                    model(batch)
            if device == "cuda":
                torch.cuda.synchronize()
            N = 100
            t0 = time.monotonic()
            with torch.no_grad():
                for _ in range(N):
                    model(batch)
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.monotonic() - t0
            sps = N / dt * B
            print(f"{B:>4} {N/dt:>9.1f} {sps:>10.1f} {1e3*dt/N:>8.3f}")
            if sps > best[1]:
                best = (B, sps)
        except RuntimeError as e:
            print(f"{B:>4}  OOM/err: {str(e)[:60]}")
            break
    print(f"peak states/s at B={best[0]}: {best[1]:.1f}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        bench("cuda")
    else:
        print("CUDA not available to Python")
