"""Does B>1 GNN-forward batching actually gain throughput on THIS box
(GTX 1650 4GB, h128 net)? Measure raw forward throughput at B=1/4/8/16/32 on
both CPU and CUDA, using the traced TorchScript module via... actually we time
the EAGER model here (same kernels) to answer the architecture question cheaply
without the Rust batched path. If B>1 doesn't help, batching is moot.

Reports forwards/sec and effective states/sec at each batch size.
"""
import time
import numpy as np
import torch
from torch_geometric.data import Batch

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg
from catan_mcts.adapter import CatanGame

HIDDEN, LAYERS = 128, 4


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
    pool = [state_to_pyg(o) for o in states(64)]
    print(f"\n=== device={device} ===")
    for B in (1, 4, 8, 16, 32):
        batch = Batch.from_data_list(pool[:B]).to(device)
        # warmup
        with torch.no_grad():
            for _ in range(5):
                model(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        N = 200
        t0 = time.monotonic()
        with torch.no_grad():
            for _ in range(N):
                model(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.monotonic() - t0
        fwd_per_s = N / dt
        states_per_s = fwd_per_s * B
        print(f"B={B:2d}: {fwd_per_s:8.1f} forwards/s  {states_per_s:9.1f} states/s  "
              f"({1e3*dt/N:.3f} ms/forward)")


if __name__ == "__main__":
    bench("cpu")
    if torch.cuda.is_available():
        bench("cuda")
