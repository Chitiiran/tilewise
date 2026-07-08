"""Phase-0 spike, step 3: export 50 fixed states + traced .ts for tch-rs parity.

Writes:
  spike/wrapper_traced.ts      -- the traced TensorWrapper (B=1, plain tensors)
  spike/golden.npz             -- 50 states' inputs + PyTorch reference outputs

The Rust spike loads wrapper_traced.ts, feeds each state's inputs, and must
reproduce value+logits BIT-EXACT (max abs diff = 0 on CPU per spec §5.1).

50 distinct states are gathered by playing seeded random games and snapshotting
observations at varied depths (covers chance nodes, early/mid/late game).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg
from catan_mcts.adapter import CatanGame
from torch_geometric.data import Batch

from try_trace_wrapper import TensorWrapper, load_model  # reuse

OUT_DIR = Path(__file__).resolve().parent
N_STATES = 50


def gather_states(n: int) -> list[dict]:
    """Collect n distinct observations across seeded random playouts."""
    obss = []
    seed = 0
    while len(obss) < n:
        game = CatanGame()
        st = game.new_initial_state()
        rng = random.Random(seed)
        seed += 1
        depth_target = rng.randrange(1, 60)
        steps = 0
        while not st.is_terminal() and steps < depth_target:
            la = st.legal_actions()
            if not la:
                break
            st.apply_action(la[rng.randrange(len(la))])
            steps += 1
        if st.is_terminal():
            continue
        obss.append(st._engine.observation())
    return obss


def main() -> None:
    model = load_model()
    wrapper = TensorWrapper(model).eval()

    obss = gather_states(N_STATES)
    print(f"gathered {len(obss)} states")

    # Trace once on the first state.
    o0 = obss[0]
    hx0 = torch.from_numpy(np.ascontiguousarray(o0["hex_features"], dtype=np.float32))
    vx0 = torch.from_numpy(np.ascontiguousarray(o0["vertex_features"], dtype=np.float32))
    ex0 = torch.from_numpy(np.ascontiguousarray(o0["edge_features"], dtype=np.float32))
    sc0 = torch.from_numpy(np.ascontiguousarray(o0["scalars"], dtype=np.float32))
    traced = torch.jit.trace(wrapper, (hx0, vx0, ex0, sc0), strict=True)
    traced.save(str(OUT_DIR / "wrapper_traced.ts"))

    hexs, verts, edges, scals, values, logits = [], [], [], [], [], []
    max_dv = max_dl = 0.0
    for o in obss:
        hx = torch.from_numpy(np.ascontiguousarray(o["hex_features"], dtype=np.float32))
        vx = torch.from_numpy(np.ascontiguousarray(o["vertex_features"], dtype=np.float32))
        ex = torch.from_numpy(np.ascontiguousarray(o["edge_features"], dtype=np.float32))
        sc = torch.from_numpy(np.ascontiguousarray(o["scalars"], dtype=np.float32))
        with torch.no_grad():
            ref_v, ref_l = model(Batch.from_data_list([state_to_pyg(o)]))
            tv, tl = traced(hx, vx, ex, sc)
        max_dv = max(max_dv, (tv - ref_v).abs().max().item())
        max_dl = max(max_dl, (tl - ref_l).abs().max().item())
        hexs.append(hx.numpy()); verts.append(vx.numpy()); edges.append(ex.numpy())
        scals.append(sc.numpy())
        values.append(ref_v.squeeze(0).numpy()); logits.append(ref_l.squeeze(0).numpy())

    print(f"traced-vs-eager over {len(obss)} states: max|dv|={max_dv:.3e} max|dl|={max_dl:.3e}")
    assert max_dv == 0.0 and max_dl == 0.0, "traced module not bit-exact to eager!"

    np.savez(
        OUT_DIR / "golden.npz",
        hex=np.stack(hexs), vertex=np.stack(verts), edge=np.stack(edges),
        scalars=np.stack(scals), value=np.stack(values), logits=np.stack(logits),
    )

    # Dependency-free flat binaries for the Rust spike: all float32 LE.
    # Layout is fixed (N=50, shapes constant), so Rust reads by known stride.
    def dump(name, arr):
        np.ascontiguousarray(arr, dtype=np.float32).tofile(OUT_DIR / name)
    dump("g_hex.bin", np.stack(hexs))        # [N,19,8]
    dump("g_vertex.bin", np.stack(verts))    # [N,54,13]
    dump("g_edge.bin", np.stack(edges))      # [N,72,6]
    dump("g_scalars.bin", np.stack(scals))   # [N,59]
    dump("g_value.bin", np.stack(values))    # [N,4]
    dump("g_logits.bin", np.stack(logits))   # [N,280]
    with open(OUT_DIR / "g_meta.txt", "w") as f:
        f.write(f"{len(obss)}\n")
    print(f"wrote golden.npz, flat .bin files, and wrapper_traced.ts "
          f"(N={len(obss)})")


if __name__ == "__main__":
    main()
