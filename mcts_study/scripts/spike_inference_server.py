"""B1 inference-server spike (spec §3 step 2) — MEASUREMENT ONLY.

Question: does a central GPU inference server + N CPU client processes beat
the current architecture (N independent procs, each with its own CUDA
context)? Go/no-go: projected aggregate self-play throughput >= 2x the
5-proc baseline measured 2026-06-11 (~150 games/h).

Design mirrors BatchedGnnEvaluator's flush logic, but across PROCESSES:
  server proc:  owns the model on GPU; drains request queue; batches up to
                MAX_BATCH or WINDOW_MS; one forward; routes replies.
  client procs: tight loop — build a real observation (state_to_pyg on a
                fresh engine state), send, wait for reply. Measures
                round-trip p50/p95 and aggregate evals/s.

We measure EVALS/S, not games/s: self-play cost ~= evals * (tree overhead),
so the ratio of aggregate evals/s (server@N=10 vs baseline in-proc) is the
projected speedup, with tree overhead identical in both worlds.

Run (WSL venv, GPU idle — do NOT run during data-gen):
    python scripts/spike_inference_server.py --clients 10 --seconds 30
    python scripts/spike_inference_server.py --baseline --seconds 30
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import time


MAX_BATCH = 64
WINDOW_MS = 5.0


def _build_request():
    """One real observation, encoded exactly as self-play does."""
    from catan_gnn.state_to_pyg import state_to_pyg
    from catan_mcts.adapter import CatanGame
    game = CatanGame(vp_target=10, bonuses=True)
    state = game.new_initial_state(seed=123)
    return state_to_pyg(state._engine.observation())


def server_proc(req_q: mp.Queue, reply_qs: dict, ckpt: str, stop_ev) -> None:
    import torch
    from torch_geometric.data import Batch
    from catan_gnn.gnn_model import GnnModel

    model = GnnModel(hidden_dim=128, num_layers=4)
    obj = torch.load(ckpt, map_location="cuda", weights_only=False)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)
    model = model.to("cuda").eval()

    pending: list[tuple[int, object]] = []
    first_ts = None
    n_batches = 0
    batch_sizes = []
    while not stop_ev.is_set():
        timeout = WINDOW_MS / 1000.0
        try:
            item = req_q.get(timeout=timeout)
            if first_ts is None:
                first_ts = time.perf_counter()
            pending.append(item)
        except Exception:
            pass
        window_up = (first_ts is not None
                     and (time.perf_counter() - first_ts) * 1000 >= WINDOW_MS)
        if pending and (len(pending) >= MAX_BATCH or window_up):
            ids = [cid for cid, _ in pending]
            datas = [d for _, d in pending]
            batch = Batch.from_data_list(datas).to("cuda")
            with torch.no_grad():
                values, logits = model(batch)
            values = values.cpu().numpy()
            logits = logits.cpu().numpy()
            for i, cid in enumerate(ids):
                reply_qs[cid].put((values[i], logits[i]))
            n_batches += 1
            batch_sizes.append(len(pending))
            pending.clear()
            first_ts = None
    if batch_sizes:
        print(f"[server] batches={n_batches} mean_batch="
              f"{statistics.mean(batch_sizes):.1f}")


def client_proc(cid: int, req_q: mp.Queue, reply_q: mp.Queue,
                seconds: float, out_q: mp.Queue) -> None:
    data = _build_request()
    lat = []
    t_end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        req_q.put((cid, data))
        reply_q.get()
        lat.append(time.perf_counter() - t0)
        n += 1
    out_q.put((cid, n, statistics.median(lat),
               statistics.quantiles(lat, n=20)[18] if len(lat) >= 20 else max(lat)))


def run_server_mode(ckpt: str, n_clients: int, seconds: float) -> None:
    mp.set_start_method("spawn", force=True)
    req_q: mp.Queue = mp.Queue()
    reply_qs = {i: mp.Queue() for i in range(n_clients)}
    out_q: mp.Queue = mp.Queue()
    stop_ev = mp.Event()
    srv = mp.Process(target=server_proc, args=(req_q, reply_qs, ckpt, stop_ev))
    srv.start()
    time.sleep(15)   # model load + CUDA warmup
    clients = [mp.Process(target=client_proc,
                          args=(i, req_q, reply_qs[i], seconds, out_q))
               for i in range(n_clients)]
    t0 = time.perf_counter()
    for c in clients:
        c.start()
    for c in clients:
        c.join()
    wall = time.perf_counter() - t0
    stop_ev.set()
    srv.join(timeout=10)
    if srv.is_alive():
        srv.terminate()
    total = 0
    p50s, p95s = [], []
    while not out_q.empty():
        cid, n, p50, p95 = out_q.get()
        total += n
        p50s.append(p50)
        p95s.append(p95)
    print(f"[B1 spike] clients={n_clients} wall={wall:.1f}s "
          f"aggregate={total / wall:.0f} evals/s "
          f"p50={statistics.mean(p50s) * 1000:.1f}ms "
          f"p95={statistics.mean(p95s) * 1000:.1f}ms")


def run_baseline_mode(ckpt: str, seconds: float) -> None:
    """In-process batch=1 GPU evals — the per-proc cost in today's design."""
    import torch
    from torch_geometric.data import Batch
    from catan_gnn.gnn_model import GnnModel
    model = GnnModel(hidden_dim=128, num_layers=4)
    obj = torch.load(ckpt, map_location="cuda", weights_only=False)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)
    model = model.to("cuda").eval()
    data = _build_request()
    n = 0
    t_end = time.perf_counter() + seconds
    t0 = time.perf_counter()
    while time.perf_counter() < t_end:
        batch = Batch.from_data_list([data]).to("cuda")
        with torch.no_grad():
            model(batch)
        n += 1
    wall = time.perf_counter() - t0
    print(f"[baseline] in-proc batch=1: {n / wall:.0f} evals/s "
          f"({wall / n * 1000:.1f} ms/eval). Multiply by live proc count for "
          f"today's aggregate ceiling (5 procs measured ~150 games/h).")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/v3/rl_checkpoints/round0_Cell6.pt")
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--baseline", action="store_true")
    args = p.parse_args()
    if args.baseline:
        run_baseline_mode(args.checkpoint, args.seconds)
    else:
        run_server_mode(args.checkpoint, args.clients, args.seconds)


if __name__ == "__main__":
    main()
