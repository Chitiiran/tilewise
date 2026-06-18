"""Train→infer bridge: trace GnnModel into a plain-tensor TorchScript module.

`torch.jit.script(GnnModel)` fails on PyG `HeteroConv`'s `*args, **kwargs`
forward signature, and `torch.jit.trace` on the raw model fails because its
input is a PyG `HeteroDataBatch` (the tracer can't infer the container type).

So we wrap GnnModel in a plain-tensor `nn.Module`: Catan's graph is fixed-
topology (19 hex / 54 vert / 72 edge with constant edge_index), so the wrapper
rebuilds the single-graph HeteroData internally from buffered edge_index
tensors and `torch.jit.trace` unrolls the PyG dict-iteration into a static op
graph. Proven BIT-EXACT and generalizing to unseen states in the Phase-0 spike
(see project_phase0_torchscript_spike_2026_06_17).

The Rust `TorchScriptEvaluator` (catan_mcts_rs) loads the saved `.ts` via
tch-rs and runs the identical graph.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import _H2V_EI, _V2H_EI, _V2E_EI, _E2V_EI
from catan_mcts.adapter import CatanGame


class TensorWrapper(nn.Module):
    """Plain-tensor seam over GnnModel for a SINGLE graph (B=1).

    forward(hex_x[19,8], vertex_x[54,13], edge_x[72,6], scalars[59])
        -> (value[1,4], policy_logits[1,280])

    Fixed edge_index tensors are registered as buffers so they travel with the
    traced module and trace records them as constants.
    """

    def __init__(self, model: GnnModel) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("h2v", _H2V_EI.clone())
        self.register_buffer("v2h", _V2H_EI.clone())
        self.register_buffer("v2e", _V2E_EI.clone())
        self.register_buffer("e2v", _E2V_EI.clone())

    def forward(self, hex_x, vertex_x, edge_x, scalars):
        data = HeteroData()
        data["hex"].x = hex_x
        data["vertex"].x = vertex_x
        data["edge"].x = edge_x
        data["hex", "to", "vertex"].edge_index = self.h2v
        data["vertex", "to", "hex"].edge_index = self.v2h
        data["vertex", "to", "edge"].edge_index = self.v2e
        data["edge", "to", "vertex"].edge_index = self.e2v
        data.scalars = scalars.view(1, -1)
        # B=1 batch vector: every node maps to graph 0.
        data["hex"].batch = torch.zeros(hex_x.size(0), dtype=torch.long)
        data["vertex"].batch = torch.zeros(vertex_x.size(0), dtype=torch.long)
        data["edge"].batch = torch.zeros(edge_x.size(0), dtype=torch.long)
        return self.model(data)  # (value, logits)


def _example_inputs():
    """One real initial-state observation as plain tensors, for tracing."""
    st = CatanGame().new_initial_state()
    o = st._engine.observation()

    def f(k):
        return torch.from_numpy(np.ascontiguousarray(o[k], dtype=np.float32))

    return (f("hex_features"), f("vertex_features"),
            f("edge_features"), f("scalars"))


def _load_gnn(checkpoint: Path, hidden_dim: int, num_layers: int) -> GnnModel:
    model = GnnModel(hidden_dim=hidden_dim, num_layers=num_layers)
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)
    return model.eval()


NH, NV, NE = 19, 54, 72  # fixed Catan topology (hex / vertex / edge counts)


def _offset_edge_index(ei, src_n, dst_n, b_max):
    cols = []
    for g in range(b_max):
        e = ei.clone()
        e[0] += g * src_n
        e[1] += g * dst_n
        cols.append(e)
    return torch.cat(cols, dim=1)


class BatchTensorWrapper(nn.Module):
    """Batched plain-tensor seam for a FIXED B_MAX:
        forward(hex_x[B_MAX*19,8], vertex_x[B_MAX*54,13],
                edge_x[B_MAX*72,6], scalars[B_MAX,59]) -> (value[B_MAX,4], logits[B_MAX,280])

    The offset edge_index (per-graph base + g*N) and per-type batch vectors are
    BAKED as buffers at the fixed B_MAX, so Rust only supplies the four feature
    tensors. Traced at B_MAX (PyG pooling freezes dim_size=batch.max()+1 as a
    constant, so B_MAX is fixed); callers pad shorter batches to B_MAX with
    copies of slot 0 and take the first-k outputs.
    """

    def __init__(self, model: GnnModel, b_max: int) -> None:
        super().__init__()
        self.model = model
        self.b_max = b_max
        self.register_buffer("h2v", _offset_edge_index(_H2V_EI, NH, NV, b_max))
        self.register_buffer("v2h", _offset_edge_index(_V2H_EI, NV, NH, b_max))
        self.register_buffer("v2e", _offset_edge_index(_V2E_EI, NV, NE, b_max))
        self.register_buffer("e2v", _offset_edge_index(_E2V_EI, NE, NV, b_max))
        self.register_buffer("hb", torch.repeat_interleave(torch.arange(b_max), NH))
        self.register_buffer("vb", torch.repeat_interleave(torch.arange(b_max), NV))
        self.register_buffer("eb", torch.repeat_interleave(torch.arange(b_max), NE))

    def forward(self, hex_x, vertex_x, edge_x, scalars):
        data = HeteroData()
        data["hex"].x = hex_x
        data["vertex"].x = vertex_x
        data["edge"].x = edge_x
        data["hex", "to", "vertex"].edge_index = self.h2v
        data["vertex", "to", "hex"].edge_index = self.v2h
        data["vertex", "to", "edge"].edge_index = self.v2e
        data["edge", "to", "vertex"].edge_index = self.e2v
        data["hex"].batch = self.hb
        data["vertex"].batch = self.vb
        data["edge"].batch = self.eb
        data.scalars = scalars
        return self.model(data)  # (value, logits)


def _batched_example(b_max: int):
    """B_MAX initial-state graphs: just the four concatenated feature tensors."""
    obss = [CatanGame().new_initial_state()._engine.observation() for _ in range(b_max)]

    def stack(key):
        return torch.cat([
            torch.from_numpy(np.ascontiguousarray(o[key], dtype=np.float32))
            for o in obss])

    sc = torch.stack([
        torch.from_numpy(np.ascontiguousarray(o["scalars"], dtype=np.float32))
        for o in obss])
    return (stack("hex_features"), stack("vertex_features"),
            stack("edge_features"), sc)


def export_batched(*, checkpoint: Path, out_ts: Path, hidden_dim: int,
                   num_layers: int, b_max: int = 32) -> Path:
    """Trace the BatchTensorWrapper at a fixed B_MAX into a TorchScript module.

    Deterministic algorithms are enabled so the (CUDA) batched forward is
    REPRODUCIBLE — same inputs -> same outputs every run (the replayability
    contract; see project_task10_batching_decision_2026_06_18). The Rust loader
    must likewise run with deterministic kernels.
    """
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    model = _load_gnn(checkpoint, hidden_dim, num_layers)
    wrapper = BatchTensorWrapper(model, b_max).eval()
    traced = torch.jit.trace(wrapper, _batched_example(b_max), strict=True)
    out_ts = Path(out_ts)
    out_ts.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_ts))
    return out_ts


def export(*, checkpoint: Path, out_ts: Path, hidden_dim: int,
           num_layers: int) -> Path:
    """Trace `checkpoint`'s GnnModel into a TorchScript module at `out_ts`.

    Returns the output path. The traced module's forward takes the four plain
    observation tensors and returns (value, logits) — bit-identical to the
    eager PyG model on the same device.
    """
    model = _load_gnn(checkpoint, hidden_dim, num_layers)
    wrapper = TensorWrapper(model).eval()
    traced = torch.jit.trace(wrapper, _example_inputs(), strict=True)
    out_ts = Path(out_ts)
    out_ts.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_ts))
    return out_ts
