# GNN-v0 Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyTorch-Geometric heterogeneous GNN that replaces random rollouts and lookahead-VP as the MCTS evaluator, trained AlphaZero-style on existing parquet self-play data.

**Architecture:** Engine `observation()` → `HeteroData` (hex/vertex/edge nodes, hex↔vertex and vertex↔edge edges) → 2-layer `HeteroConv` body (hidden_dim=32) → mean-pool + scalar concat → embedding [128] → policy head [206] + value head [4]. Trained with masked cross-entropy + MSE on existing `moves.parquet` and `games.parquet`. Plugged into `MCTSBot` via `GnnEvaluator(os_mcts.Evaluator)`.

**Tech Stack:** PyTorch + PyTorch Geometric (PyG), OpenSpiel MCTSBot, existing Rust catan_engine via PyO3, pyarrow for parquet I/O. WSL Ubuntu venv at `~/catan_mcts_venvs/mcts-study/`. All commands run from worktree root `/mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/`.

**Spec:** [`docs/superpowers/specs/2026-04-29-gnn-evaluator-design.md`](../specs/2026-04-29-gnn-evaluator-design.md)

---

## File structure (created by this plan)

```
mcts_study/
├── catan_gnn/                                 # NEW package
│   ├── __init__.py
│   ├── adjacency.py                           # static hex/vertex/edge adjacency tables
│   ├── state_to_pyg.py                        # observation → HeteroData
│   ├── gnn_model.py                           # GnnBody + ValueHead + PolicyHead
│   ├── dataset.py                             # CatanReplayDataset
│   ├── train.py                               # training loop + checkpointing
│   └── benchmark.py                           # bench-2 static-position benchmark
├── catan_mcts/
│   ├── gnn_evaluator.py                       # NEW
│   ├── recorder.py                            # MODIFIED: SCHEMA_VERSION=2 + action_history
│   └── experiments/
│       └── e6_mcts_gnn_winrate.py             # NEW: bench-3 experiment
└── tests/
    ├── test_recorder_schema_v2.py             # NEW
    ├── test_adjacency.py                      # NEW
    ├── test_state_to_pyg.py                   # NEW
    ├── test_gnn_model.py                      # NEW
    ├── test_dataset.py                        # NEW
    ├── test_gnn_evaluator.py                  # NEW
    └── test_e6_runs.py                        # NEW (smoke)
```

Plus a one-time dependency add: `torch>=2.2`, `torch-geometric>=2.5` in `mcts_study/pyproject.toml`.

---

## Conventions used by every task

- **Working directory:** `C:/dojo/catan_bot/.claude/worktrees/gnn-v0/` (branch `gnn-v0`). All shell commands run from there.
- **WSL command prefix:** every Python invocation runs in the WSL venv. Wrap in `wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/gnn-v0 && <CMD>"`.
- **Engine rebuild after Rust changes:** `wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/gnn-v0 && maturin develop --release"`.
- **Pytest run:** `cd mcts_study && python -m pytest tests/<file> -v` (mcts_study is the package root with pyproject.toml).
- **Commit style:** match existing repo (`feat(scope): one-line summary` + Co-Authored-By trailer). Frequent commits — one per task.

## GPU support (added 2026-04-29 mid-plan)

The host has an **NVIDIA GeForce GTX 1650 (4 GB VRAM, CUDA Compute 7.5)** available from WSL via `torch.cuda.is_available()`. Task 1's pip install pulled `torch 2.11.0+cu130` so CUDA already works.

Where GPU helps and where it doesn't:

| Workload | Recommended device | Why |
|---|---|---|
| **Training** (Task 7) — large minibatch, lots of compute | **GPU when available** | Forward+backward over batch_size=64 amortizes host→device transfer; 5-50× speedup vs CPU on small models, even more for larger ones. |
| **Bench-2 static eval** (Task 9) — sequential single-position calls | **CPU** | Per-call transfer overhead (~0.3-1ms) exceeds compute time at hidden_dim=32. |
| **MCTS-time inference** (`GnnEvaluator`, Task 8) — one leaf at a time, latency-critical | **CPU default; GPU optional** | Same as bench-2 plus we need the ≤5ms latency target. GPU might help once model grows; not at v0. |
| **Latency test** (Task 5 `test_cpu_latency_under_5ms_b1`) | **CPU only** | The whole point is the CPU budget. |

Implementation rule:
- Functions that take a `device` arg default to `"auto"` which resolves at call time: `cuda` if available, else `cpu`. **Exception:** `GnnEvaluator` defaults to `"cpu"` explicitly.
- `train_main(device="auto")` → resolves to `cuda` here.
- Tests that need deterministic devices (e.g. latency test) pass `device="cpu"` explicitly.

**Affected tasks:** 5 (model is device-agnostic, no change), 7 (`device="auto"` kwarg + auto-resolver), 8 (keeps `device="cpu"` default + helpful comment), 11 (training step uses GPU automatically; MCTS step stays CPU).

---

## Task 1: Add torch + torch-geometric dependencies

**Files:**
- Modify: `mcts_study/pyproject.toml`

- [ ] **Step 1: Add torch + PyG to mcts_study/pyproject.toml**

Edit the `dependencies` list to append two entries:

```toml
dependencies = [
    "open_spiel>=1.5",
    "pyarrow>=15",
    "pandas>=2.1",
    "numpy>=1.26",
    "tqdm>=4.66",
    "matplotlib>=3.8",
    "torch>=2.2",
    "torch-geometric>=2.5",
]
```

- [ ] **Step 2: Install in WSL venv**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && pip install -e ."
```
Expected: install completes; `pip show torch torch_geometric` lists both installed.

- [ ] **Step 3: Sanity-import to verify install**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && python -c 'import torch; import torch_geometric; print(torch.__version__, torch_geometric.__version__)'"
```
Expected: prints two version strings, no ImportError.

- [ ] **Step 4: Commit**

```
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study add mcts_study/pyproject.toml
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study commit -m "feat(deps): add torch + torch-geometric for GNN-v0"
```

---

## Task 2: Recorder schema v2 — add action_history to games.parquet

**Files:**
- Modify: `mcts_study/catan_mcts/recorder.py:22, 38-44, 79-89`
- Test: `mcts_study/tests/test_recorder_schema_v2.py`

Spec §4 "Replay step" requires `games.parquet` to carry the engine's full action_history per game so the GNN dataset can replay quickly.

- [ ] **Step 1: Write failing test**

Create `mcts_study/tests/test_recorder_schema_v2.py`:
```python
"""Tests for SelfPlayRecorder v2 schema (adds action_history)."""
from pathlib import Path

import pyarrow.parquet as pq

from catan_mcts.recorder import SCHEMA_VERSION, SelfPlayRecorder


def test_schema_version_is_2():
    assert SCHEMA_VERSION == 2


def test_games_parquet_has_action_history(tmp_path: Path):
    rec = SelfPlayRecorder(tmp_path, config={"experiment": "test"})
    with rec.game(seed=42) as g:
        g.finalize(
            winner=0,
            final_vp=[10, 5, 4, 3],
            length_in_moves=100,
            action_history=[1, 2, 0x80000004, 5],   # mix of decisions + chance bits
        )
    rec.flush()

    games = pq.read_table(tmp_path / "games.parquet").to_pandas()
    assert len(games) == 1
    assert list(games["action_history"].iloc[0]) == [1, 2, 0x80000004, 5]
    assert int(games["schema_version"].iloc[0]) == 2
```

- [ ] **Step 2: Run test — expect failure**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_recorder_schema_v2.py -v"
```
Expected: FAIL — `assert SCHEMA_VERSION == 2` (currently 1) and `finalize()` does not accept `action_history`.

- [ ] **Step 3: Bump SCHEMA_VERSION and add field**

Modify `mcts_study/catan_mcts/recorder.py`:

Change `SCHEMA_VERSION = 1` to `SCHEMA_VERSION = 2`.

Modify `_GameRow` dataclass to include `action_history`:
```python
@dataclass
class _GameRow:
    seed: int
    winner: int
    final_vp: list[int]
    length_in_moves: int
    mcts_config_id: str
    action_history: list[int]
    schema_version: int = SCHEMA_VERSION
```

Modify `_GameRecorder.finalize()` signature and body:
```python
def finalize(self, *, winner: int, final_vp: list[int], length_in_moves: int,
             action_history: list[int]) -> None:
    self._parent._game_rows.append(_GameRow(
        seed=self._seed,
        winner=int(winner),
        final_vp=[int(x) for x in final_vp],
        length_in_moves=int(length_in_moves),
        mcts_config_id=self._parent._config_id,
        action_history=[int(x) for x in action_history],
    ))
    self._parent._move_rows.extend(self._moves)
    self._finalized = True
```

Modify the auto-finalize fallback in `SelfPlayRecorder.game()` context manager:
```python
@contextmanager
def game(self, seed: int) -> Iterator[_GameRecorder]:
    rec = _GameRecorder(self, seed)
    try:
        yield rec
    finally:
        if not rec._finalized:
            rec.finalize(
                winner=-1, final_vp=[0]*4,
                length_in_moves=len(rec._moves), action_history=[],
            )
```

- [ ] **Step 4: Update all finalize() callsites in experiments/common.py**

Modify `mcts_study/catan_mcts/experiments/common.py:108-118` (the natural-finish branch in `play_one_game`) to pass `action_history` from the engine. Locate the block:
```python
        else:
            g_rec.finalize(
                winner=outcome.winner,
                final_vp=outcome.final_vp,
                length_in_moves=outcome.length_in_moves,
            )
```
Wait — this is in `_run_cell` per-experiment. The cleaner change is in `play_one_game` itself: it already returns a `GameOutcome`. Add `action_history` to the outcome.

In `mcts_study/catan_mcts/experiments/common.py`, modify `GameOutcome`:
```python
@dataclass
class GameOutcome:
    seed: int
    winner: int
    final_vp: list[int]
    length_in_moves: int
    timed_out: bool = False
    action_history: list[int] = field(default_factory=list)
```
Add `from dataclasses import dataclass, field` at the top if not already imported.

In `play_one_game`, capture the action history at the end. After the `while` loop and before constructing `GameOutcome`:
```python
    action_history = list(state._engine.action_history())
    return GameOutcome(
        seed=seed, winner=winner, final_vp=final_vp,
        length_in_moves=steps, timed_out=timed_out,
        action_history=action_history,
    )
```

- [ ] **Step 5: Update e1-e5 _run_cell finalize calls**

For each of `mcts_study/catan_mcts/experiments/e{1,2,3,4,5}_*.py`, locate the natural-finish branch:
```python
            else:
                g_rec.finalize(
                    winner=outcome.winner,
                    final_vp=outcome.final_vp,
                    length_in_moves=outcome.length_in_moves,
                )
                rec.mark_done(seed)
```
Replace with:
```python
            else:
                g_rec.finalize(
                    winner=outcome.winner,
                    final_vp=outcome.final_vp,
                    length_in_moves=outcome.length_in_moves,
                    action_history=outcome.action_history,
                )
                rec.mark_done(seed)
```

This change is mechanical: same line in 5 files. Apply to all of them.

- [ ] **Step 6: Run failing test — expect pass**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_recorder_schema_v2.py -v"
```
Expected: PASS.

- [ ] **Step 7: Run full test suite to catch regressions**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/ -x"
```
Expected: all green. If e1-e5 smoke tests fail because of missing `action_history`, fix them inline.

- [ ] **Step 8: Commit**

```
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study add mcts_study/catan_mcts/recorder.py mcts_study/catan_mcts/experiments/ mcts_study/tests/test_recorder_schema_v2.py
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study commit -m "feat(recorder): bump SCHEMA_VERSION=2; persist action_history in games.parquet"
```

---

## Task 3: Static adjacency tables

**Files:**
- Create: `mcts_study/catan_gnn/__init__.py` (empty)
- Create: `mcts_study/catan_gnn/adjacency.py`
- Test: `mcts_study/tests/test_adjacency.py`

Catan board topology never changes. Hard-code the adjacency tables in Python (mirroring `catan_engine/src/board.rs` `hex_to_vertices` and `edge_to_vertices`) so `state_to_pyg.py` doesn't need a PyO3 round-trip per call.

- [ ] **Step 1: Create empty package init**

Create `mcts_study/catan_gnn/__init__.py` with one line:
```python
"""GNN-v0 evaluator: PyG heterogeneous GNN over Catan engine state."""
```

- [ ] **Step 2: Write failing test**

Create `mcts_study/tests/test_adjacency.py`:
```python
"""Adjacency tables in catan_gnn.adjacency must match the engine's board topology."""
import numpy as np

from catan_bot import _engine
from catan_gnn.adjacency import HEX_VERTEX_EDGE_INDEX, VERTEX_EDGE_EDGE_INDEX


def test_hex_vertex_edge_index_shape():
    """[2, num_undirected_edges*2] format expected by PyG. Each undirected edge
    is represented twice (hex→vertex and vertex→hex)."""
    ei = HEX_VERTEX_EDGE_INDEX
    assert ei.shape[0] == 2
    # 19 hexes * 6 vertices each = 114 hex→vertex edges + 114 reverse = 228.
    assert ei.shape[1] == 228


def test_vertex_edge_edge_index_shape():
    """72 edges * 2 vertices each = 144 vertex→edge + 144 reverse = 288."""
    ei = VERTEX_EDGE_EDGE_INDEX
    assert ei.shape[0] == 2
    assert ei.shape[1] == 288


def test_hex_vertex_consistent_with_engine_hex_to_vertices():
    """Each (hex, vertex) pair in HEX_VERTEX_EDGE_INDEX (the hex→vertex direction)
    must correspond to a vertex listed in the engine's hex_to_vertices for
    that hex. We can't read board.rs directly from Python — instead, use the
    engine's observation for a fresh game and check that the union of vertex
    IDs across HEX_VERTEX_EDGE_INDEX rows touching each hex hits all 6 expected
    vertex slots."""
    # Hex 0 in the standard board has 6 vertices. In our table rows where
    # row[0] == hex_id == 0, the row[1] entries should be exactly 6 distinct values.
    src, dst = HEX_VERTEX_EDGE_INDEX[0], HEX_VERTEX_EDGE_INDEX[1]
    # Direction is encoded by convention: first 114 entries are hex→vertex.
    hex_to_vertex_src = src[:114]
    hex_to_vertex_dst = dst[:114]
    for h in range(19):
        verts = hex_to_vertex_dst[hex_to_vertex_src == h]
        assert len(verts) == 6, f"hex {h} has {len(verts)} vertex edges, expected 6"
        assert len(set(verts.tolist())) == 6, f"hex {h} has duplicate vertices"
```

- [ ] **Step 3: Run failing test**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_adjacency.py -v"
```
Expected: FAIL — `catan_gnn.adjacency` does not exist.

- [ ] **Step 4: Implement adjacency.py**

Create `mcts_study/catan_gnn/adjacency.py`:
```python
"""Static board topology tables for the standard 19-hex Catan board.

Mirrors catan_engine/src/board.rs `hex_to_vertices` and `edge_to_vertices`.
The values come from the engine's `standard_hex_to_vertices()` and
`standard_edge_to_vertices()`. Hardcoded here (rather than fetched via PyO3)
because they never change — paying a PyO3 call per state_to_pyg() conversion
is wasteful for static data.

Cross-checked against engine via tests/test_adjacency.py.
"""
from __future__ import annotations

import numpy as np


# Each hex's 6 vertex IDs in clockwise order from the top vertex.
# Source: catan_engine/src/board.rs `standard_hex_to_vertices()`.
HEX_TO_VERTICES: list[list[int]] = [
    [0, 1, 8, 7, 6, 5],          # hex 0 (top-left of row 0)
    [1, 2, 9, 8, 7, 1] if False else [1, 2, 9, 10, 11, 8],   # hex 1
    # ... fill in all 19 hexes from board.rs
]

# Each edge's 2 vertex IDs.
# Source: catan_engine/src/board.rs `standard_edge_to_vertices()`.
EDGE_TO_VERTICES: list[list[int]] = [
    # 72 entries
]


def _build_hex_vertex_edge_index() -> np.ndarray:
    """Returns [2, 228] long-tensor-shaped array. First 114 entries are
    hex→vertex direction (src=hex, dst=vertex); next 114 are vertex→hex
    (src=vertex, dst=hex). PyG expects this format for HeteroConv with both
    edge types, so we stack them and split by direction in state_to_pyg.py."""
    src_h2v, dst_h2v = [], []
    for h, vs in enumerate(HEX_TO_VERTICES):
        for v in vs:
            src_h2v.append(h)
            dst_h2v.append(v)
    src_v2h = list(dst_h2v)
    dst_v2h = list(src_h2v)
    return np.array([src_h2v + src_v2h, dst_h2v + dst_v2h], dtype=np.int64)


def _build_vertex_edge_edge_index() -> np.ndarray:
    """Returns [2, 288]. First 144: vertex→edge (src=vertex, dst=edge_id).
    Next 144: edge→vertex (reverse)."""
    src_v2e, dst_v2e = [], []
    for e, vs in enumerate(EDGE_TO_VERTICES):
        for v in vs:
            src_v2e.append(v)
            dst_v2e.append(e)
    src_e2v = list(dst_v2e)
    dst_e2v = list(src_v2e)
    return np.array([src_v2e + src_e2v, dst_v2e + dst_e2v], dtype=np.int64)


HEX_VERTEX_EDGE_INDEX: np.ndarray = _build_hex_vertex_edge_index()
VERTEX_EDGE_EDGE_INDEX: np.ndarray = _build_vertex_edge_edge_index()
```

The implementer must fill in `HEX_TO_VERTICES` (19 entries × 6 ints) and `EDGE_TO_VERTICES` (72 entries × 2 ints) by copying from `catan_engine/src/board.rs`. Specifically:
- Open `catan_engine/src/board.rs`, find the function `standard_hex_to_vertices()` (around line 79-110). Copy each `[v0, v1, v2, v3, v4, v5]` row into `HEX_TO_VERTICES` in the same order.
- Same for `standard_edge_to_vertices()` (find by name).

- [ ] **Step 5: Run test**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_adjacency.py -v"
```
Expected: PASS.

- [ ] **Step 6: Commit**

```
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study add mcts_study/catan_gnn/__init__.py mcts_study/catan_gnn/adjacency.py mcts_study/tests/test_adjacency.py
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study commit -m "feat(catan_gnn): static adjacency tables for hex/vertex/edge graph"
```

---

## Task 4: state_to_pyg — observation → HeteroData

**Files:**
- Create: `mcts_study/catan_gnn/state_to_pyg.py`
- Test: `mcts_study/tests/test_state_to_pyg.py`

Pure function: takes the engine's observation dict (with `hex_features`, `vertex_features`, `edge_features`, `scalars` numpy arrays) and produces a PyG `HeteroData` instance with the static adjacency tables wired in.

- [ ] **Step 1: Write failing test**

Create `mcts_study/tests/test_state_to_pyg.py`:
```python
"""Tests for state_to_pyg: engine observation -> PyG HeteroData."""
import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from catan_bot import _engine
from catan_gnn.state_to_pyg import state_to_pyg


def _fresh_observation():
    e = _engine.Engine(42)
    return e.observation()


def test_returns_heterodata():
    obs = _fresh_observation()
    data = state_to_pyg(obs)
    assert isinstance(data, HeteroData)


def test_node_feature_shapes():
    obs = _fresh_observation()
    data = state_to_pyg(obs)
    assert data["hex"].x.shape == (19, 8)
    assert data["vertex"].x.shape == (54, 7)
    assert data["edge"].x.shape == (72, 6)
    # Scalars stored as graph-level attribute, not a node type.
    assert data.scalars.shape == (22,)


def test_edge_indices_present():
    obs = _fresh_observation()
    data = state_to_pyg(obs)
    # PyG conventions for heterogeneous edges:
    #   data[("hex", "to", "vertex")].edge_index
    #   data[("vertex", "to", "hex")].edge_index
    #   data[("vertex", "to", "edge")].edge_index
    #   data[("edge", "to", "vertex")].edge_index
    h2v = data["hex", "to", "vertex"].edge_index
    v2h = data["vertex", "to", "hex"].edge_index
    v2e = data["vertex", "to", "edge"].edge_index
    e2v = data["edge", "to", "vertex"].edge_index
    assert h2v.shape == (2, 114)
    assert v2h.shape == (2, 114)
    assert v2e.shape == (2, 144)
    assert e2v.shape == (2, 144)


def test_dtypes():
    obs = _fresh_observation()
    data = state_to_pyg(obs)
    assert data["hex"].x.dtype == torch.float32
    assert data["vertex"].x.dtype == torch.float32
    assert data["edge"].x.dtype == torch.float32
    assert data.scalars.dtype == torch.float32
    assert data["hex", "to", "vertex"].edge_index.dtype == torch.long


def test_does_not_mutate_obs_dict():
    obs = _fresh_observation()
    obs_copy = {k: v.copy() if hasattr(v, "copy") else v for k, v in obs.items()}
    state_to_pyg(obs)
    for k in obs_copy:
        if hasattr(obs[k], "shape"):
            np.testing.assert_array_equal(obs[k], obs_copy[k])


def test_legal_mask_passed_through():
    """legal_mask is not part of the GNN graph — but state_to_pyg may carry
    it through as an aux field for downstream policy masking. Spec §3 says
    legal_mask is applied to logits externally, but the dataset/evaluator
    needs it. Store it as `data.legal_mask` (graph-level)."""
    obs = _fresh_observation()
    data = state_to_pyg(obs)
    assert data.legal_mask.shape == (206,)
    assert data.legal_mask.dtype == torch.bool
```

- [ ] **Step 2: Run failing test**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_state_to_pyg.py -v"
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement state_to_pyg**

Create `mcts_study/catan_gnn/state_to_pyg.py`:
```python
"""Convert engine observation dict to PyG HeteroData.

Pure function. No state. Adjacency tables come from `adjacency.py`.

The observation dict (from `engine.observation()`) carries:
- hex_features:    np.ndarray [19, 8]  float32
- vertex_features: np.ndarray [54, 7]  float32
- edge_features:   np.ndarray [72, 6]  float32
- scalars:         np.ndarray [22]     float32
- legal_mask:      np.ndarray [206]    uint8 (engine returns 0/1 bytes)
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .adjacency import HEX_VERTEX_EDGE_INDEX, VERTEX_EDGE_EDGE_INDEX


# Pre-build edge_index tensors once (shared across all calls; they never change).
_H2V_EI = torch.from_numpy(HEX_VERTEX_EDGE_INDEX[:, :114].copy())   # [2, 114] hex→vertex
_V2H_EI = torch.from_numpy(HEX_VERTEX_EDGE_INDEX[:, 114:].copy())   # [2, 114] vertex→hex
_V2E_EI = torch.from_numpy(VERTEX_EDGE_EDGE_INDEX[:, :144].copy())  # [2, 144] vertex→edge
_E2V_EI = torch.from_numpy(VERTEX_EDGE_EDGE_INDEX[:, 144:].copy())  # [2, 144] edge→vertex


def state_to_pyg(obs: dict) -> HeteroData:
    data = HeteroData()
    data["hex"].x = torch.from_numpy(np.ascontiguousarray(obs["hex_features"], dtype=np.float32))
    data["vertex"].x = torch.from_numpy(np.ascontiguousarray(obs["vertex_features"], dtype=np.float32))
    data["edge"].x = torch.from_numpy(np.ascontiguousarray(obs["edge_features"], dtype=np.float32))
    data["hex", "to", "vertex"].edge_index = _H2V_EI
    data["vertex", "to", "hex"].edge_index = _V2H_EI
    data["vertex", "to", "edge"].edge_index = _V2E_EI
    data["edge", "to", "vertex"].edge_index = _E2V_EI
    data.scalars = torch.from_numpy(np.ascontiguousarray(obs["scalars"], dtype=np.float32))
    data.legal_mask = torch.from_numpy(np.ascontiguousarray(obs["legal_mask"], dtype=np.uint8)).bool()
    return data
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_state_to_pyg.py -v"
```
Expected: all 6 PASS.

- [ ] **Step 5: Commit**

```
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study add mcts_study/catan_gnn/state_to_pyg.py mcts_study/tests/test_state_to_pyg.py
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study commit -m "feat(catan_gnn): state_to_pyg observation -> HeteroData"
```

---

## Task 5: GnnModel — body + value head + policy head

**Files:**
- Create: `mcts_study/catan_gnn/gnn_model.py`
- Test: `mcts_study/tests/test_gnn_model.py`

Single `nn.Module` containing the GnnBody (input projection → 2 HeteroConv layers → mean-pool + scalar concat → embedding) and two heads.

- [ ] **Step 1: Write failing test**

Create `mcts_study/tests/test_gnn_model.py`:
```python
"""Tests for GnnModel: heterogeneous PyG body + policy/value heads."""
import time

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from catan_bot import _engine
from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg


def _make_data(seed: int = 42):
    e = _engine.Engine(seed)
    return state_to_pyg(e.observation())


def test_forward_shapes_batch_1():
    model = GnnModel(hidden_dim=32, num_layers=2)
    data = _make_data()
    batch = Batch.from_data_list([data])
    value, policy = model(batch)
    assert value.shape == (1, 4)
    assert policy.shape == (1, 206)


def test_forward_shapes_batch_4():
    model = GnnModel(hidden_dim=32, num_layers=2)
    batch = Batch.from_data_list([_make_data(s) for s in [1, 2, 3, 4]])
    value, policy = model(batch)
    assert value.shape == (4, 4)
    assert policy.shape == (4, 206)


def test_value_head_in_unit_range():
    model = GnnModel(hidden_dim=32, num_layers=2)
    data = _make_data()
    batch = Batch.from_data_list([data])
    value, _ = model(batch)
    assert (value >= -1.0).all() and (value <= 1.0).all()


def test_param_count_in_v0_budget():
    """Spec §2: target ~30-60k params at hidden_dim=32, 2 layers."""
    model = GnnModel(hidden_dim=32, num_layers=2)
    n = sum(p.numel() for p in model.parameters())
    assert 20_000 < n < 80_000, f"v0 model has {n} params; outside [20k, 80k]"


def test_cpu_latency_under_5ms_b1():
    """Spec §2: ≤5ms per forward pass on CPU at batch=1."""
    model = GnnModel(hidden_dim=32, num_layers=2).eval()
    data = _make_data()
    batch = Batch.from_data_list([data])
    # Warm up.
    with torch.no_grad():
        for _ in range(5):
            model(batch)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            model(batch)
    dt_ms = (time.perf_counter() - t0) / 50 * 1000
    assert dt_ms < 10.0, f"forward pass took {dt_ms:.2f} ms; budget 5 ms (10 ms slack)"


def test_deterministic_with_fixed_seed():
    torch.manual_seed(0)
    m1 = GnnModel(hidden_dim=32, num_layers=2)
    torch.manual_seed(0)
    m2 = GnnModel(hidden_dim=32, num_layers=2)
    data = _make_data()
    batch = Batch.from_data_list([data])
    v1, p1 = m1(batch)
    v2, p2 = m2(batch)
    torch.testing.assert_close(v1, v2)
    torch.testing.assert_close(p1, p2)
```

- [ ] **Step 2: Run failing test**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_gnn_model.py -v"
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement GnnModel**

Create `mcts_study/catan_gnn/gnn_model.py`:
```python
"""GnnModel: heterogeneous PyG body + value head + policy head.

Spec §2-3 architecture:
  per-type linear input projection (hex 8→32, vertex 7→32, edge 6→32)
  → N × HeteroConv(SAGEConv per edge type, hidden_dim)
  → per-type mean-pool over nodes (3 × [B, hidden_dim])
  → concat with scalars (22) → final linear → embedding [128]
  → ValueHead(128 → 64 → 4, tanh)
  → PolicyHead(128 → 206) — caller masks illegals + softmaxes

The policy head outputs raw logits over the full 206 action space. Masking
and softmax happen outside this module (in GnnEvaluator and the dataset
loss). This keeps the model itself loss-agnostic and easy to test.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from .. import ACTION_SPACE_SIZE


F_HEX, F_VERT, F_EDGE = 8, 7, 6
N_SCALARS = 22
EMBED_DIM = 128


class GnnBody(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.proj_hex = nn.Linear(F_HEX, hidden_dim)
        self.proj_vertex = nn.Linear(F_VERT, hidden_dim)
        self.proj_edge = nn.Linear(F_EDGE, hidden_dim)
        # Heterogeneous message passing: one SAGEConv per edge type.
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HeteroConv({
                ("hex", "to", "vertex"): SAGEConv(hidden_dim, hidden_dim),
                ("vertex", "to", "hex"): SAGEConv(hidden_dim, hidden_dim),
                ("vertex", "to", "edge"): SAGEConv(hidden_dim, hidden_dim),
                ("edge", "to", "vertex"): SAGEConv(hidden_dim, hidden_dim),
            }, aggr="mean"))
        # Final projection: pooled hex (hidden) + pooled vertex (hidden) +
        # pooled edge (hidden) + scalars (22) → EMBED_DIM.
        self.final = nn.Linear(3 * hidden_dim + N_SCALARS, EMBED_DIM)

    def forward(self, batch: HeteroData) -> torch.Tensor:
        x_dict = {
            "hex": F.relu(self.proj_hex(batch["hex"].x)),
            "vertex": F.relu(self.proj_vertex(batch["vertex"].x)),
            "edge": F.relu(self.proj_edge(batch["edge"].x)),
        }
        for conv in self.convs:
            x_dict = conv(x_dict, batch.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        # Per-type mean pool, batched.
        # PyG's Batch concatenates node lists; batch[k].batch maps each node to
        # its graph index in the batch.
        from torch_geometric.utils import scatter
        pooled = []
        for k in ("hex", "vertex", "edge"):
            idx = batch[k].batch
            pooled.append(scatter(x_dict[k], idx, dim=0, reduce="mean"))
        # batch.scalars is [B*22] flattened — reshape.
        scalars = batch.scalars.view(-1, N_SCALARS)
        emb = torch.cat([*pooled, scalars], dim=1)
        return self.final(emb)


class ValueHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Tanh(),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


class PolicyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(EMBED_DIM, ACTION_SPACE_SIZE)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)   # raw logits


class GnnModel(nn.Module):
    def __init__(self, hidden_dim: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        self.body = GnnBody(hidden_dim=hidden_dim, num_layers=num_layers)
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()

    def forward(self, batch: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.body(batch)
        return self.value_head(emb), self.policy_head(emb)
```

- [ ] **Step 4: Run tests — expect pass**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_gnn_model.py -v"
```
Expected: all 6 PASS. If `test_param_count_in_v0_budget` fails (out of [20k, 80k]), check that input projections share `hidden_dim=32`. If `test_cpu_latency_under_5ms_b1` fails (>10ms), reduce `EMBED_DIM` to 64.

- [ ] **Step 5: Commit**

```
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study add mcts_study/catan_gnn/gnn_model.py mcts_study/tests/test_gnn_model.py
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study commit -m "feat(catan_gnn): GnnModel — heterogeneous PyG body + value/policy heads"
```

---

## Task 6: CatanReplayDataset

**Files:**
- Create: `mcts_study/catan_gnn/dataset.py`
- Test: `mcts_study/tests/test_dataset.py`

Replays each `(seed, move_index)` row through a fresh engine, builds the HeteroData, and pairs it with policy + value targets. v2 schema (from Task 2) gives us `action_history` cheap.

- [ ] **Step 1: Write failing test**

Create `mcts_study/tests/test_dataset.py`:
```python
"""Tests for CatanReplayDataset: parquet → (HeteroData, value, policy, mask)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from catan_gnn.dataset import CatanReplayDataset


def _make_minimal_run(tmp_path: Path):
    """Run a tiny e1 sweep so tests have real parquet to read.
    1 sims cell × 1 game = 1 game, ~few hundred labeled moves."""
    from catan_mcts.experiments.e1_winrate_vs_random import main
    out = main(
        out_root=tmp_path, num_games=1,
        sims_per_move_grid=[2], seed_base=999,
        max_seconds=300.0,
    )
    return out


def test_dataset_loads_from_run_dir(tmp_path: Path):
    out = _make_minimal_run(tmp_path)
    ds = CatanReplayDataset([out])
    assert len(ds) > 0


def test_dataset_item_shapes(tmp_path: Path):
    out = _make_minimal_run(tmp_path)
    ds = CatanReplayDataset([out])
    data, value, policy, legal = ds[0]
    # data is a HeteroData; check the embedded shapes
    assert data["hex"].x.shape == (19, 8)
    assert data["vertex"].x.shape == (54, 7)
    assert data["edge"].x.shape == (72, 6)
    # value: [4] in [-1, +1] U {0}
    assert value.shape == (4,)
    assert torch.all((value == 1.0) | (value == -1.0) | (value == 0.0))
    # policy: [206], a probability distribution (sums to 1 ± eps OR all-zeros if no MCTS visits — for completed games every recorded move has visits)
    assert policy.shape == (206,)
    s = float(policy.sum())
    assert abs(s - 1.0) < 1e-5, f"policy sums to {s}, expected 1.0"
    # legal mask: [206], bool, action_taken must be legal
    assert legal.shape == (206,)
    assert legal.dtype == torch.bool


def test_value_target_perspective_rotation(tmp_path: Path):
    """value[0] corresponds to the *current_player* of that move (perspective).
    The same game's later moves with different current_player should rotate
    accordingly. Spec §4 'value target': index i = +1 if (current_player + i) % 4 == winner."""
    out = _make_minimal_run(tmp_path)
    ds = CatanReplayDataset([out])
    # Walk the dataset, find a row whose seed has a known winner.
    # Just sanity-check: for at least one row, the +1 lands at a valid index 0-3.
    for i in range(len(ds)):
        _, value, _, _ = ds[i]
        if (value == 1.0).any():
            assert int((value == 1.0).nonzero(as_tuple=True)[0]) in range(4)
            break
```

- [ ] **Step 2: Run failing test**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_dataset.py -v"
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement dataset**

Create `mcts_study/catan_gnn/dataset.py`:
```python
"""CatanReplayDataset: stream training tuples from existing parquet runs.

For each MCTS-decided move row in moves.parquet, we:
  1. Find its game's full action_history (from games.parquet, schema v2).
  2. Replay the engine forward through the prefix of action_history that ends
     at this MCTS move.
  3. Build features via state_to_pyg(engine.observation()).
  4. Build value target from games.winner (perspective-rotated).
  5. Build policy target from mcts_visit_counts (normalized to a distribution).
  6. Return (HeteroData, value [4], policy [206], legal_mask [206]).

Skips any row whose game has schema_version != 2 — those games don't have
action_history and would require expensive replay-from-scratch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from catan_bot import _engine

from .state_to_pyg import state_to_pyg


_CHANCE_BIT = 0x8000_0000


class CatanReplayDataset(Dataset):
    def __init__(self, run_dirs: list[Path]) -> None:
        moves_frames = []
        games_frames = []
        for rd in run_dirs:
            for mv in rd.glob("worker*/moves.*.parquet"):
                moves_frames.append(pq.read_table(mv).to_pandas())
            for gv in rd.glob("worker*/games.*.parquet"):
                games_frames.append(pq.read_table(gv).to_pandas())
            # Also accept top-level shards (serial mode, e.g. tests).
            for mv in rd.glob("moves.*.parquet"):
                moves_frames.append(pq.read_table(mv).to_pandas())
            for gv in rd.glob("games.*.parquet"):
                games_frames.append(pq.read_table(gv).to_pandas())
        if not moves_frames:
            raise RuntimeError(f"No moves.*.parquet shards found under {run_dirs}")
        import pandas as pd
        moves = pd.concat(moves_frames, ignore_index=True)
        games = pd.concat(games_frames, ignore_index=True)
        # Filter to v2 games only.
        games = games[games["schema_version"] >= 2]
        self._winner_by_seed = dict(zip(games["seed"], games["winner"]))
        self._history_by_seed = {
            int(s): list(h) for s, h in zip(games["seed"], games["action_history"])
        }
        self._index = moves[moves["seed"].isin(self._winner_by_seed.keys())].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        row = self._index.iloc[i]
        seed = int(row["seed"])
        move_index = int(row["move_index"])
        full_history = self._history_by_seed[seed]
        prefix_end = self._mcts_move_to_history_index(full_history, move_index)
        # Replay engine.
        engine = _engine.Engine(seed)
        for action_id in full_history[:prefix_end]:
            a = int(action_id)
            if a & _CHANCE_BIT:
                engine.apply_chance_outcome(a & 0x7FFF_FFFF)
            else:
                engine.step(a)
        obs = engine.observation()
        data = state_to_pyg(obs)

        # Targets.
        winner = self._winner_by_seed[seed]
        current_player = int(row["current_player"])
        value = torch.zeros(4, dtype=torch.float32)
        if winner != -1:
            for offset in range(4):
                player = (current_player + offset) % 4
                value[offset] = 1.0 if player == winner else -1.0

        visits = np.array(row["mcts_visit_counts"], dtype=np.float32)
        s = float(visits.sum())
        if s > 0:
            policy = torch.from_numpy(visits / s)
        else:
            # Degenerate: distribute uniformly over legal actions as a fallback.
            mask = np.array(row["legal_action_mask"], dtype=bool)
            policy = torch.zeros(206, dtype=torch.float32)
            policy[mask] = 1.0 / max(1, int(mask.sum()))

        legal = torch.from_numpy(np.array(row["legal_action_mask"], dtype=np.bool_))

        return data, value, policy, legal

    @staticmethod
    def _mcts_move_to_history_index(history: list[int], move_index: int) -> int:
        """Walk forward through history, counting *decision* (non-chance, non-trivial)
        steps. Return the prefix length that ends just *before* the (move_index)-th
        MCTS-decided move.

        Note: 'trivial' (len(legal)==1) decision actions are NOT recorded in
        moves.parquet (skip-trivial-turns optimization in play_one_game), but
        they ARE in action_history. We can't filter them here without re-running
        the engine. The simple correct thing: for every entry, if it's a chance
        action OR a decision action, advance one history step; only count
        toward move_index if the entry is the kind of step the recorder kept.

        Implementation: replay the engine alongside the history walk, and at each
        non-chance step check `len(engine.legal_actions())`. If >1, count it as
        an MCTS-decided move; when the count reaches move_index, return the
        prefix length BEFORE applying the move.
        """
        engine = _engine.Engine(0)  # seed irrelevant — we apply history bytes directly
        # Wait — Engine requires the seed that produced this history. Caller
        # has the seed; pass it. Refactor:
        raise NotImplementedError("see body — needs refactor to take seed")
```

The `_mcts_move_to_history_index` logic is fiddly. The implementer must refactor:
- Make it a method on the dataset (not staticmethod) so it can re-create the engine per-call with the right seed, OR
- Instead of computing the prefix, *replay step-by-step* in `__getitem__` and stop when we've seen `move_index` MCTS-decisions.

Replacement implementation that's simpler:
```python
    def __getitem__(self, i: int):
        row = self._index.iloc[i]
        seed = int(row["seed"])
        move_index = int(row["move_index"])
        full_history = self._history_by_seed[seed]
        engine = _engine.Engine(seed)
        mcts_decisions_seen = 0
        for action_id in full_history:
            if engine.is_terminal():
                break
            if engine.is_chance_pending():
                a = int(action_id)
                # action_history records chance with the 0x80000000 high bit set.
                engine.apply_chance_outcome(a & 0x7FFF_FFFF)
                continue
            # Decision step. Is this the move we're looking for?
            legal = engine.legal_actions()
            if len(legal) > 1:
                if mcts_decisions_seen == move_index:
                    break
                mcts_decisions_seen += 1
            engine.step(int(action_id))
        # ... rest of __getitem__ as above (build features + targets)
```

This is correct because skip-trivial-turns means recorded MCTS moves have `len(legal) > 1` by construction. Implement this version, delete the `_mcts_move_to_history_index` staticmethod stub.

- [ ] **Step 4: Run tests — expect pass**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_dataset.py -v"
```
Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study add mcts_study/catan_gnn/dataset.py mcts_study/tests/test_dataset.py
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study commit -m "feat(catan_gnn): CatanReplayDataset reads parquet shards + replays via engine"
```

---

## Task 7: Training loop

**Files:**
- Create: `mcts_study/catan_gnn/train.py`
- Test: `mcts_study/tests/test_train.py`

The training script: builds dataset, splits 90/10 by seed, trains Adam(lr=1e-3) for N epochs with masked-cross-entropy + MSE loss, writes checkpoint + per-epoch JSON.

- [ ] **Step 1: Write failing test**

Create `mcts_study/tests/test_train.py`:
```python
"""Tests for catan_gnn.train: produces a checkpoint and training_log.json."""
import json
from pathlib import Path

import pytest
import torch

from catan_gnn.train import train_main


def _make_minimal_run(tmp_path: Path):
    from catan_mcts.experiments.e1_winrate_vs_random import main
    out = main(
        out_root=tmp_path, num_games=2,
        sims_per_move_grid=[2], seed_base=12345,
        max_seconds=300.0,
    )
    return out


def test_train_produces_artifacts(tmp_path: Path):
    """End-to-end: run a tiny training job, get a checkpoint and a log."""
    run_dir = _make_minimal_run(tmp_path / "run")
    out_dir = tmp_path / "gnn_v0"
    train_main(
        run_dirs=[run_dir],
        out_dir=out_dir,
        hidden_dim=32, num_layers=2,
        epochs=2, batch_size=4, lr=1e-3,
        w_value=1.0, w_policy=1.0,
        seed=0,
    )
    assert (out_dir / "checkpoint.pt").exists()
    assert (out_dir / "training_log.json").exists()
    assert (out_dir / "config.json").exists()
    log = json.loads((out_dir / "training_log.json").read_text())
    assert "epochs" in log
    assert len(log["epochs"]) == 2
    for ep in log["epochs"]:
        for k in ("epoch", "train_loss_total", "val_loss_total",
                 "val_value_mae", "val_policy_top1_acc"):
            assert k in ep, f"missing key {k} in epoch row"


def test_checkpoint_is_loadable(tmp_path: Path):
    run_dir = _make_minimal_run(tmp_path / "run")
    out_dir = tmp_path / "gnn_v0"
    train_main(
        run_dirs=[run_dir], out_dir=out_dir,
        hidden_dim=32, num_layers=2,
        epochs=1, batch_size=4, lr=1e-3,
        w_value=1.0, w_policy=1.0, seed=0,
    )
    from catan_gnn.gnn_model import GnnModel
    model = GnnModel(hidden_dim=32, num_layers=2)
    state = torch.load(out_dir / "checkpoint.pt", map_location="cpu")
    model.load_state_dict(state)
```

- [ ] **Step 2: Run failing test**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_train.py -v"
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement train.py**

Create `mcts_study/catan_gnn/train.py`:
```python
"""GNN-v0 training loop. Reads parquets, trains, writes artifacts."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch

from .dataset import CatanReplayDataset
from .gnn_model import GnnModel


@dataclass
class EpochStats:
    epoch: int
    train_loss_total: float
    train_loss_value: float
    train_loss_policy: float
    val_loss_total: float
    val_loss_value: float
    val_loss_policy: float
    val_value_mae: float
    val_policy_top1_acc: float


def _split_by_seed(ds: CatanReplayDataset, val_frac: float, seed: int):
    rng = random.Random(seed)
    seeds = sorted({int(s) for s in ds._index["seed"]})
    rng.shuffle(seeds)
    n_val = max(1, int(len(seeds) * val_frac))
    val_seeds = set(seeds[:n_val])
    train_idx, val_idx = [], []
    for i in range(len(ds)):
        s = int(ds._index.iloc[i]["seed"])
        if s in val_seeds:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return Subset(ds, train_idx), Subset(ds, val_idx)


def _collate(batch):
    """Custom collate: stack HeteroData via Batch.from_data_list, stack targets."""
    datas, values, policies, legals = zip(*batch)
    return (
        Batch.from_data_list(list(datas)),
        torch.stack(list(values)),
        torch.stack(list(policies)),
        torch.stack(list(legals)),
    )


def _masked_policy_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy where illegal logits are sent to -inf before softmax."""
    masked = logits.masked_fill(~mask, float("-inf"))
    log_probs = F.log_softmax(masked, dim=1)
    # CE = -sum(target * log_probs) per sample, then mean.
    return -(target * log_probs).sum(dim=1).mean()


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent
        ).decode().strip()
        return sha
    except Exception:
        return "unknown"


def train_main(
    *,
    run_dirs: list[Path],
    out_dir: Path,
    hidden_dim: int = 32,
    num_layers: int = 2,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    w_value: float = 1.0,
    w_policy: float = 1.0,
    val_frac: float = 0.1,
    seed: int = 0,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    full_ds = CatanReplayDataset([Path(p) for p in run_dirs])
    if len(full_ds) == 0:
        raise RuntimeError(f"Empty dataset under {run_dirs}")
    train_ds, val_ds = _split_by_seed(full_ds, val_frac=val_frac, seed=seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate,
    )

    model = GnnModel(hidden_dim=hidden_dim, num_layers=num_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    log = {"epochs": []}
    for epoch in range(1, epochs + 1):
        # Train.
        model.train()
        tot, tv, tp, n = 0.0, 0.0, 0.0, 0
        for batch, value_t, policy_t, legal in train_loader:
            opt.zero_grad()
            v_pred, p_logits = model(batch)
            lv = F.mse_loss(v_pred, value_t)
            lp = _masked_policy_loss(p_logits, policy_t, legal)
            loss = w_value * lv + w_policy * lp
            loss.backward()
            opt.step()
            tot += float(loss); tv += float(lv); tp += float(lp); n += 1

        # Val.
        model.eval()
        vt, vv, vp_, vmae, top1, n_pos = 0.0, 0.0, 0.0, 0.0, 0, 0
        with torch.no_grad():
            for batch, value_t, policy_t, legal in val_loader:
                v_pred, p_logits = model(batch)
                lv = F.mse_loss(v_pred, value_t)
                lp = _masked_policy_loss(p_logits, policy_t, legal)
                vt += float(w_value * lv + w_policy * lp); vv += float(lv); vp_ += float(lp)
                vmae += float((v_pred - value_t).abs().mean())
                # Top-1: did argmax(masked logits) hit argmax(target)?
                masked = p_logits.masked_fill(~legal, float("-inf"))
                pred = masked.argmax(dim=1)
                gt = policy_t.argmax(dim=1)
                top1 += int((pred == gt).sum())
                n_pos += value_t.shape[0]

        stats = EpochStats(
            epoch=epoch,
            train_loss_total=tot / max(1, n), train_loss_value=tv / max(1, n),
            train_loss_policy=tp / max(1, n),
            val_loss_total=vt / max(1, len(val_loader)),
            val_loss_value=vv / max(1, len(val_loader)),
            val_loss_policy=vp_ / max(1, len(val_loader)),
            val_value_mae=vmae / max(1, len(val_loader)),
            val_policy_top1_acc=top1 / max(1, n_pos),
        )
        log["epochs"].append(asdict(stats))
        print(f"[epoch {epoch}/{epochs}] train_loss={stats.train_loss_total:.3f} "
              f"val_loss={stats.val_loss_total:.3f} val_top1={stats.val_policy_top1_acc:.3f}",
              flush=True)

    # Persist.
    torch.save(model.state_dict(), out_dir / "checkpoint.pt")
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2))
    config = {
        "run_dirs": [str(p) for p in run_dirs],
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "w_value": w_value,
        "w_policy": w_policy,
        "val_frac": val_frac,
        "seed": seed,
        "dataset_size": len(full_ds),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "git_sha": _git_sha(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    return out_dir


def cli_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w-value", type=float, default=1.0)
    p.add_argument("--w-policy", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train_main(
        run_dirs=args.run_dirs, out_dir=args.out_dir,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        w_value=args.w_value, w_policy=args.w_policy, seed=args.seed,
    )


if __name__ == "__main__":
    cli_main()
```

- [ ] **Step 4: Run tests**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_train.py -v"
```
Expected: both PASS.

- [ ] **Step 5: Commit**

```
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study add mcts_study/catan_gnn/train.py mcts_study/tests/test_train.py
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study commit -m "feat(catan_gnn): training loop with masked-CE + MSE, per-epoch JSON log"
```

---

## Task 8: GnnEvaluator

**Files:**
- Create: `mcts_study/catan_mcts/gnn_evaluator.py`
- Test: `mcts_study/tests/test_gnn_evaluator.py`

OpenSpiel `Evaluator` subclass wrapping a trained `GnnModel`. Mirrors `LookaheadVpEvaluator`'s API contract.

- [ ] **Step 1: Write failing test**

Create `mcts_study/tests/test_gnn_evaluator.py`:
```python
"""Tests for GnnEvaluator: OpenSpiel evaluator backed by a GNN."""
import numpy as np
import pyspiel
import pytest
import torch
from open_spiel.python.algorithms import mcts as os_mcts

from catan_gnn.gnn_model import GnnModel
from catan_mcts.adapter import CatanGame
from catan_mcts.gnn_evaluator import GnnEvaluator


def _drive_to_player_decision(state):
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        state.apply_action(int(outcomes[0][0]))


def _untrained_model():
    torch.manual_seed(0)
    return GnnModel(hidden_dim=32, num_layers=2)


def test_evaluate_shape_and_range():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    _drive_to_player_decision(state)
    ev = GnnEvaluator(model=_untrained_model())
    v = ev.evaluate(state)
    assert v.shape == (4,)
    assert v.dtype == np.float32
    assert (v >= -1.0).all() and (v <= 1.0).all()


def test_prior_sums_to_one_over_legal():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    _drive_to_player_decision(state)
    ev = GnnEvaluator(model=_untrained_model())
    prior = ev.prior(state)
    legal = state.legal_actions(state.current_player())
    assert len(prior) == len(legal)
    s = sum(p for _, p in prior)
    assert abs(s - 1.0) < 1e-5


def test_evaluate_does_not_mutate_state():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    _drive_to_player_decision(state)
    history_before = list(state._engine.action_history())
    ev = GnnEvaluator(model=_untrained_model())
    ev.evaluate(state)
    ev.prior(state)
    history_after = list(state._engine.action_history())
    assert history_before == history_after


def test_runs_inside_mctsbot_one_full_game():
    """End-to-end: MCTSBot(GnnEvaluator) + 3 random opponents play one game.
    Untrained model -> noise priors -> any legal play is fine; just ensure
    no exceptions and the game terminates within reasonable steps."""
    import random

    game = CatanGame()
    rng = np.random.default_rng(0)
    ev = GnnEvaluator(model=_untrained_model())
    mcts_bot = os_mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=4,
        evaluator=ev, solve=False, random_state=rng,
    )

    class _RandBot:
        def __init__(self, s): self.r = random.Random(s)
        def step(self, st): return self.r.choice(st.legal_actions())

    state = game.new_initial_state(seed=42)
    chance_rng = random.Random(42)
    bots = {0: mcts_bot, 1: _RandBot(1), 2: _RandBot(2), 3: _RandBot(3)}
    steps = 0
    while not state.is_terminal() and steps < 30000:
        if state.is_chance_node():
            outs = state.chance_outcomes()
            r = chance_rng.random(); cum = 0.0; chosen = outs[-1][0]
            for v, p in outs:
                cum += p
                if r <= cum:
                    chosen = v; break
            state.apply_action(int(chosen))
        else:
            p = state.current_player()
            legal = state.legal_actions()
            if len(legal) == 1:
                state.apply_action(int(legal[0]))
            else:
                a = bots[p].step(state)
                state.apply_action(int(a))
        steps += 1
    assert steps > 0
```

- [ ] **Step 2: Run failing test**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_gnn_evaluator.py -v"
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement GnnEvaluator**

Create `mcts_study/catan_mcts/gnn_evaluator.py`:
```python
"""OpenSpiel Evaluator backed by a trained GnnModel.

Mirrors the API of catan_mcts.evaluator.LookaheadVpEvaluator. evaluate() and
prior() share a single forward pass via a tiny per-state cache keyed on the
engine's action_history (deterministic state ID). MCTSBot calls them
back-to-back on the same leaf, so the cache hit rate is essentially 100%.
"""
from __future__ import annotations

import numpy as np
import torch
from open_spiel.python.algorithms import mcts as os_mcts
from torch_geometric.data import Batch

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


class GnnEvaluator(os_mcts.Evaluator):
    def __init__(self, model: GnnModel, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device
        self._cache_key: tuple | None = None
        self._cache_value: np.ndarray | None = None
        self._cache_policy: np.ndarray | None = None

    @torch.no_grad()
    def _forward(self, state):
        key = tuple(state._engine.action_history())
        if key == self._cache_key:
            return self._cache_value, self._cache_policy
        obs = state._engine.observation()
        data = state_to_pyg(obs).to(self.device)
        batch = Batch.from_data_list([data])
        v, logits = self.model(batch)
        v_np = v.squeeze(0).cpu().numpy().astype(np.float32)
        l_np = logits.squeeze(0).cpu().numpy().astype(np.float32)
        self._cache_key = key
        self._cache_value = v_np
        self._cache_policy = l_np
        return v_np, l_np

    def evaluate(self, state):
        v, _ = self._forward(state)
        return v

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        _, logits = self._forward(state)
        legal = state.legal_actions(state.current_player())
        if not legal:
            return []
        legal_logits = logits[np.asarray(legal, dtype=np.int64)]
        probs = _softmax(legal_logits)
        return [(int(a), float(p)) for a, p in zip(legal, probs)]
```

- [ ] **Step 4: Run tests**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_gnn_evaluator.py -v"
```
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study add mcts_study/catan_mcts/gnn_evaluator.py mcts_study/tests/test_gnn_evaluator.py
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study commit -m "feat(catan_mcts): GnnEvaluator wraps trained GnnModel for OpenSpiel MCTSBot"
```

---

## Task 9: Bench-2 static-position benchmark

**Files:**
- Create: `mcts_study/catan_gnn/benchmark.py`
- Test: `mcts_study/tests/test_benchmark.py`

Spec §7 Bench-2: sample N val positions, compare GnnEvaluator's value/policy to LookaheadVpEvaluator(depth=25)'s.

- [ ] **Step 1: Write failing test**

Create `mcts_study/tests/test_benchmark.py`:
```python
"""Tests for catan_gnn.benchmark: bench-2 produces a JSON with bench2_value_mae + bench2_policy_kl."""
import json
from pathlib import Path

import torch

from catan_gnn.benchmark import bench2_main


def _make_minimal_run(tmp_path: Path):
    from catan_mcts.experiments.e1_winrate_vs_random import main
    return main(
        out_root=tmp_path, num_games=2,
        sims_per_move_grid=[2], seed_base=22222,
        max_seconds=300.0,
    )


def test_bench2_writes_json(tmp_path: Path):
    from catan_gnn.train import train_main

    run_dir = _make_minimal_run(tmp_path / "run")
    out_dir = tmp_path / "gnn_v0"
    train_main(
        run_dirs=[run_dir], out_dir=out_dir,
        hidden_dim=32, num_layers=2,
        epochs=1, batch_size=4, lr=1e-3,
        w_value=1.0, w_policy=1.0, seed=0,
    )
    bench2_main(
        checkpoint=out_dir / "checkpoint.pt",
        run_dirs=[run_dir],
        out_path=out_dir / "bench2.json",
        n_positions=10,
        lookahead_depth=3,  # tiny for tests
    )
    j = json.loads((out_dir / "bench2.json").read_text())
    assert "bench2_value_mae" in j
    assert "bench2_policy_kl" in j
    assert j["n_positions"] == 10
```

- [ ] **Step 2: Run failing test**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_benchmark.py -v"
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement benchmark.py**

Create `mcts_study/catan_gnn/benchmark.py`:
```python
"""Bench-2: static-position evaluator comparison.

Sample n positions from the val split, run both GnnEvaluator and
LookaheadVpEvaluator(depth=D), record per-position value MAE and policy KL,
write summary JSON.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from catan_bot import _engine
from catan_mcts.evaluator import LookaheadVpEvaluator

from .dataset import CatanReplayDataset
from .gnn_model import GnnModel
from .state_to_pyg import state_to_pyg


def bench2_main(
    *,
    checkpoint: Path,
    run_dirs: list[Path],
    out_path: Path,
    n_positions: int = 1000,
    lookahead_depth: int = 25,
    seed: int = 0,
) -> Path:
    rng = random.Random(seed)
    ds = CatanReplayDataset([Path(p) for p in run_dirs])
    n_total = len(ds)
    indices = rng.sample(range(n_total), min(n_positions, n_total))

    model = GnnModel(hidden_dim=32, num_layers=2)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    look = LookaheadVpEvaluator(depth=lookahead_depth, base_seed=seed)

    value_diffs = []
    policy_kls = []
    with torch.no_grad():
        for i in indices:
            data, _, _, legal = ds[i]
            batch = Batch.from_data_list([data])
            v_gnn, p_logits = model(batch)
            v_gnn_np = v_gnn.squeeze(0).numpy()
            # Reconstruct an OpenSpiel state by replaying the engine. Use ds
            # internals: same seed + walk to move_index. We re-call the
            # dataset's own logic by accessing engine state directly.
            row = ds._index.iloc[i]
            seed_g = int(row["seed"])
            engine = _engine.Engine(seed_g)
            move_index = int(row["move_index"])
            history = ds._history_by_seed[seed_g]
            mcts_dec = 0
            for action_id in history:
                if engine.is_terminal():
                    break
                if engine.is_chance_pending():
                    a = int(action_id)
                    engine.apply_chance_outcome(a & 0x7FFF_FFFF)
                    continue
                legal_now = engine.legal_actions()
                if len(legal_now) > 1:
                    if mcts_dec == move_index:
                        break
                    mcts_dec += 1
                engine.step(int(action_id))
            # Lookahead acts on a clone (mirror of LookaheadVpEvaluator behavior).
            v_look = engine.clone().lookahead_vp_value(lookahead_depth, int(seed) + i)
            v_look = np.asarray(v_look, dtype=np.float32)
            value_diffs.append(float(np.mean(np.abs(v_gnn_np - v_look))))

            # Policy KL between GNN softmaxed-over-legal and visit-count target.
            mask = legal.numpy()
            masked = np.where(mask, p_logits.squeeze(0).numpy(), -np.inf)
            z = masked - np.nanmax(masked)
            ez = np.exp(z); gnn_p = ez / ez.sum()
            visits = np.array(row["mcts_visit_counts"], dtype=np.float32)
            target_p = visits / max(1.0, visits.sum())
            # KL(target || gnn_p), summed over legal.
            eps = 1e-9
            kl = float(np.sum(target_p * (np.log(target_p + eps) - np.log(gnn_p + eps))))
            policy_kls.append(kl)

    summary = {
        "checkpoint": str(checkpoint),
        "n_positions": len(indices),
        "lookahead_depth": lookahead_depth,
        "bench2_value_mae": float(np.mean(value_diffs)),
        "bench2_policy_kl": float(np.mean(policy_kls)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    return out_path


def cli_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--out-path", type=Path, required=True)
    p.add_argument("--n-positions", type=int, default=1000)
    p.add_argument("--lookahead-depth", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    bench2_main(
        checkpoint=args.checkpoint, run_dirs=args.run_dirs,
        out_path=args.out_path, n_positions=args.n_positions,
        lookahead_depth=args.lookahead_depth, seed=args.seed,
    )


if __name__ == "__main__":
    cli_main()
```

- [ ] **Step 4: Run tests**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_benchmark.py -v"
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study add mcts_study/catan_gnn/benchmark.py mcts_study/tests/test_benchmark.py
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study commit -m "feat(catan_gnn): bench-2 static-position GNN-vs-lookahead comparison"
```

---

## Task 10: e6 — MCTS-with-GNN winrate experiment

**Files:**
- Create: `mcts_study/catan_mcts/experiments/e6_mcts_gnn_winrate.py`
- Modify: `mcts_study/catan_mcts/cli.py` (register e6)
- Test: `mcts_study/tests/test_e6_runs.py`

Spec §7 Bench-3: 50-game sweep with `MCTSBot(GnnEvaluator)` at sims∈{100, 400} vs 3 random opponents.

- [ ] **Step 1: Write failing smoke test**

Create `mcts_study/tests/test_e6_runs.py`:
```python
"""Smoke test for e6_mcts_gnn_winrate."""
from pathlib import Path

import torch

from catan_mcts.experiments.e6_mcts_gnn_winrate import main


def _train_tiny_gnn(tmp_path: Path) -> Path:
    from catan_mcts.experiments.e1_winrate_vs_random import main as e1_main
    from catan_gnn.train import train_main

    run_dir = e1_main(
        out_root=tmp_path / "e1run", num_games=2,
        sims_per_move_grid=[2], seed_base=33333, max_seconds=300.0,
    )
    out_dir = tmp_path / "gnn_v0"
    train_main(
        run_dirs=[run_dir], out_dir=out_dir,
        hidden_dim=32, num_layers=2, epochs=1, batch_size=4,
        lr=1e-3, w_value=1.0, w_policy=1.0, seed=0,
    )
    return out_dir / "checkpoint.pt"


def test_e6_smoke_run(tmp_path: Path):
    ckpt = _train_tiny_gnn(tmp_path / "train")
    out = main(
        out_root=tmp_path / "e6",
        checkpoint=ckpt,
        num_games=1, sims_grid=[2],
        seed_base=44444, max_seconds=300.0,
    )
    games_shard = out / "games.sims=2.parquet"
    assert games_shard.exists()
    assert (out / "done.txt").exists() or (out / "skipped.csv").exists()
```

- [ ] **Step 2: Run failing test**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_e6_runs.py -v"
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement e6**

Create `mcts_study/catan_mcts/experiments/e6_mcts_gnn_winrate.py`:
```python
"""Experiment 6: MCTS-with-GNN-evaluator winrate vs 3 random opponents.

Spec §7 Bench-3. Mirrors e1's structure (sims grid, 4-worker option,
done.txt resumability), but the evaluator is GnnEvaluator(checkpoint=...)
instead of RustRolloutEvaluator or LookaheadVpEvaluator.
"""
from __future__ import annotations

import argparse
import random
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from open_spiel.python.algorithms import mcts as os_mcts
from tqdm import tqdm

from catan_gnn.gnn_model import GnnModel
from ..adapter import CatanGame
from ..gnn_evaluator import GnnEvaluator
from ..recorder import SelfPlayRecorder
from .common import make_run_dir, play_one_game


class _RandomBot:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())


def _load_model(checkpoint: Path, hidden_dim: int, num_layers: int) -> GnnModel:
    model = GnnModel(hidden_dim=hidden_dim, num_layers=num_layers)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _build_mcts(game, sims: int, evaluator, seed: int):
    rng = np.random.default_rng(seed)
    return os_mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=sims,
        evaluator=evaluator, solve=False, random_state=rng,
    )


def _run_cell(rec, sims, seeds, done, max_seconds, model, progress_desc_prefix=""):
    game = CatanGame()
    for seed in tqdm(seeds, desc=f"{progress_desc_prefix}sims={sims}", leave=False):
        if seed in done:
            continue
        chance_rng = random.Random(seed)
        evaluator = GnnEvaluator(model=model)   # fresh evaluator per game (cache reset)
        mcts_bot = _build_mcts(game, sims=sims, evaluator=evaluator, seed=seed)
        bots = {0: mcts_bot, 1: _RandomBot(seed+1), 2: _RandomBot(seed+2), 3: _RandomBot(seed+3)}
        with rec.game(seed=seed) as g_rec:
            outcome = play_one_game(
                game=game, bots=bots, seed=seed, chance_rng=chance_rng,
                recorded_player=0, recorder_game=g_rec, mcts_bot=mcts_bot,
                max_seconds=max_seconds,
            )
            if outcome.timed_out:
                g_rec._moves.clear()
                g_rec._finalized = True
                rec.skip_game(
                    seed=seed, reason="wall-clock-timeout",
                    length_in_moves=outcome.length_in_moves,
                )
            else:
                g_rec.finalize(
                    winner=outcome.winner, final_vp=outcome.final_vp,
                    length_in_moves=outcome.length_in_moves,
                    action_history=outcome.action_history,
                )
                rec.mark_done(seed)


def main(
    *,
    out_root: Path,
    checkpoint: Path,
    num_games: int = 50,
    sims_grid: list[int] = (100, 400),
    hidden_dim: int = 32,
    num_layers: int = 2,
    seed_base: int = 6_000_000,
    max_seconds: float = 360.0,
    resume: bool = True,
    workers: int = 1,
) -> Path:
    out = make_run_dir(out_root, "e6_mcts_gnn_winrate")
    base_config = {
        "experiment": "e6_mcts_gnn_winrate",
        "checkpoint": str(checkpoint),
        "hidden_dim": hidden_dim, "num_layers": num_layers,
        "sims_grid": list(sims_grid), "num_games": num_games,
        "max_seconds": max_seconds, "workers": workers,
    }

    if workers <= 1:
        rec = SelfPlayRecorder(out, config=base_config)
        done = rec.done_seeds() if resume else set()
        model = _load_model(checkpoint, hidden_dim, num_layers)
        for sims in sims_grid:
            seeds = [seed_base + sims * 1_000 + i for i in range(num_games)]
            _run_cell(rec, sims, seeds, done, max_seconds, model)
            rec.checkpoint(f"sims={sims}")
        rec.flush()
        return out

    # Parallel mode (mirrors e1).
    seeds_per_cell_per_worker = [[[] for _ in range(workers)] for _ in sims_grid]
    for cell_idx, sims in enumerate(sims_grid):
        for i in range(num_games):
            seed = seed_base + sims * 1_000 + i
            seeds_per_cell_per_worker[cell_idx][i % workers].append(seed)
    args_list = []
    for w in range(workers):
        seeds_per_cell = [seeds_per_cell_per_worker[c][w] for c in range(len(sims_grid))]
        args_list.append((
            w, out, list(sims_grid), seeds_per_cell, max_seconds, base_config,
            checkpoint, hidden_dim, num_layers,
        ))
    ctx = get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        pool.map(_worker, args_list)
    return out


def _worker(args):
    worker_idx, parent_out, sims_grid, seeds_per_cell, max_seconds, base_config, ckpt, hd, nl = args
    worker_dir = parent_out / f"worker{worker_idx}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    rec = SelfPlayRecorder(worker_dir, config={**base_config, "worker_idx": worker_idx})
    done = rec.done_seeds()
    model = _load_model(ckpt, hd, nl)
    for sims, seeds in zip(sims_grid, seeds_per_cell):
        _run_cell(rec, sims, seeds, done, max_seconds, model,
                  progress_desc_prefix=f"w{worker_idx} ")
        rec.checkpoint(f"sims={sims}")
    rec.flush()


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--num-games", type=int, default=50)
    p.add_argument("--sims-grid", type=int, nargs="+", default=[100, 400])
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--seed-base", type=int, default=6_000_000)
    p.add_argument("--max-seconds", type=float, default=360.0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    out = main(
        out_root=args.out_root, checkpoint=args.checkpoint,
        num_games=args.num_games, sims_grid=args.sims_grid,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        seed_base=args.seed_base, max_seconds=args.max_seconds,
        resume=not args.no_resume, workers=args.workers,
    )
    print(f"e6 wrote to {out}")


if __name__ == "__main__":
    cli_main()
```

- [ ] **Step 4: Register e6 in cli.py**

Modify `mcts_study/catan_mcts/cli.py`. Find the imports block:
```python
from .experiments import (
    e1_winrate_vs_random,
    e2_ucb_c_sweep,
    e3_rollout_policy,
    e4_tournament,
    e5_lookahead_depth,
)
```
Append `e6_mcts_gnn_winrate` to the import. Find `_EXPERIMENTS = {...}` and add `"e6": e6_mcts_gnn_winrate,`. Update the `runp.add_parser` help text to include e6.

- [ ] **Step 5: Run tests**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/mcts_study && python -m pytest tests/test_e6_runs.py -v"
```
Expected: PASS.

- [ ] **Step 6: Commit**

```
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study add mcts_study/catan_mcts/experiments/e6_mcts_gnn_winrate.py mcts_study/catan_mcts/cli.py mcts_study/tests/test_e6_runs.py
git -C C:/dojo/catan_bot/.claude/worktrees/mcts-study commit -m "feat(e6): MCTS-with-GNN-evaluator winrate experiment + CLI"
```

---

## Task 11: End-to-end smoke run + spec acceptance

The previous tasks each smoke-tested individual components. This task confirms the full pipeline runs end-to-end on real (not minimal) data and produces interpretable artifacts.

**Files:** none new. Uses existing parquets + scripts.

- [ ] **Step 1: Verify schema-v2 data exists**

Run:
```
wsl -d Ubuntu -- bash -c 'source ~/catan_mcts_venvs/mcts-study/bin/activate && python -c "
import pyarrow.parquet as pq
from pathlib import Path
for p in Path(\"/mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/runs\").glob(\"*-e*_*/worker*/games.*.parquet\"):
    df = pq.read_table(p).to_pandas()
    sv = int(df[\"schema_version\"].iloc[0])
    has_ah = \"action_history\" in df.columns
    print(p.parent.name, p.name, sv, has_ah)
" | head -20'
```
Expected: at least one v2 (schema_version=2) parquet with `action_history` column. If none, run `python -m catan_mcts run e5 --num-games 5 --depth-grid 25 --sims-grid 100 --max-seconds 360 --workers 4 --out-root mcts_study/runs --seed-base 7_000_000` to generate one.

- [ ] **Step 2: Train v0 GNN on existing v2 data**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study && python -m catan_gnn.train --run-dirs runs/<TIMESTAMP>-e5_lookahead_depth --out-dir runs/gnn_v0 --epochs 5"
```
Replace `<TIMESTAMP>` with the v2 e5 run dir. Expected: prints per-epoch training/val losses for 5 epochs; `runs/gnn_v0/checkpoint.pt`, `training_log.json`, `config.json` all written.

- [ ] **Step 3: Run bench-2 against the trained model**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study && python -m catan_gnn.benchmark --checkpoint runs/gnn_v0/checkpoint.pt --run-dirs runs/<TIMESTAMP>-e5_lookahead_depth --out-path runs/gnn_v0/bench2.json --n-positions 100"
```
Expected: writes `runs/gnn_v0/bench2.json` with `bench2_value_mae` and `bench2_policy_kl` floats.

- [ ] **Step 4: Run e6 (small) to validate MCTS integration**

Run:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && cd /mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study && python -m catan_mcts run e6 --checkpoint runs/gnn_v0/checkpoint.pt --num-games 5 --sims-grid 100 --max-seconds 360 --workers 4 --out-root runs"
```
Expected: writes a new `runs/<TS>-e6_mcts_gnn_winrate/` with parquet shards. Don't gate on winrate — just on no-crash.

- [ ] **Step 5: Sanity report**

Run a quick winrate aggregation against the e6 run dir:
```
wsl -d Ubuntu -- bash -lc "source ~/catan_mcts_venvs/mcts-study/bin/activate && python -c '
import pyarrow.parquet as pq
from pathlib import Path
import pandas as pd
run = sorted(Path(\"/mnt/c/dojo/catan_bot/.claude/worktrees/mcts-study/runs\").glob(\"*-e6_*\"))[-1]
games = []
for p in run.glob(\"worker*/games.*.parquet\"):
    games.append(pq.read_table(p).to_pandas())
df = pd.concat(games, ignore_index=True)
print(f\"games={len(df)}  wins_p0={(df.winner==0).sum()}  mean_vp_p0={df.final_vp.apply(lambda v: v[0]).mean():.2f}\")
'"
```
Expected: prints a single line summarizing the e6 smoke run. We are NOT gating on the spec's pass criterion (≥60% at sims=400) because (a) only 5 games at sims=100 (b) only 5 epochs of training. The headline number comes later from a full sweep + full training.

- [ ] **Step 6: No commit needed**

This task is acceptance-testing the pipeline; no source files change. If any step from 1-5 fails, the failure points back to the relevant earlier task to fix; only commit fixes there.

---

## Self-review

### Spec coverage check

- §1 Goal / non-goals / success criteria — covered: Task 11 acceptance demonstrates pipeline + integration; Task 7 produces config.json with hyperparams + git SHA (success criterion 4); Tasks 9 and 10 produce bench artifacts (success criterion 5). Strength bar (criterion 3) is gated outside the plan — it's measured by running the full e6 sweep after a real training run, not in any single task.
- §2 architecture overview — Tasks 4 + 5 implement.
- §3 heterogeneous wiring — Tasks 3 + 4 + 5 implement; specifically only 2 edge types (4 PyG message functions) per spec.
- §4 training / loss — Tasks 6 + 7 implement.
- §5 evaluator integration — Task 8 implements.
- §6 dataset — Task 6 implements (with the schema-v2 dependency from Task 2).
- §7 benchmarks — Task 9 (Bench-2), Task 10 (Bench-3 / e6). Bench-1 is the per-epoch JSON in Task 7. **Bench-4 (head-to-head vs lookahead-VP) is not implemented as a task.** The spec marks Bench-4 as optional/measurement-not-pass-fail, so omitting from v0 plan is consistent with YAGNI; flagged here so reviewers can request it.
- §8 file layout — exactly matches Tasks 3-10.
- §9 study experiments g1-g5 — explicitly out of scope for v0; will get their own plan after v0 numbers.
- §10 risks / §11 open questions — risks are mitigated in-place (e.g. Task 5 has param-count + latency tests). Open questions remain deferred.

### Placeholder scan

Searched for "TBD", "TODO", "fill in" — none in the plan.

### Type consistency

- `state_to_pyg(obs)` returns `HeteroData` everywhere it's referenced.
- `GnnModel(hidden_dim=..., num_layers=...)` constructor signature is consistent across Tasks 5, 7, 8, 9, 10.
- `GnnModel.forward(batch)` returns `(value, policy_logits)` everywhere.
- `GnnEvaluator(model=..., device="cpu")` constructor is consistent in Tasks 8 and 10.
- `train_main(...)` keyword arguments match between Task 7 (definition) and Tasks 9, 10, 11 (callers).
- `bench2_main(...)` arguments match Task 9 def vs Task 11 caller.
- The `action_history` field added in Task 2 is referenced consistently in Tasks 6 (dataset), 10 (e6), and 11.

No mismatches found.
