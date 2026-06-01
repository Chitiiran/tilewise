# Interactive Play-vs-Bots Mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A live local web app where the user plays Catan as one seat against per-seat-selectable bots (including any `.pt` checkpoint from the training library) and responds to bot trade requests, hosted alongside the existing offline replay viewer.

**Architecture:** A FastAPI server holds one live game per server-side session (`GameSession` owning a `CatanState` + bots). The driving loop cooperatively yields to the human and intercepts bot `ProposeTrade` actions *before* applying them (the engine auto-resolves trades, so trade-response lives in Python). A vanilla-JS single-page frontend talks to the server over REST + SSE, reusing the replay's board geometry / PNG / per-state serialization (extracted into shared modules). Local-first but deploy-ready: configurable paths, no engine changes.

**Tech Stack:** Python 3.10+, FastAPI + uvicorn (new deps), the Rust `catan_bot._engine` via the `CatanGame`/`CatanState` OpenSpiel adapter, existing bots (`GreedyBaselineBot`, `PureGnnBot`, `build_lookahead_mcts_v3`, `build_gnn_mcts_bot`), matplotlib (board PNG), vanilla JS + SSE frontend, pytest.

**Run location:** WSL Ubuntu (torch, checkpoints, maturin-built engine live there). All commands below run from `mcts_study/` with the mcts-study venv active. Reference: project memory "WSL setup for MCTS-study", "Rebuild PyO3 after engine changes" (NOT needed here — no Rust changes).

---

## Phase overview

- **Phase 1 — Shared-module extraction** (refactor `playback.py`; golden-guarded). Tasks 1-3.
- **Phase 2 — Bot registry** (discover + build bots/checkpoints). Tasks 4-5.
- **Phase 3 — Action decoding** (raw ints → UI-friendly action objects). Task 6.
- **Phase 4 — GameSession** (the core: driving loop + trade intercept). Tasks 7-11.
- **Phase 5 — FastAPI server** (REST + SSE). Tasks 12-15.
- **Phase 6 — Frontend** (setup lobby + game screen + replay tab). Tasks 16-20.
- **Phase 7 — Wiring & docs**. Tasks 21-22.

Each phase ends in a runnable, tested state. Frontend (Phase 6) is hard-gated only by the manual smoke check; the Python API tests are the contract gate.

---

## Phase 0: Dependencies

### Task 0: Add FastAPI + uvicorn to the env

**Files:**
- Modify: `mcts_study/pyproject.toml`

- [ ] **Step 1: Add the web dependencies**

Add a new optional-dependency group to `pyproject.toml` (keep core deps lean — the web server is opt-in). After the existing `[project.optional-dependencies]` `dev = [...]` block, add:

```toml
web = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "httpx>=0.27",       # FastAPI TestClient dependency
]
```

- [ ] **Step 2: Install into the active venv**

Run (WSL, mcts-study venv active, from `mcts_study/`):
```bash
pip install -e ".[web,dev]"
```
Expected: installs fastapi, uvicorn, starlette, httpx without touching torch/open_spiel.

- [ ] **Step 3: Verify imports**

Run:
```bash
python -c "import fastapi, uvicorn, httpx; from fastapi.testclient import TestClient; print('web deps OK')"
```
Expected: `web deps OK`

- [ ] **Step 4: Commit**

```bash
git add mcts_study/pyproject.toml
git commit -m "build(web): add fastapi/uvicorn/httpx optional deps"
```

---

## Phase 1: Shared-module extraction

Extract board geometry, the board PNG renderer, and per-state serialization out of `playback.py` into `web/board_layout.py` and `web/serializers.py`, so the live server and the offline replay share one source of truth. Guarded by a golden test so replay output does not change.

### Task 1: Golden snapshot of current replay output (regression guard FIRST)

**Files:**
- Test: `mcts_study/tests/test_serializers_golden.py`

This locks current behavior before we refactor. The test renders the same fixture as `test_playback.py` and snapshots the serialized states + layout, so the extraction in Tasks 2-3 is proven byte-identical.

- [ ] **Step 1: Write the golden-capture test**

Create `mcts_study/tests/test_serializers_golden.py`:

```python
"""Golden guard for the playback -> shared-module extraction.

Captures the per-state list and layout dict produced by the CURRENT
playback internals, so the Phase-1 refactor can be proven to preserve
them exactly. Run BEFORE extraction to write the golden; after extraction
the same assertions must still pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def minimal_run_dir(tmp_path_factory):
    from catan_mcts.experiments.e1_winrate_vs_random import main
    out_root = tmp_path_factory.mktemp("golden_runs")
    return main(
        out_root=out_root,
        num_games=1, sims_per_move_grid=[2],
        seed_base=4242, max_seconds=300.0,
    )


def test_replay_states_shape_is_stable(minimal_run_dir):
    """Every state dict carries the full field set the viewer renders."""
    from catan_mcts import playback
    seed = 4242 + 2 * 1_000
    history, winner, final_vp = playback._read_action_history(minimal_run_dir, seed)
    states = playback._replay_to_states(seed, history)
    assert len(states) >= 1
    required = {"n", "cp", "phase", "s", "c", "r", "rh", "vp", "hands",
                "bank", "dev_held", "ports", "lr_len", "knights", "built",
                "lr_holder", "la_holder", "vp_played"}
    for st in states:
        assert required.issubset(st.keys()), f"missing fields: {required - st.keys()}"
    # The whole list must be JSON-serializable (it ships to JS verbatim).
    json.dumps(states)
```

- [ ] **Step 2: Run it against current code (must PASS now)**

Run:
```bash
pytest tests/test_serializers_golden.py -v
```
Expected: PASS (this documents current behavior; it is the regression guard for Tasks 2-3).

- [ ] **Step 3: Commit**

```bash
git add mcts_study/tests/test_serializers_golden.py
git commit -m "test(playback): golden guard before shared-module extraction"
```

### Task 2: Extract board geometry + PNG into `web/board_layout.py`

**Files:**
- Create: `mcts_study/catan_mcts/web/__init__.py`
- Create: `mcts_study/catan_mcts/web/board_layout.py`
- Modify: `mcts_study/catan_mcts/playback.py` (import from the new module)
- Test: `mcts_study/tests/test_board_layout.py`

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_board_layout.py`:

```python
"""Tests for the extracted board geometry + PNG renderer."""
from __future__ import annotations

from pathlib import Path


def test_build_layout_returns_geometry():
    from catan_mcts.web import board_layout
    vertex_xy, edges, hex_centers = board_layout.build_layout()
    assert len(vertex_xy) == 54      # 54 board vertices
    assert len(edges) == 72          # 72 board edges
    assert len(hex_centers) == 19    # 19 hexes


def test_layout_dict_is_json_ready():
    from catan_mcts.web import board_layout
    d = board_layout.layout_dict()
    assert set(d.keys()) == {"xlim", "ylim", "vertices", "edges", "hex_centers"}
    assert len(d["vertices"]) == 54
    assert len(d["edges"]) == 72


def test_render_board_png_writes_file(tmp_path):
    from catan_mcts.web import board_layout
    out = tmp_path / "board.png"
    board_layout.render_board_png(seed=4242, out_path=out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run it to confirm it fails**

Run:
```bash
pytest tests/test_board_layout.py -v
```
Expected: FAIL — `ModuleNotFoundError: catan_mcts.web`.

- [ ] **Step 3: Create the package + module**

Create `mcts_study/catan_mcts/web/__init__.py`:

```python
"""Interactive play-vs-bots web app (FastAPI server + frontend).

This package hosts the live game server and shares board geometry and
per-state serialization with the offline replay viewer (catan_mcts.playback).
"""
```

Create `mcts_study/catan_mcts/web/board_layout.py` by **moving** these pieces verbatim out of `playback.py` (cut from playback, paste here): `ROW_LENGTHS`, `HEX_RADIUS`, `HEX_ROW_COL` construction, `_hex_center_pointy`, `_build_layout`, the plot-bounds constants (`XLIM`, `YLIM`, `FIG_WIDTH_INCHES`, `FIG_HEIGHT_INCHES`, `FIG_DPI`), the color constants (`RESOURCE_COLORS`, `DESERT_COLOR`, `RESOURCE_LABEL`, `RESOURCE_EMOJI`, `RESOURCE_LETTER`), the `PORTS` table + `PORT_KIND_TO_RESOURCE_IDX`, `_emoji_font_props`, `_shade`, and `_render_static_board_png`. Then add public wrappers:

```python
# (top of file)
from __future__ import annotations
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from catan_bot import _engine

# ... [all the moved constants + helpers above] ...


def build_layout():
    """Public: (vertex_xy: dict[int,(x,y)], edges: list[(x1,y1,x2,y2)], hex_centers: list[(x,y)])."""
    return _build_layout()


def layout_dict() -> dict:
    """Public: JSON-ready layout for the frontend (same shape playback emits)."""
    vertex_xy, edges, hex_centers = _build_layout()
    return {
        "xlim": list(XLIM),
        "ylim": list(YLIM),
        "vertices": {str(v): list(xy) for v, xy in vertex_xy.items()},
        "edges": [list(e) for e in edges],
        "hex_centers": [list(c) for c in hex_centers],
    }


def render_board_png(seed: int, out_path: Path, vertex_xy: dict | None = None) -> None:
    """Public: render the static board PNG for `seed`."""
    if vertex_xy is None:
        vertex_xy, _, _ = _build_layout()
    _render_static_board_png(seed, out_path, vertex_xy=vertex_xy)
```

- [ ] **Step 4: Update `playback.py` to import from the new module**

In `playback.py`, delete the moved definitions and add near the top:

```python
from catan_mcts.web.board_layout import (
    build_layout as _build_layout,
    layout_dict as _layout_dict,
    render_board_png as _render_static_board_png_public,
    XLIM, YLIM,
)
```

In `playback.render()`, replace the inline board-PNG call and layout dict with:
```python
    vertex_xy, edges, hex_centers = _build_layout()
    _render_static_board_png_public(seed, board_png, vertex_xy=vertex_xy)
    layout = _layout_dict()
```
(Keep everything else in `playback.py` unchanged — `_action_desc`, `_replay_to_states`, `INDEX_HTML`, `render`.)

- [ ] **Step 5: Run new + existing tests**

Run:
```bash
pytest tests/test_board_layout.py tests/test_playback.py tests/test_serializers_golden.py -v
```
Expected: all PASS (new module works; replay output unchanged).

- [ ] **Step 6: Commit**

```bash
git add mcts_study/catan_mcts/web/__init__.py mcts_study/catan_mcts/web/board_layout.py mcts_study/catan_mcts/playback.py mcts_study/tests/test_board_layout.py
git commit -m "refactor(playback): extract board geometry+PNG to web/board_layout"
```

### Task 3: Extract per-state serialization into `web/serializers.py`

**Files:**
- Create: `mcts_study/catan_mcts/web/serializers.py`
- Modify: `mcts_study/catan_mcts/playback.py`
- Test: `mcts_study/tests/test_serializers.py`

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_serializers.py`:

```python
"""Tests for the extracted engine-state serializer."""
from __future__ import annotations

import json


def test_serialize_state_has_full_field_set():
    from catan_bot import _engine
    from catan_mcts.web import serializers
    eng = _engine.Engine(4242)
    st = serializers.serialize_state(eng, narration="(initial)")
    required = {"n", "cp", "phase", "s", "c", "r", "rh", "vp", "hands",
                "bank", "dev_held", "ports", "lr_len", "knights", "built",
                "lr_holder", "la_holder", "vp_played"}
    assert required.issubset(st.keys())
    json.dumps(st)  # must be JSON-serializable


def test_action_desc_blocks():
    from catan_mcts.web import serializers
    assert "BuildSettlement" in serializers.action_desc(0)
    assert "ProposeTrade" in serializers.action_desc(260)
    assert serializers.action_desc(204) == "EndTurn"
```

- [ ] **Step 2: Run to confirm it fails**

Run:
```bash
pytest tests/test_serializers.py -v
```
Expected: FAIL — `serializers` has no `serialize_state` / `action_desc`.

- [ ] **Step 3: Create `serializers.py`**

Create `mcts_study/catan_mcts/web/serializers.py`. **Move** `_action_desc` and all the `SCALAR_*` constants + `DEV_CARD_NAMES`, `PORT_NAMES`, `PHASE_NAMES`, `MAX_SETTLEMENTS/CITIES/ROADS` out of `playback.py` into here. Then refactor the **inner `snapshot()` body** of `playback._replay_to_states` into a standalone `serialize_state(eng, narration)` that takes an engine and returns the per-state dict:

```python
from __future__ import annotations
import numpy as np

# ... [moved SCALAR_* constants, name tables, MAX_* constants] ...


def action_desc(a: int) -> str:
    # ... [moved body of playback._action_desc, verbatim] ...


def serialize_state(eng, narration: str) -> dict:
    """Snapshot one engine state into the client/replay state dict.

    Extracted verbatim from playback._replay_to_states' inner snapshot().
    `eng` is a catan_bot._engine.Engine; `narration` is the last-action label.
    """
    cp = -1 if eng.is_terminal() else int(eng.current_player())
    obs = eng.observation()
    hfeat = obs["hex_features"]
    obs_abs = eng.observation_for(0)
    hands_arr = eng.all_hands()
    bank = list(map(int, eng.bank()))
    vfeat_abs = obs_abs["vertex_features"]
    efeat_abs = obs_abs["edge_features"]
    # ... [rest of the snapshot body from playback.py:382-500, returning the dict] ...
    return {
        "n": narration, "cp": cp, "phase": phase_name,
        "s": settlements, "c": cities, "r": roads, "rh": robber_hex,
        "vp": vps, "hands": hands_breakdown, "bank": bank,
        "dev_held": [pp["dev_held"] for pp in per_player],
        "ports": [pp["ports"] for pp in per_player],
        "lr_len": lr_len, "knights": knights,
        "built": [{"settle": settle_built[p], "city": city_built[p], "road": road_built[p]} for p in range(4)],
        "lr_holder": lr_holder, "la_holder": la_holder, "vp_played": vp_played,
    }
```

> Implementer note: copy the snapshot body from `playback.py` lines ~372-500 exactly (the local-variable computation of `settlements`, `cities`, `roads`, `robber_hex`, `vps`, `per_player`, `lr_len`, `knights`, `*_built`, `lr_holder`, `la_holder`, `vp_played`, `phase_name`, `hands_breakdown`). Only the framing changes (function args + return), not the logic.

- [ ] **Step 4: Rewire `playback.py` to use the serializer**

In `playback.py`: delete the moved `_action_desc` + constants; add `from catan_mcts.web.serializers import action_desc as _action_desc, serialize_state`. Rewrite `_replay_to_states`'s `snapshot(narration)` inner function to delegate:

```python
    def snapshot(narration: str):
        states.append(serialize_state(eng, narration))
```
(Keep the surrounding history-walking loop in `_replay_to_states` unchanged.)

- [ ] **Step 5: Run all Phase-1 tests**

Run:
```bash
pytest tests/test_serializers.py tests/test_serializers_golden.py tests/test_playback.py tests/test_board_layout.py -v
```
Expected: all PASS. The golden test proves the extraction preserved replay output.

- [ ] **Step 6: Commit**

```bash
git add mcts_study/catan_mcts/web/serializers.py mcts_study/catan_mcts/playback.py mcts_study/tests/test_serializers.py
git commit -m "refactor(playback): extract per-state serializer to web/serializers"
```

---

## Phase 2: Bot registry

### Task 4: Bot type listing + non-GNN bot construction

**Files:**
- Create: `mcts_study/catan_mcts/web/bot_registry.py`
- Test: `mcts_study/tests/test_bot_registry.py`

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_bot_registry.py`:

```python
"""Tests for bot discovery + construction."""
from __future__ import annotations

import pytest


def test_list_types_includes_core_bots():
    from catan_mcts.web import bot_registry
    types = {t["id"] for t in bot_registry.list_types()}
    assert {"Random", "Greedy", "LookaheadMctsV3", "PureGnn", "GnnMcts"} <= types


def test_build_random_bot():
    from catan_mcts.web import bot_registry
    from catan_mcts.adapter import CatanGame
    game = CatanGame()
    bot = bot_registry.build({"type": "Random"}, game=game, seed=7)
    state = game.new_initial_state(seed=7)
    action = bot.step(state)
    assert action in state.legal_actions()


def test_build_greedy_bot():
    from catan_mcts.web import bot_registry
    from catan_mcts.adapter import CatanGame
    game = CatanGame()
    bot = bot_registry.build({"type": "Greedy"}, game=game, seed=1)
    state = game.new_initial_state(seed=1)
    assert bot.step(state) in state.legal_actions()


def test_build_unknown_type_raises():
    from catan_mcts.web import bot_registry
    from catan_mcts.adapter import CatanGame
    with pytest.raises(ValueError, match="unknown bot type"):
        bot_registry.build({"type": "Nope"}, game=CatanGame(), seed=0)
```

- [ ] **Step 2: Run to confirm it fails**

Run:
```bash
pytest tests/test_bot_registry.py -v
```
Expected: FAIL — `ModuleNotFoundError` / missing functions.

- [ ] **Step 3: Implement `bot_registry.py` (non-GNN parts)**

Create `mcts_study/catan_mcts/web/bot_registry.py`:

```python
"""Discover available bot types + checkpoints, and build bot instances.

A "bot" here is anything with a `.step(state) -> int` method. We reuse the
existing bot classes; GNN types are loaded lazily so importing this module
never forces torch.
"""
from __future__ import annotations

import random
from pathlib import Path


class _RandomBot:
    """Picks a uniformly-random legal action."""
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def step(self, state) -> int:
        legal = state.legal_actions()
        if not legal:
            raise RuntimeError("_RandomBot: no legal actions")
        return self._rng.choice(legal)


def list_types() -> list[dict]:
    """Bot types selectable in the lobby. `needs_checkpoint` drives the UI."""
    return [
        {"id": "Random", "label": "Random", "needs_checkpoint": False},
        {"id": "Greedy", "label": "Greedy baseline", "needs_checkpoint": False},
        {"id": "LookaheadMctsV3", "label": "Lookahead MCTS v3", "needs_checkpoint": False},
        {"id": "PureGnn", "label": "Pure GNN", "needs_checkpoint": True},
        {"id": "GnnMcts", "label": "GNN + MCTS", "needs_checkpoint": True},
    ]


def build(spec: dict, *, game, seed: int):
    """Construct a bot from a spec like {"type": "Random"} or
    {"type": "PureGnn", "checkpoint": "/abs/path.pt"}.

    `game` is a CatanGame (needed by MCTS/GNN bots); `seed` seeds the bot.
    """
    t = spec.get("type")
    if t == "Random":
        return _RandomBot(seed=seed)
    if t == "Greedy":
        from catan_mcts.bots import GreedyBaselineBot
        return GreedyBaselineBot(seed=seed)
    if t == "LookaheadMctsV3":
        from catan_mcts.players_v3 import build_lookahead_mcts_v3
        return build_lookahead_mcts_v3(game, seed=seed)
    if t in ("PureGnn", "GnnMcts"):
        return _build_gnn_bot(spec, game=game, seed=seed)
    raise ValueError(f"unknown bot type: {t!r}")


def _build_gnn_bot(spec, *, game, seed):  # implemented in Task 5
    raise NotImplementedError("GNN bot construction lands in Task 5")
```

- [ ] **Step 4: Run tests**

Run:
```bash
pytest tests/test_bot_registry.py::test_list_types_includes_core_bots tests/test_bot_registry.py::test_build_random_bot tests/test_bot_registry.py::test_build_greedy_bot tests/test_bot_registry.py::test_build_unknown_type_raises -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/web/bot_registry.py mcts_study/tests/test_bot_registry.py
git commit -m "feat(web): bot registry — type listing + non-GNN construction"
```

### Task 5: Checkpoint discovery + GNN bot construction

**Files:**
- Modify: `mcts_study/catan_mcts/web/bot_registry.py`
- Test: `mcts_study/tests/test_bot_registry.py`

- [ ] **Step 1: Write the failing tests**

Append to `mcts_study/tests/test_bot_registry.py`:

```python
def test_list_checkpoints_scans_dir(tmp_path):
    from catan_mcts.web import bot_registry
    (tmp_path / "a.pt").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.pt").write_bytes(b"y")
    (tmp_path / "notes.txt").write_text("ignore me")
    cps = bot_registry.list_checkpoints(tmp_path)
    names = {c["name"] for c in cps}
    assert "a.pt" in names and "b.pt" in names
    assert all(c["path"].endswith(".pt") for c in cps)
    assert not any(c["name"] == "notes.txt" for c in cps)


def test_build_gnn_bad_checkpoint_raises(tmp_path):
    from catan_mcts.web import bot_registry
    from catan_mcts.adapter import CatanGame
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a torch checkpoint")
    with pytest.raises(ValueError, match="checkpoint"):
        bot_registry.build(
            {"type": "PureGnn", "checkpoint": str(bad)},
            game=CatanGame(), seed=0,
        )


def test_build_gnn_missing_checkpoint_raises(tmp_path):
    from catan_mcts.web import bot_registry
    from catan_mcts.adapter import CatanGame
    with pytest.raises(ValueError, match="checkpoint"):
        bot_registry.build(
            {"type": "PureGnn", "checkpoint": str(tmp_path / "nope.pt")},
            game=CatanGame(), seed=0,
        )
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
pytest tests/test_bot_registry.py::test_list_checkpoints_scans_dir tests/test_bot_registry.py::test_build_gnn_bad_checkpoint_raises tests/test_bot_registry.py::test_build_gnn_missing_checkpoint_raises -v
```
Expected: FAIL — `list_checkpoints` missing; `_build_gnn_bot` raises `NotImplementedError`, not `ValueError`.

- [ ] **Step 3: Implement checkpoint discovery + GNN loading**

In `bot_registry.py`, add `list_checkpoints` and a real `_build_gnn_bot` (replacing the stub). Mirror the proven loader from `experiments/e10e_gnn_mcts.py:57-82`:

```python
def list_checkpoints(checkpoints_dir) -> list[dict]:
    """Recursively list *.pt files under `checkpoints_dir` (sorted by name)."""
    root = Path(checkpoints_dir)
    if not root.exists():
        return []
    out = []
    for p in sorted(root.rglob("*.pt")):
        out.append({"name": p.name, "path": str(p)})
    return out


def _load_gnn_model(checkpoint: str, *, hidden_dim: int, num_layers: int, device: str):
    """Load a GnnModel from a .pt checkpoint (handles {'model_state': ...} wrappers)."""
    from pathlib import Path as _P
    if not _P(checkpoint).exists():
        raise ValueError(f"checkpoint not found: {checkpoint}")
    import torch
    from catan_gnn.gnn_model import GnnModel
    try:
        obj = torch.load(checkpoint, map_location=device, weights_only=False)
        state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
        model = GnnModel(hidden_dim=hidden_dim, num_layers=num_layers)
        model.load_state_dict(state)
    except Exception as e:  # bad/corrupt file or shape mismatch
        raise ValueError(f"failed to load checkpoint {checkpoint!r}: {e}") from e
    return model.to(device).eval()


def _build_gnn_bot(spec, *, game, seed):
    checkpoint = spec.get("checkpoint")
    if not checkpoint:
        raise ValueError("GNN bot requires a 'checkpoint' path")
    device = spec.get("device", "cpu")
    hidden_dim = int(spec.get("hidden_dim", 32))
    num_layers = int(spec.get("num_layers", 2))
    model = _load_gnn_model(checkpoint, hidden_dim=hidden_dim,
                            num_layers=num_layers, device=device)
    if spec["type"] == "PureGnn":
        from catan_mcts.bots_gnn import PureGnnBot
        return PureGnnBot(model=model, device=device)
    # GnnMcts
    from catan_mcts.experiments.e10e_gnn_mcts import build_gnn_mcts_bot
    sims = int(spec.get("sims", 200))
    return build_gnn_mcts_bot(game, model, sims=sims, seed=seed, device=device)
```

- [ ] **Step 4: Run the new tests**

Run:
```bash
pytest tests/test_bot_registry.py -v
```
Expected: all PASS (the two GNN tests assert the `ValueError` path; they don't need a real checkpoint or torch model load to succeed because both hit the validation branches early).

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/web/bot_registry.py mcts_study/tests/test_bot_registry.py
git commit -m "feat(web): checkpoint discovery + GNN bot construction with validation"
```

---

## Phase 3: Action decoding

### Task 6: Decode raw action ints into UI action objects

**Files:**
- Create: `mcts_study/catan_mcts/web/action_decode.py`
- Test: `mcts_study/tests/test_action_decode.py`

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_action_decode.py`:

```python
"""Tests for action int -> UI object decoding."""
from __future__ import annotations


def test_decode_settlement():
    from catan_mcts.web import action_decode
    d = action_decode.decode(12)
    assert d["id"] == 12
    assert d["kind"] == "build_settlement"
    assert d["target"] == 12
    assert "Settlement" in d["label"]


def test_decode_road_target_is_edge():
    from catan_mcts.web import action_decode
    d = action_decode.decode(108)  # first road
    assert d["kind"] == "build_road"
    assert d["target"] == 0


def test_decode_move_robber_target_is_hex():
    from catan_mcts.web import action_decode
    d = action_decode.decode(180)
    assert d["kind"] == "move_robber"
    assert d["target"] == 0


def test_decode_non_spatial_has_null_target():
    from catan_mcts.web import action_decode
    for a, kind in [(205, "roll"), (204, "end_turn"), (226, "buy_dev"),
                    (206, "trade_bank"), (260, "propose_trade"), (227, "play_dev")]:
        d = action_decode.decode(a)
        assert d["kind"] == kind, (a, d)
        assert d["target"] is None


def test_decode_many():
    from catan_mcts.web import action_decode
    out = action_decode.decode_many([0, 108, 204, 205])
    assert [o["id"] for o in out] == [0, 108, 204, 205]
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
pytest tests/test_action_decode.py -v
```
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `action_decode.py`**

Create `mcts_study/catan_mcts/web/action_decode.py`. Reuse `serializers.action_desc` for the label; map ranges to `kind` + spatial `target` per the action space in `bots.py` / `playback._action_desc`:

```python
"""Decode raw engine action ids into UI-friendly objects.

Action space (v2, 280 actions):
  0..53    BuildSettlement(v)   target = vertex
  54..107  BuildCity(v)         target = vertex (v = a-54)
  108..179 BuildRoad(e)         target = edge   (e = a-108)
  180..198 MoveRobber(h)        target = hex    (h = a-180)
  199..203 Discard(res)         non-spatial
  204      EndTurn              non-spatial
  205      RollDice             non-spatial
  206..225 TradeBank            non-spatial
  226      BuyDevCard           non-spatial
  227      PlayKnight           non-spatial
  228      PlayRoadBuilding     non-spatial
  229..233 PlayMonopoly         non-spatial
  234..258 PlayYearOfPlenty     non-spatial
  259      PlayVpCard           non-spatial
  260..279 ProposeTrade         non-spatial
"""
from __future__ import annotations

from catan_mcts.web.serializers import action_desc


def _kind_and_target(a: int):
    if 0 <= a < 54:    return "build_settlement", a
    if 54 <= a < 108:  return "build_city", a - 54
    if 108 <= a < 180: return "build_road", a - 108
    if 180 <= a < 199: return "move_robber", a - 180
    if 199 <= a < 204: return "discard", None
    if a == 204:       return "end_turn", None
    if a == 205:       return "roll", None
    if 206 <= a < 226: return "trade_bank", None
    if a == 226:       return "buy_dev", None
    if a in (227, 228) or 229 <= a < 234 or 234 <= a < 259 or a == 259:
        return "play_dev", None
    if 260 <= a < 280: return "propose_trade", None
    return "unknown", None


def decode(a: int) -> dict:
    a = int(a)
    kind, target = _kind_and_target(a)
    return {"id": a, "label": action_desc(a), "kind": kind, "target": target}


def decode_many(actions) -> list[dict]:
    return [decode(int(a)) for a in actions]
```

- [ ] **Step 4: Run tests**

Run:
```bash
pytest tests/test_action_decode.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/web/action_decode.py mcts_study/tests/test_action_decode.py
git commit -m "feat(web): action_decode — raw ids to {id,label,kind,target}"
```

---

## Phase 4: GameSession (core)

### Task 7: Trade-match prediction helper

**Files:**
- Create: `mcts_study/catan_mcts/web/trade_logic.py`
- Test: `mcts_study/tests/test_trade_logic.py`

This replicates the engine's seat-order auto-match (rules.rs:347-374) so the session knows, before applying, whether a bot's `ProposeTrade` would swap with the human.

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_trade_logic.py`:

```python
"""Tests for engine-faithful trade-match prediction."""
from __future__ import annotations


def test_decode_propose_trade_give_get():
    from catan_mcts.web import trade_logic
    # 260 = first ProposeTrade: give=Wood(0), get=first-other=Brick(1)
    give, get = trade_logic.decode_propose_trade(260)
    assert give == 0 and get == 1


def test_first_acceptor_seat_order():
    from catan_mcts.web import trade_logic
    # current player 0 gives wood(0) wants brick(1). Hands [4x5].
    hands = [
        [1, 0, 0, 0, 0],  # P0 proposer
        [0, 0, 0, 0, 0],  # P1 can't accept
        [0, 1, 0, 0, 0],  # P2 has brick -> first acceptor
        [0, 1, 0, 0, 0],  # P3 also has brick (not reached)
    ]
    acceptor = trade_logic.first_acceptor(current_player=0, give=0, get=1, hands=hands)
    assert acceptor == 2


def test_no_acceptor_returns_minus_one():
    from catan_mcts.web import trade_logic
    hands = [[1, 0, 0, 0, 0]] + [[0, 0, 0, 0, 0]] * 3
    assert trade_logic.first_acceptor(0, 0, 1, hands) == -1


def test_would_match_human():
    from catan_mcts.web import trade_logic
    hands = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    # P0 proposes; P1 (the human) is first acceptor.
    assert trade_logic.would_match_human(current_player=0, action=260,
                                          hands=hands, human_seat=1) is True
    assert trade_logic.would_match_human(current_player=0, action=260,
                                          hands=hands, human_seat=2) is False
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
pytest tests/test_trade_logic.py -v
```
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `trade_logic.py`**

Create `mcts_study/catan_mcts/web/trade_logic.py`. The ProposeTrade encoding matches `playback._action_desc` (give = idx//4; get = the (idx%4)-th of the other four resources):

```python
"""Engine-faithful prediction of who would accept a ProposeTrade.

Mirrors catan_engine/src/rules.rs:347-374: scan opponents in seat order
(current+1,+2,+3); the first holding >=1 of `get` accepts a 1-for-1 swap.
"""
from __future__ import annotations

PROPOSE_TRADE_BASE = 260


def decode_propose_trade(action: int) -> tuple[int, int]:
    """action 260..279 -> (give_resource_idx, get_resource_idx)."""
    idx = int(action) - PROPOSE_TRADE_BASE
    if not (0 <= idx < 20):
        raise ValueError(f"not a ProposeTrade action: {action}")
    give = idx // 4
    get_in_others = idx % 4
    others = [r for r in range(5) if r != give]
    return give, others[get_in_others]


def first_acceptor(current_player: int, give: int, get: int, hands) -> int:
    """Seat of the first opponent (current+1,+2,+3) holding >=1 of `get`, else -1."""
    for offset in range(1, 4):
        opp = (current_player + offset) % 4
        if hands[opp][get] >= 1:
            return opp
    return -1


def would_match_human(current_player: int, action: int, hands, human_seat: int) -> bool:
    """True iff the engine would auto-match the human for this ProposeTrade."""
    give, get = decode_propose_trade(action)
    return first_acceptor(current_player, give, get, hands) == human_seat
```

- [ ] **Step 4: Run tests**

Run:
```bash
pytest tests/test_trade_logic.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/web/trade_logic.py mcts_study/tests/test_trade_logic.py
git commit -m "feat(web): trade_logic — engine-faithful trade-match prediction"
```

### Task 8: GameSession skeleton — construction + state_json

**Files:**
- Create: `mcts_study/catan_mcts/web/game_session.py`
- Test: `mcts_study/tests/test_game_session.py`

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_game_session.py`:

```python
"""Tests for the live GameSession."""
from __future__ import annotations

import pytest

from catan_mcts.web.game_session import GameSession


def _setup(human_seat=0):
    return {
        "human_seat": human_seat,
        "seats": {str(s): {"type": "Random"} for s in range(4) if s != human_seat},
        "rules": {"vp_target": 10, "bonuses": True},
        "seed": 4242,
    }


def test_construct_and_state_json():
    sess = GameSession(_setup())
    s = sess.state_json()
    assert s["human_seat"] == 0
    assert s["status"] in {"your_turn", "bot_thinking", "trade_offer", "game_over"}
    assert "state" in s and "seat_names" in s
    assert len(s["seat_names"]) == 4


def test_board_payload_present():
    sess = GameSession(_setup())
    board = sess.board_payload()
    assert "layout" in board and "png_b64" in board
    assert board["png_b64"]  # non-empty base64
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
pytest tests/test_game_session.py::test_construct_and_state_json tests/test_game_session.py::test_board_payload_present -v
```
Expected: FAIL — module/class missing.

- [ ] **Step 3: Implement the skeleton**

Create `mcts_study/catan_mcts/web/game_session.py`:

```python
"""One live interactive game: engine + bots + cooperative driving loop.

The session owns a CatanState and three bots, drives chance + bot turns,
and yields control to the human at their turn or when a bot's ProposeTrade
would auto-match the human (the trade-intercept; see advance()).
"""
from __future__ import annotations

import base64
import random
import tempfile
from pathlib import Path

from catan_mcts.adapter import CatanGame
from catan_mcts.web import bot_registry, board_layout, serializers, action_decode, trade_logic

EVENT_SEED_OFFSET = 0


class GameSession:
    def __init__(self, setup: dict) -> None:
        self.human_seat = int(setup["human_seat"])
        rules = setup.get("rules", {})
        self._vp_target = int(rules.get("vp_target", 10))
        self._bonuses = bool(rules.get("bonuses", True))
        self.seed = int(setup.get("seed") if setup.get("seed") is not None
                        else random.Random().randint(1, 2**31 - 1))
        self._game = CatanGame(vp_target=self._vp_target, bonuses=self._bonuses)
        self._state = self._game.new_initial_state(seed=self.seed)
        self._rng = random.Random(self.seed ^ 0x5EED)
        # Build the three bots (one per non-human seat).
        self._bots: dict[int, object] = {}
        self._seat_specs: dict[int, dict] = {}
        for seat_str, spec in setup["seats"].items():
            seat = int(seat_str)
            self._bots[seat] = bot_registry.build(spec, game=self._game, seed=self.seed + seat)
            self._seat_specs[seat] = spec
        self._pending_trade = None   # (proposer_seat, action_id) when paused on a trade
        self._last_narration = "(game start)"
        self._error = None

    # ---- public read API -------------------------------------------------
    def seat_names(self) -> list[str]:
        names = []
        for s in range(4):
            if s == self.human_seat:
                names.append("You")
            else:
                names.append(f"P{s} {self._seat_specs[s]['type']}")
        return names

    def board_payload(self) -> dict:
        vertex_xy, _, _ = board_layout.build_layout()
        with tempfile.TemporaryDirectory() as td:
            png = Path(td) / "board.png"
            board_layout.render_board_png(self.seed, png, vertex_xy=vertex_xy)
            b64 = base64.b64encode(png.read_bytes()).decode("ascii")
        return {"layout": board_layout.layout_dict(), "png_b64": b64}

    def _status(self) -> str:
        if self._error is not None:
            return "error"
        if self._state.is_terminal():
            return "game_over"
        if self._pending_trade is not None:
            return "trade_offer"
        if int(self._state.current_player()) == self.human_seat:
            return "your_turn"
        return "bot_thinking"

    def state_json(self) -> dict:
        eng = self._state._engine
        status = self._status()
        out = {
            "status": status,
            "human_seat": self.human_seat,
            "current_player": -1 if eng.is_terminal() else int(eng.current_player()),
            "phase": None,
            "narration": self._last_narration,
            "state": serializers.serialize_state(eng, self._last_narration),
            "seat_names": self.seat_names(),
        }
        out["phase"] = out["state"]["phase"]
        if status == "your_turn":
            out["legal_actions"] = action_decode.decode_many(self._state.legal_actions())
        if status == "trade_offer":
            out["trade_offer"] = self._trade_offer_payload()
        if status == "game_over":
            out["returns"] = self._state.returns()
        if status == "error":
            out["error"] = str(self._error)
        return out

    def _trade_offer_payload(self) -> dict:
        proposer, action = self._pending_trade
        give, get = trade_logic.decode_propose_trade(action)
        # Human gives the bot's requested resource (get), receives the offered (give).
        return {"from_seat": proposer, "you_give": [get, 1], "you_get": [give, 1]}
```

- [ ] **Step 4: Run tests**

Run:
```bash
pytest tests/test_game_session.py::test_construct_and_state_json tests/test_game_session.py::test_board_payload_present -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/web/game_session.py mcts_study/tests/test_game_session.py
git commit -m "feat(web): GameSession skeleton — construction, state_json, board payload"
```

### Task 9: GameSession.advance() — chance + bot driving (no trade intercept yet)

**Files:**
- Modify: `mcts_study/catan_mcts/web/game_session.py`
- Test: `mcts_study/tests/test_game_session.py`

- [ ] **Step 1: Write the failing test**

Append to `mcts_study/tests/test_game_session.py`:

```python
def test_advance_reaches_human_turn_or_terminal():
    sess = GameSession(_setup(human_seat=0))
    res = sess.advance()
    # Either it's the human's turn (with legal actions) or the game ended.
    assert res["status"] in {"your_turn", "game_over"}
    if res["status"] == "your_turn":
        assert len(res["legal_actions"]) >= 1
        assert int(sess.state_json()["current_player"]) == 0


def test_full_game_with_stub_human_terminates():
    """Drive a whole game: human always plays its first legal action."""
    sess = GameSession(_setup(human_seat=0))
    for _ in range(100000):
        res = sess.advance()
        if res["status"] == "game_over":
            break
        if res["status"] == "your_turn":
            sess.apply_human_action(res["legal_actions"][0]["id"])
        elif res["status"] == "trade_offer":
            sess.respond_to_trade(accept=False)
        else:
            raise AssertionError(f"unexpected status {res['status']}")
    assert sess.state_json()["status"] == "game_over"
    assert sess.state_json()["returns"] is not None
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
pytest tests/test_game_session.py::test_advance_reaches_human_turn_or_terminal -v
```
Expected: FAIL — `advance` / `apply_human_action` not defined.

- [ ] **Step 3: Implement advance() + apply_human_action() (trade intercept stubbed to apply normally)**

Add to `GameSession` in `game_session.py`:

```python
    # ---- driving loop ----------------------------------------------------
    def _sample_chance(self) -> int:
        outcomes = self._state.chance_outcomes()
        r = self._rng.random()
        cum = 0.0
        for v, p in outcomes:
            cum += p
            if r <= cum:
                return int(v)
        return int(outcomes[-1][0])

    def advance(self, max_steps: int = 100000) -> dict:
        """Run chance + bot turns until human turn / trade offer / terminal."""
        steps = 0
        while steps < max_steps:
            if self._error is not None:
                return self.state_json()
            if self._state.is_terminal():
                return self.state_json()
            if self._state.is_chance_node():
                self._state.apply_action(self._sample_chance())
                steps += 1
                continue
            cp = int(self._state.current_player())
            if cp == self.human_seat:
                return self.state_json()
            legal = self._state.legal_actions()
            if len(legal) == 1:
                self._apply_and_narrate(int(legal[0]), cp)
                steps += 1
                continue
            try:
                action = int(self._bots[cp].step(self._state))
            except Exception as e:  # bot crashed
                self._error = f"bot P{cp} errored: {e}"
                return self.state_json()
            # Trade intercept lands in Task 10; for now apply normally.
            self._apply_and_narrate(action, cp)
            steps += 1
        return self.state_json()

    def _apply_and_narrate(self, action: int, player: int) -> None:
        self._last_narration = f"P{player} {serializers.action_desc(action)}"
        self._state.apply_action(int(action))

    def apply_human_action(self, action: int) -> dict:
        if int(self._state.current_player()) != self.human_seat:
            raise ValueError("not your turn")
        legal = self._state.legal_actions()
        if int(action) not in legal:
            raise ValueError(f"illegal action {action}")
        self._apply_and_narrate(int(action), self.human_seat)
        return self.advance()

    def respond_to_trade(self, accept: bool) -> dict:
        # Full implementation in Task 10; placeholder so the stub game loop runs.
        self._pending_trade = None
        return self.advance()
```

- [ ] **Step 4: Run the advance + full-game tests**

Run:
```bash
pytest tests/test_game_session.py::test_advance_reaches_human_turn_or_terminal tests/test_game_session.py::test_full_game_with_stub_human_terminates -v
```
Expected: PASS (the full game terminates; trade intercept not yet exercised because it's stubbed off).

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/web/game_session.py mcts_study/tests/test_game_session.py
git commit -m "feat(web): GameSession.advance + apply_human_action driving loop"
```

### Task 10: Trade intercept + Accept/Reject

**Files:**
- Modify: `mcts_study/catan_mcts/web/game_session.py`
- Test: `mcts_study/tests/test_game_session.py`

- [ ] **Step 1: Write the failing tests**

Append to `mcts_study/tests/test_game_session.py`:

```python
def _trade_session(human_seat=1):
    """Session positioned so we can force a ProposeTrade targeting the human."""
    return GameSession(_setup(human_seat=human_seat))


def test_intercept_pauses_when_trade_targets_human(monkeypatch):
    from catan_mcts.web import game_session as gs_mod
    sess = _trade_session(human_seat=1)
    # Force engine into Main phase, P0 to move, hands so P0 wood->brick first-matches P1.
    eng = sess._state._engine
    # Drive to a P0 Main decision via advance, then monkeypatch the acting bot
    # to return a ProposeTrade and the hands check to target the human.
    monkeypatch.setattr(sess, "_predict_trade_acceptor",
                        lambda cp, action: sess.human_seat)
    monkeypatch.setattr(sess._bots[0] if 0 in sess._bots else object(),
                        "step", lambda state: 260, raising=False)
    # Put P0 on the clock as a non-human, non-trivial decision:
    # (advance will naturally stop at human; we instead call the intercept directly)
    res = sess._maybe_intercept_trade(current_player=0, action=260)
    assert res is True
    assert sess.state_json()["status"] == "trade_offer"
    offer = sess.state_json()["trade_offer"]
    assert offer["from_seat"] == 0


def test_reject_leaves_human_hand_unchanged(monkeypatch):
    sess = _trade_session(human_seat=1)
    sess._pending_trade = (0, 260)  # P0 offered; human is P1
    before = [list(h) for h in sess._state._engine.all_hands()]
    # Reject: must NOT apply the trade; re-query bot is masked.
    monkeypatch.setattr(sess._bots[0], "step", lambda state: 204)  # bot ends turn
    sess.respond_to_trade(accept=False)
    after = [list(h) for h in sess._state._engine.all_hands()]
    assert after[1] == before[1], "human hand changed on reject"


def test_no_intercept_when_trade_targets_other_bot():
    sess = _trade_session(human_seat=1)
    # acceptor predicted as seat 2 (another bot) -> no pause
    sess._predict_trade_acceptor = lambda cp, action: 2
    assert sess._maybe_intercept_trade(current_player=0, action=260) is False
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
pytest tests/test_game_session.py::test_no_intercept_when_trade_targets_other_bot -v
```
Expected: FAIL — `_maybe_intercept_trade` / `_predict_trade_acceptor` not defined.

- [ ] **Step 3: Implement the intercept**

In `game_session.py`, add helpers and wire them into `advance()` + `respond_to_trade()`:

```python
    def _predict_trade_acceptor(self, current_player: int, action: int) -> int:
        if not (260 <= int(action) < 280):
            return -1
        give, get = trade_logic.decode_propose_trade(action)
        hands = [list(h) for h in self._state._engine.all_hands()]
        return trade_logic.first_acceptor(current_player, give, get, hands)

    def _maybe_intercept_trade(self, current_player: int, action: int) -> bool:
        """If this bot ProposeTrade would auto-match the human, pause. Returns
        True iff intercepted (caller must stop driving and surface trade_offer)."""
        if not (260 <= int(action) < 280):
            return False
        if self._predict_trade_acceptor(current_player, action) == self.human_seat:
            self._pending_trade = (current_player, int(action))
            return True
        return False
```

In `advance()`, replace the "apply normally" comment + line with:

```python
            if self._maybe_intercept_trade(cp, action):
                return self.state_json()
            self._apply_and_narrate(action, cp)
```

Replace the placeholder `respond_to_trade` with:

```python
    def respond_to_trade(self, accept: bool) -> dict:
        if self._pending_trade is None:
            return self.advance()
        proposer, action = self._pending_trade
        self._pending_trade = None
        if accept:
            self._apply_and_narrate(action, proposer)
        else:
            # Reject: do NOT apply (would auto-swap with human). Re-query the
            # bot with this trade masked; fall back to EndTurn (204).
            substitute = self._requery_bot_masked(proposer, masked_action=action)
            self._apply_and_narrate(substitute, proposer)
        return self.advance()

    def _requery_bot_masked(self, seat: int, masked_action: int) -> int:
        """Ask the bot for an action with `masked_action` removed; else EndTurn."""
        legal = [a for a in self._state.legal_actions() if int(a) != int(masked_action)]
        if not legal:
            return 204  # EndTurn
        try:
            a = int(self._bots[seat].step(_MaskedLegalView(self._state, masked_action)))
            if a in legal:
                return a
        except Exception:
            pass
        return 204 if 204 in legal else int(legal[0])
```

Add a tiny masked-state view at module scope (wraps the state so `legal_actions()` hides the rejected trade; everything else delegates):

```python
class _MaskedLegalView:
    """Wraps a CatanState so legal_actions() omits one action; all else delegates."""
    def __init__(self, state, masked_action: int):
        self._state = state
        self._masked = int(masked_action)

    def legal_actions(self):
        return [a for a in self._state.legal_actions() if int(a) != self._masked]

    def __getattr__(self, name):
        return getattr(self._state, name)
```

> Note: GNN/MCTS bots read `state._engine.observation()` and `state.current_player()`; `__getattr__` delegates both, and they ignore the masked single action — acceptable for the rare reject-fallback path. If a re-queried bot returns the masked action anyway, we fall back to EndTurn.

- [ ] **Step 4: Run the trade tests + the full Phase-4 suite**

Run:
```bash
pytest tests/test_game_session.py tests/test_trade_logic.py -v
```
Expected: all PASS (intercept pauses on human-targeted trades; reject leaves the human hand unchanged; non-human trades pass through; full game still terminates).

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/web/game_session.py mcts_study/tests/test_game_session.py
git commit -m "feat(web): GameSession trade intercept + accept/reject"
```

### Task 11: Threaded advance for slow bots (background run)

**Files:**
- Modify: `mcts_study/catan_mcts/web/game_session.py`
- Test: `mcts_study/tests/test_game_session.py`

- [ ] **Step 1: Write the failing test**

Append to `mcts_study/tests/test_game_session.py`:

```python
import time


def test_advance_async_runs_in_background():
    sess = GameSession(_setup(human_seat=0))
    sess.advance_async()
    # Poll until the worker resolves to a yield point.
    for _ in range(200):
        if not sess.is_advancing():
            break
        time.sleep(0.02)
    assert not sess.is_advancing()
    assert sess.state_json()["status"] in {"your_turn", "game_over"}
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
pytest tests/test_game_session.py::test_advance_async_runs_in_background -v
```
Expected: FAIL — `advance_async` / `is_advancing` not defined.

- [ ] **Step 3: Implement threaded advance**

Add to `GameSession` (import `threading` at top):

```python
    # ---- async driving (for slow GNN/MCTS bots) --------------------------
    def advance_async(self) -> None:
        """Run advance() in a daemon thread; poll is_advancing()/state_json()."""
        import threading
        if getattr(self, "_thread", None) is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.advance, daemon=True)
        self._thread.start()

    def is_advancing(self) -> bool:
        t = getattr(self, "_thread", None)
        return bool(t is not None and t.is_alive())
```

- [ ] **Step 4: Run test**

Run:
```bash
pytest tests/test_game_session.py::test_advance_async_runs_in_background -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/web/game_session.py mcts_study/tests/test_game_session.py
git commit -m "feat(web): GameSession.advance_async for slow bots"
```

---

## Phase 5: FastAPI server

### Task 12: App factory + `/api/bots`

**Files:**
- Create: `mcts_study/catan_mcts/web/server.py`
- Test: `mcts_study/tests/test_web_api.py`

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_web_api.py`:

```python
"""FastAPI endpoint contract tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    from catan_mcts.web.server import create_app
    app = create_app(checkpoints_dir=tmp_path, replays_dir=tmp_path)
    return TestClient(app)


def test_bots_endpoint(client):
    r = client.get("/api/bots")
    assert r.status_code == 200
    body = r.json()
    ids = {t["id"] for t in body["types"]}
    assert {"Random", "PureGnn"} <= ids
    assert "checkpoints" in body
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
pytest tests/test_web_api.py::test_bots_endpoint -v
```
Expected: FAIL — `create_app` missing.

- [ ] **Step 3: Implement the app factory + `/api/bots`**

Create `mcts_study/catan_mcts/web/server.py`:

```python
"""FastAPI app: serves the play-vs-bots frontend + REST/SSE API.

create_app(checkpoints_dir, replays_dir) returns a configured app. Paths are
parameters (no hardcoded WSL paths) so the same code runs locally or deployed.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from catan_mcts.web import bot_registry
from catan_mcts.web.game_session import GameSession

_STATIC = Path(__file__).parent / "static"


class SetupSpec(BaseModel):
    human_seat: int
    seats: dict
    rules: dict | None = None
    seed: int | None = None


def create_app(*, checkpoints_dir, replays_dir) -> FastAPI:
    app = FastAPI(title="Catan Play-vs-Bots")
    checkpoints_dir = Path(checkpoints_dir)
    replays_dir = Path(replays_dir)
    games: dict[str, GameSession] = {}

    @app.get("/api/bots")
    def get_bots():
        return {
            "types": bot_registry.list_types(),
            "checkpoints": bot_registry.list_checkpoints(checkpoints_dir),
        }

    app.state.games = games
    app.state.checkpoints_dir = checkpoints_dir
    app.state.replays_dir = replays_dir
    return app
```

- [ ] **Step 4: Run test**

Run:
```bash
pytest tests/test_web_api.py::test_bots_endpoint -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/web/server.py mcts_study/tests/test_web_api.py
git commit -m "feat(web): FastAPI app factory + /api/bots"
```

### Task 13: Game lifecycle endpoints (create / state / action / trade-response)

**Files:**
- Modify: `mcts_study/catan_mcts/web/server.py`
- Test: `mcts_study/tests/test_web_api.py`

- [ ] **Step 1: Write the failing test**

Append to `mcts_study/tests/test_web_api.py`:

```python
def _all_random_setup(human_seat=0):
    return {
        "human_seat": human_seat,
        "seats": {str(s): {"type": "Random"} for s in range(4) if s != human_seat},
        "rules": {"vp_target": 10, "bonuses": True},
        "seed": 4242,
    }


def test_create_and_play_to_terminal(client):
    r = client.post("/api/games", json=_all_random_setup())
    assert r.status_code == 200
    body = r.json()
    gid = body["game_id"]
    assert "board" in body and body["board"]["png_b64"]
    state = body["state"]
    # Drive: keep applying the first legal action / rejecting trades.
    for _ in range(100000):
        if state["status"] == "game_over":
            break
        if state["status"] == "your_turn":
            aid = state["legal_actions"][0]["id"]
            state = client.post(f"/api/games/{gid}/action", json={"action": aid}).json()
        elif state["status"] == "trade_offer":
            state = client.post(f"/api/games/{gid}/trade-response", json={"accept": False}).json()
        else:
            state = client.get(f"/api/games/{gid}/state").json()
    assert state["status"] == "game_over"
    assert state["returns"] is not None


def test_illegal_action_returns_409(client):
    gid = client.post("/api/games", json=_all_random_setup()).json()["game_id"]
    # 9999 is never a legal action id.
    r = client.post(f"/api/games/{gid}/action", json={"action": 9999})
    assert r.status_code == 409


def test_unknown_game_404(client):
    r = client.get("/api/games/does-not-exist/state")
    assert r.status_code == 404
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
pytest tests/test_web_api.py::test_unknown_game_404 -v
```
Expected: FAIL — endpoints missing → 404 from a different cause / route absent.

- [ ] **Step 3: Implement the lifecycle endpoints**

Inside `create_app` (before `return app`), add:

```python
    class ActionBody(BaseModel):
        action: int

    class TradeBody(BaseModel):
        accept: bool

    def _get(gid: str) -> GameSession:
        sess = games.get(gid)
        if sess is None:
            raise HTTPException(status_code=404, detail="game not found")
        return sess

    @app.post("/api/games")
    def create_game(spec: SetupSpec):
        try:
            sess = GameSession(spec.model_dump())
        except ValueError as e:   # bad checkpoint / bad spec
            raise HTTPException(status_code=400, detail=str(e))
        gid = uuid.uuid4().hex[:12]
        games[gid] = sess
        state = sess.advance()    # run any opening bot turns up to the human
        return {"game_id": gid, "board": sess.board_payload(), "state": state}

    @app.get("/api/games/{gid}/state")
    def get_state(gid: str):
        return _get(gid).state_json()

    @app.post("/api/games/{gid}/action")
    def post_action(gid: str, body: ActionBody):
        sess = _get(gid)
        try:
            return sess.apply_human_action(body.action)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.post("/api/games/{gid}/trade-response")
    def post_trade(gid: str, body: TradeBody):
        return _get(gid).respond_to_trade(accept=body.accept)
```

- [ ] **Step 4: Run the lifecycle tests**

Run:
```bash
pytest tests/test_web_api.py -v
```
Expected: all PASS (create → play to terminal; 409 on illegal; 404 on unknown game).

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/web/server.py mcts_study/tests/test_web_api.py
git commit -m "feat(web): game lifecycle endpoints (create/state/action/trade-response)"
```

### Task 14: SSE events endpoint

**Files:**
- Modify: `mcts_study/catan_mcts/web/server.py`
- Test: `mcts_study/tests/test_web_api.py`

- [ ] **Step 1: Write the failing test**

Append to `mcts_study/tests/test_web_api.py`:

```python
def test_sse_emits_at_least_one_event(client):
    gid = client.post("/api/games", json=_all_random_setup()).json()["game_id"]
    with client.stream("GET", f"/api/games/{gid}/events") as r:
        assert r.status_code == 200
        got = None
        for line in r.iter_lines():
            if line and line.startswith("data:"):
                got = line
                break
        assert got is not None
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
pytest tests/test_web_api.py::test_sse_emits_at_least_one_event -v
```
Expected: FAIL — `/events` route absent (404).

- [ ] **Step 3: Implement the SSE endpoint**

Add inside `create_app` (before `return app`). It pushes the current state immediately, then polls the session while a background advance runs, emitting on status change; it ends the stream once the session is at a yield point (not advancing):

```python
    import json as _json
    from fastapi.responses import StreamingResponse

    @app.get("/api/games/{gid}/events")
    def events(gid: str):
        sess = _get(gid)

        def gen():
            last = None
            # Emit the current snapshot right away.
            snap = sess.state_json()
            yield f"data: {_json.dumps(snap)}\n\n"
            last = snap["status"]
            # If a slow advance is running in the background, stream updates
            # until the session settles at a yield point.
            import time
            for _ in range(600):  # ~30s cap at 50ms
                if not sess.is_advancing():
                    final = sess.state_json()
                    if final["status"] != last:
                        yield f"data: {_json.dumps(final)}\n\n"
                    break
                cur = sess.state_json()
                if cur["status"] != last:
                    yield f"data: {_json.dumps(cur)}\n\n"
                    last = cur["status"]
                time.sleep(0.05)

        return StreamingResponse(gen(), media_type="text/event-stream")
```

- [ ] **Step 4: Run test**

Run:
```bash
pytest tests/test_web_api.py::test_sse_emits_at_least_one_event -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/web/server.py mcts_study/tests/test_web_api.py
git commit -m "feat(web): SSE events endpoint"
```

### Task 15: Static file serving + `__main__` launcher

**Files:**
- Modify: `mcts_study/catan_mcts/web/server.py`
- Create: `mcts_study/catan_mcts/web/__main__.py`
- Create: `mcts_study/catan_mcts/web/static/index.html` (placeholder; filled in Phase 6)
- Test: `mcts_study/tests/test_web_api.py`

- [ ] **Step 1: Write the failing test**

Append to `mcts_study/tests/test_web_api.py`:

```python
def test_root_serves_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
pytest tests/test_web_api.py::test_root_serves_index -v
```
Expected: FAIL — no `/` route / no static dir.

- [ ] **Step 3: Add static serving + a launcher + placeholder index**

Create `mcts_study/catan_mcts/web/static/index.html` (placeholder, replaced in Task 16):

```html
<!doctype html>
<html><head><meta charset="utf-8"><title>Catan — Play vs Bots</title></head>
<body><div id="app">loading…</div></body></html>
```

In `server.py`, before `return app`, add the root route + static mount:

```python
    @app.get("/")
    def index():
        return FileResponse(_STATIC / "index.html")

    if _STATIC.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
```

Create `mcts_study/catan_mcts/web/__main__.py`:

```python
"""Launch the play-vs-bots server.

Usage (WSL, mcts-study venv active, from mcts_study/):
    python -m catan_mcts.web --checkpoints-dir /path/to/checkpoints \
                             --replays-dir /path/to/replays --port 8000
Then open http://localhost:8000 in the Windows browser.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from catan_mcts.web.server import create_app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints-dir", type=Path, default=Path("."),
                    help="dir scanned recursively for *.pt GNN checkpoints")
    ap.add_argument("--replays-dir", type=Path, default=Path("."),
                    help="dir scanned for existing playback_seed_*/index.html replays")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    app = create_app(checkpoints_dir=args.checkpoints_dir, replays_dir=args.replays_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test + full API suite**

Run:
```bash
pytest tests/test_web_api.py -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/web/server.py mcts_study/catan_mcts/web/__main__.py mcts_study/catan_mcts/web/static/index.html mcts_study/tests/test_web_api.py
git commit -m "feat(web): static serving + python -m catan_mcts.web launcher"
```

---

## Phase 6: Frontend

Vanilla JS, no build step. The board rendering (SVG overlay, player panel, narration formatter) is **ported from `playback.py`'s `<script>`** — same `PLAYER_COLORS`, `dataToPx`, `renderState`, `formatNarration` logic, fed by HTTP state instead of a baked blob.

### Task 16: App shell + nav tabs + style

**Files:**
- Modify: `mcts_study/catan_mcts/web/static/index.html`
- Create: `mcts_study/catan_mcts/web/static/style.css`

- [ ] **Step 1: Write `index.html` (shell + tabs)**

Replace `static/index.html` with the shell. Two tab panels (`#tab-play`, `#tab-replay`), a nav bar, and `<script>` tags for `play.js` + `replay.js`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Catan — Play vs Bots</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<header class="topbar">
  <h1>Catan</h1>
  <nav>
    <button class="tab-btn active" data-tab="play">Play</button>
    <button class="tab-btn" data-tab="replay">Replay</button>
  </nav>
</header>
<main>
  <section id="tab-play" class="tab-panel active"></section>
  <section id="tab-replay" class="tab-panel"></section>
</main>
<script>
  // Tab switching.
  document.querySelectorAll('.tab-btn').forEach(b => b.onclick = () => {
    document.querySelectorAll('.tab-btn').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    document.getElementById('tab-' + b.dataset.tab).classList.add('active');
  });
</script>
<script src="/static/play.js"></script>
<script src="/static/replay.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `style.css`**

Create `static/style.css` matching the project idiom (lift palette + panel styles from `playback.py`'s CSS and `grid_dashboard.html`):

```css
body { font-family: system-ui, sans-serif; margin: 0; background: #f5f5f5; color: #222; }
.topbar { display: flex; align-items: center; gap: 16px; padding: 10px 16px;
          background: #1f2a3a; color: #f0e8c8; }
.topbar h1 { font-size: 18px; margin: 0; }
.tab-btn { padding: 6px 14px; font-size: 14px; cursor: pointer; border: none;
           background: #2c3a52; color: #cdd6e6; border-radius: 4px; }
.tab-btn.active { background: #ffd633; color: #1f2a3a; font-weight: 700; }
.tab-panel { display: none; padding: 16px; }
.tab-panel.active { display: block; }
.panel { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }
.row { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
button { padding: 5px 10px; font-size: 13px; cursor: pointer; }
button.primary { background: #ffd633; border: 1px solid #c90; font-weight: 700; }
button:disabled { opacity: 0.45; cursor: default; }
select { padding: 4px 6px; font-size: 13px; }
#boardWrap { position: relative; width: 700px; max-width: 100%; }
#board { display: block; width: 100%; height: auto; }
svg#overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%;
              pointer-events: none; overflow: visible; }
.clickable { cursor: pointer; }
.modal-bg { position: fixed; inset: 0; background: rgba(0,0,0,0.5);
            display: flex; align-items: center; justify-content: center; }
.modal { background: #fff; padding: 20px 24px; border-radius: 8px; min-width: 280px; }
#log { font-family: ui-monospace, monospace; font-size: 12px; max-height: 200px;
       overflow-y: auto; background: #1f2a3a; color: #f0e8c8; padding: 8px; border-radius: 4px; }
.seat-0 { color: #cc3333; } .seat-1 { color: #3366cc; }
.seat-2 { color: #33aa55; } .seat-3 { color: #cc8833; }
```

- [ ] **Step 3: Manual check**

Run (WSL, from `mcts_study/`):
```bash
python -m catan_mcts.web --checkpoints-dir . --replays-dir . --port 8000
```
Open `http://localhost:8000` in the Windows browser. Expected: topbar with Play/Replay tabs that switch panels (panels empty for now). Ctrl-C to stop.

- [ ] **Step 4: Commit**

```bash
git add mcts_study/catan_mcts/web/static/index.html mcts_study/catan_mcts/web/static/style.css
git commit -m "feat(web): frontend shell — nav tabs + base styles"
```

### Task 17: Setup lobby (play.js part 1)

**Files:**
- Create: `mcts_study/catan_mcts/web/static/play.js`

- [ ] **Step 1: Write the lobby**

Create `static/play.js`. On load, fetch `/api/bots`, render the 4-seat lobby into `#tab-play`: a "You" radio per seat, a bot-type `<select>` for the other seats, a checkpoint `<select>` that appears when a GNN type is chosen, rules controls, and a Start button that POSTs to `/api/games` and calls `startGame(body)` (defined in Task 18).

```javascript
const PLAY = document.getElementById('tab-play');
let BOTS = null;

async function initLobby() {
  BOTS = await (await fetch('/api/bots')).json();
  renderLobby();
}

function botSelect(seat) {
  const opts = BOTS.types.map(t => `<option value="${t.id}">${t.label}</option>`).join('');
  return `<select class="bot-type" data-seat="${seat}">${opts}</select>
          <select class="bot-ckpt" data-seat="${seat}" style="display:none"></select>`;
}

function renderLobby() {
  let rows = '';
  for (let s = 0; s < 4; s++) {
    rows += `<tr>
      <td><label><input type="radio" name="human" value="${s}" ${s===0?'checked':''}> Seat P${s}</label></td>
      <td class="seat-bot" data-seat="${s}">${botSelect(s)}</td></tr>`;
  }
  PLAY.innerHTML = `
    <div class="panel" style="max-width:520px">
      <h2>New game</h2>
      <table>${rows}</table>
      <div style="margin:10px 0">
        VP target <select id="vp"><option value="10" selected>10 (full)</option><option value="5">5 (short)</option></select>
        &nbsp; <label><input type="checkbox" id="bonuses" checked> bonuses (LR/LA +2)</label>
        &nbsp; seed <input id="seed" type="number" placeholder="random" style="width:90px">
      </div>
      <button class="primary" id="start">Start Game</button>
      <span id="lobby-err" style="color:#c33"></span>
    </div>`;
  syncHumanSeat();
  PLAY.querySelectorAll('input[name=human]').forEach(r => r.onchange = syncHumanSeat);
  PLAY.querySelectorAll('.bot-type').forEach(sel => sel.onchange = () => syncCkpt(sel));
  document.getElementById('start').onclick = onStart;
}

function syncHumanSeat() {
  const human = +PLAY.querySelector('input[name=human]:checked').value;
  for (let s = 0; s < 4; s++) {
    const cell = PLAY.querySelector(`.seat-bot[data-seat="${s}"]`);
    cell.style.visibility = (s === human) ? 'hidden' : 'visible';
  }
}

function syncCkpt(sel) {
  const seat = sel.dataset.seat;
  const ck = PLAY.querySelector(`.bot-ckpt[data-seat="${seat}"]`);
  const type = BOTS.types.find(t => t.id === sel.value);
  if (type && type.needs_checkpoint) {
    ck.innerHTML = BOTS.checkpoints.map(c => `<option value="${c.path}">${c.name}</option>`).join('')
                   || '<option value="">(no .pt found)</option>';
    ck.style.display = '';
  } else {
    ck.style.display = 'none';
  }
}

async function onStart() {
  const human = +PLAY.querySelector('input[name=human]:checked').value;
  const seats = {};
  for (let s = 0; s < 4; s++) {
    if (s === human) continue;
    const type = PLAY.querySelector(`.bot-type[data-seat="${s}"]`).value;
    const spec = { type };
    const ck = PLAY.querySelector(`.bot-ckpt[data-seat="${s}"]`);
    if (ck.style.display !== 'none' && ck.value) spec.checkpoint = ck.value;
    seats[s] = spec;
  }
  const seedVal = document.getElementById('seed').value;
  const body = {
    human_seat: human, seats,
    rules: { vp_target: +document.getElementById('vp').value,
             bonuses: document.getElementById('bonuses').checked },
    seed: seedVal === '' ? null : +seedVal,
  };
  const r = await fetch('/api/games', { method: 'POST', headers: {'Content-Type':'application/json'},
                                        body: JSON.stringify(body) });
  if (!r.ok) { document.getElementById('lobby-err').textContent = (await r.json()).detail; return; }
  startGame(await r.json());   // defined in Task 18
}

initLobby();
```

- [ ] **Step 2: Manual check**

Restart the server (Task 16 command), reload `http://localhost:8000`. Expected: a New-game panel with 4 seat rows, picking "Seat P1" as You hides P1's bot dropdown; choosing PureGnn for a seat reveals a checkpoint dropdown (empty if no `.pt` under `--checkpoints-dir`). Start Game errors only because `startGame` isn't defined yet (next task).

- [ ] **Step 3: Commit**

```bash
git add mcts_study/catan_mcts/web/static/play.js
git commit -m "feat(web): setup lobby — per-seat bot + checkpoint selection"
```

### Task 18: Game screen — board render + player panel (play.js part 2)

**Files:**
- Modify: `mcts_study/catan_mcts/web/static/play.js`

- [ ] **Step 1: Append the game screen renderer**

Append to `play.js`. Port `dataToPx`, the SVG building/road/robber drawing, the player table, and `formatNarration` from `playback.py`'s `<script>` (lines 644-879), driven by the live state object. `startGame(body)` stores the board layout + PNG, swaps `#tab-play` to the game view, renders, and connects SSE.

```javascript
let G = null;  // { gid, layout, states... }

function startGame(body) {
  G = { gid: body.game_id, layout: body.board.layout, png: body.board.png_b64,
        state: body.state };
  PLAY.innerHTML = `
    <div class="row">
      <div class="panel board-col">
        <div id="boardWrap">
          <img id="board" src="data:image/png;base64,${G.png}">
          <svg id="overlay" xmlns="http://www.w3.org/2000/svg"></svg>
        </div>
        <div id="actionBar" style="margin-top:8px"></div>
      </div>
      <div class="panel" style="flex:1 1 360px; min-width:340px">
        <div id="status"></div>
        <div id="players"></div>
        <h3 style="font-size:13px;margin:8px 0 4px">Log</h3>
        <div id="log"></div>
      </div>
    </div>`;
  document.getElementById('board').addEventListener('load', renderGame);
  connectSSE();
  applyState(G.state);
}

const PLAYER_COLORS = ["#cc3333", "#3366cc", "#33aa55", "#cc8833"];
const RES = ['🪵','🧱','🐑','🌾','⛰️'];

function dataToPx(x, y) {
  const img = document.getElementById('board');
  const w = img.clientWidth, h = img.clientHeight;
  const [x0,x1] = G.layout.xlim, [y0,y1] = G.layout.ylim;
  return [((x-x0)/(x1-x0))*w, h-((y-y0)/(y1-y0))*h];
}

function applyState(st) {
  G.state = st;
  renderGame();
  // Append narration to log.
  if (st.narration) {
    const log = document.getElementById('log');
    log.innerHTML += `<div>${st.narration}</div>`;
    log.scrollTop = log.scrollHeight;
  }
  if (st.status === 'trade_offer') showTradeModal(st.trade_offer);   // Task 19
  renderActionBar(st);                                               // Task 19
}

function renderGame() {
  if (!G || !G.state) return;
  const st = G.state.state;       // the serialized board sub-object
  const svg = document.getElementById('overlay');
  const img = document.getElementById('board');
  svg.setAttribute('viewBox', `0 0 ${img.clientWidth} ${img.clientHeight}`);
  let body = '';
  // Robber.
  if (st.rh >= 0) {
    const [hx,hy] = G.layout.hex_centers[st.rh]; const [px,py] = dataToPx(hx,hy);
    body += `<circle cx="${px}" cy="${py-10}" r="6" fill="#222" stroke="#fff" stroke-width="1.5"/>`;
  }
  // Roads.
  for (const [eid,o] of st.r) {
    const e = G.layout.edges[eid];
    const [x1,y1] = dataToPx(e[0],e[1]); const [x2,y2] = dataToPx(e[2],e[3]);
    body += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#fff" stroke-width="7" stroke-linecap="round"/>`;
    body += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${PLAYER_COLORS[o]}" stroke-width="4.5" stroke-linecap="round"/>`;
  }
  // Settlements + cities.
  for (const [vid,o] of st.s) {
    const v = G.layout.vertices[String(vid)]; const [px,py] = dataToPx(v[0],v[1]);
    body += `<rect x="${px-8}" y="${py-8}" width="16" height="16" fill="${PLAYER_COLORS[o]}" stroke="#222" stroke-width="1.5"/>`;
  }
  for (const [vid,o] of st.c) {
    const v = G.layout.vertices[String(vid)]; const [px,py] = dataToPx(v[0],v[1]);
    body += `<rect x="${px-10}" y="${py-10}" width="20" height="20" rx="3" fill="${PLAYER_COLORS[o]}" stroke="#fff" stroke-width="2"/>`;
  }
  // Clickable spatial targets when it's your turn (Task 19 fills click handlers).
  body += spatialTargetsSvg(G.state);
  svg.innerHTML = body;
  renderPlayers(G.state);
  renderStatus(G.state);
}

function renderPlayers(g) {
  const st = g.state; let rows = '';
  for (let i = 0; i < 4; i++) {
    const h = st.hands[i];
    const hand = h.breakdown.map((n,r) => n>0?`${RES[r]}${n}`:'').filter(Boolean).join(' ');
    const me = i === g.human_seat ? ' (You)' : '';
    const cp = g.current_player === i ? '▶ ' : '';
    rows += `<tr><td class="seat-${i}"><b>${cp}${g.seat_names[i]}${me}</b></td>
             <td>${st.vp[i]} VP</td><td>${hand||'—'}</td></tr>`;
  }
  document.getElementById('players').innerHTML =
    `<table><tr><th>seat</th><th>VP</th><th>hand</th></tr>${rows}</table>`;
}

function renderStatus(g) {
  const map = { your_turn: 'Your turn', bot_thinking: 'Bot thinking…',
                trade_offer: 'Trade offer', game_over: 'Game over', error: 'Error' };
  let txt = map[g.status] || g.status;
  if (g.status === 'game_over' && g.returns) {
    const w = g.returns.indexOf(1);
    txt = (w === g.human_seat) ? 'You win 🎉' : `${g.seat_names[w]} wins`;
  }
  document.getElementById('status').innerHTML = `<b>${txt}</b>`;
}

// Filled in Task 19:
function spatialTargetsSvg(g) { return ''; }
function renderActionBar(g) {}
function showTradeModal(o) {}
function connectSSE() {}
```

- [ ] **Step 2: Manual check**

Restart server, start an all-Random game with you as P0. Expected: the board PNG renders with the SVG overlay (roads/settlements appear as bots build), the player panel shows VP + hands, status shows "Your turn" / "Bot thinking…". You can't act yet (action bar lands next task), but you should see the initial state and the log.

- [ ] **Step 3: Commit**

```bash
git add mcts_study/catan_mcts/web/static/play.js
git commit -m "feat(web): game screen — board overlay + player panel + status"
```

### Task 19: Interaction — action bar, clickable board, trade modal, SSE

**Files:**
- Modify: `mcts_study/catan_mcts/web/static/play.js`

- [ ] **Step 1: Replace the four stubs with real implementations**

In `play.js`, replace `spatialTargetsSvg`, `renderActionBar`, `showTradeModal`, and `connectSSE`:

```javascript
async function postAction(actionId) {
  const r = await fetch(`/api/games/${G.gid}/action`,
    { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ action: actionId }) });
  if (r.status === 409) { const s = await fetch(`/api/games/${G.gid}/state`); applyState(await s.json()); return; }
  applyState(await r.json());
}

function spatialTargetsSvg(g) {
  if (g.status !== 'your_turn' || !g.legal_actions) return '';
  let out = '';
  for (const a of g.legal_actions) {
    if (a.target === null) continue;
    let px, py;
    if (a.kind === 'build_road') {
      const e = G.layout.edges[a.target]; [px,py] = dataToPx((e[0]+e[2])/2,(e[1]+e[3])/2);
    } else if (a.kind === 'move_robber') {
      const c = G.layout.hex_centers[a.target]; [px,py] = dataToPx(c[0],c[1]);
    } else {
      const v = G.layout.vertices[String(a.target)]; [px,py] = dataToPx(v[0],v[1]);
    }
    out += `<circle class="clickable" cx="${px}" cy="${py}" r="10" fill="#ffd633" fill-opacity="0.5"
             stroke="#c90" stroke-width="2" style="pointer-events:all" onclick="postAction(${a.id})"/>`;
  }
  return out;
}

function renderActionBar(g) {
  const bar = document.getElementById('actionBar');
  if (g.status !== 'your_turn' || !g.legal_actions) { bar.innerHTML = ''; return; }
  // Non-spatial actions become buttons; spatial ones are board clicks.
  const NON_SPATIAL = new Set(['roll','end_turn','buy_dev','trade_bank','propose_trade','play_dev','discard']);
  const seen = new Map();
  for (const a of g.legal_actions) {
    if (!NON_SPATIAL.has(a.kind)) continue;
    if (!seen.has(a.id)) seen.set(a.id, a);
  }
  bar.innerHTML = [...seen.values()]
    .map(a => `<button onclick="postAction(${a.id})">${a.label}</button>`).join(' ');
}

async function respondTrade(accept) {
  document.querySelectorAll('.modal-bg').forEach(m => m.remove());
  const r = await fetch(`/api/games/${G.gid}/trade-response`,
    { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ accept }) });
  applyState(await r.json());
}

function showTradeModal(o) {
  document.querySelectorAll('.modal-bg').forEach(m => m.remove());
  const div = document.createElement('div');
  div.className = 'modal-bg';
  div.innerHTML = `<div class="modal">
    <p><b class="seat-${o.from_seat}">${G.state.seat_names[o.from_seat]}</b> offers a trade:</p>
    <p>You give ${RES[o.you_give[0]]}×${o.you_give[1]}, you get ${RES[o.you_get[0]]}×${o.you_get[1]}</p>
    <button class="primary" onclick="respondTrade(true)">Accept</button>
    <button onclick="respondTrade(false)">Reject</button></div>`;
  document.body.appendChild(div);
}

function connectSSE() {
  if (G._sse) G._sse.close();
  G._sse = new EventSource(`/api/games/${G.gid}/events`);
  G._sse.onmessage = (ev) => { try { applyState(JSON.parse(ev.data)); } catch(_){} };
  G._sse.onerror = () => { /* stream ends at each yield point; reopened on next action */ };
}
```

- [ ] **Step 2: Manual check (full play-through)**

Restart server, start an all-Random game as P0. Expected: on your turn, legal build spots glow amber on the board and click-to-build works; Roll / End Turn / Buy Dev Card / Bank Trade / Propose Trade appear as buttons; when a bot offers you a trade a modal pops with Accept/Reject; the game plays to a "You win"/"… wins" status. Try a game where you're P1 with a `LookaheadMctsV3` opponent to confirm "Bot thinking…" shows during slow moves.

- [ ] **Step 3: Commit**

```bash
git add mcts_study/catan_mcts/web/static/play.js
git commit -m "feat(web): interaction — action bar, clickable board, trade modal, SSE"
```

### Task 20: Replay tab (replay.js)

**Files:**
- Create: `mcts_study/catan_mcts/web/static/replay.js`
- Modify: `mcts_study/catan_mcts/web/server.py` (add `/api/replays`)
- Test: `mcts_study/tests/test_web_api.py`

- [ ] **Step 1: Write the failing test**

Append to `mcts_study/tests/test_web_api.py`:

```python
def test_replays_listing(tmp_path):
    from catan_mcts.web.server import create_app
    # Make a fake replay output dir.
    rd = tmp_path / "playback_seed_4242"
    rd.mkdir()
    (rd / "index.html").write_text("<html>replay</html>")
    app = create_app(checkpoints_dir=tmp_path, replays_dir=tmp_path)
    c = TestClient(app)
    r = c.get("/api/replays")
    assert r.status_code == 200
    names = [x["name"] for x in r.json()["replays"]]
    assert "playback_seed_4242" in names
```

- [ ] **Step 2: Run to confirm failure**

Run:
```bash
pytest tests/test_web_api.py::test_replays_listing -v
```
Expected: FAIL — `/api/replays` route absent.

- [ ] **Step 3: Add `/api/replays` + replay file serving**

In `server.py`, inside `create_app` before `return app`:

```python
    @app.get("/api/replays")
    def list_replays():
        out = []
        if replays_dir.exists():
            for d in sorted(replays_dir.glob("playback_seed_*")):
                if (d / "index.html").exists():
                    out.append({"name": d.name, "url": f"/replays/{d.name}/index.html"})
        return {"replays": out}

    if replays_dir.exists():
        app.mount("/replays", StaticFiles(directory=str(replays_dir)), name="replays")
```

Create `static/replay.js`:

```javascript
const REPLAY = document.getElementById('tab-replay');

async function initReplay() {
  const data = await (await fetch('/api/replays')).json();
  if (!data.replays.length) {
    REPLAY.innerHTML = `<div class="panel">No replays found. Generate one with
      <code>python -m catan_mcts.playback &lt;run_dir&gt; &lt;seed&gt;</code> into the
      server's <code>--replays-dir</code>.</div>`;
    return;
  }
  REPLAY.innerHTML = `<div class="panel"><h2>Replays</h2><ul>` +
    data.replays.map(r => `<li><a href="${r.url}" target="_blank">${r.name}</a></li>`).join('') +
    `</ul></div>`;
}

initReplay();
```

- [ ] **Step 4: Run test + manual check**

Run:
```bash
pytest tests/test_web_api.py::test_replays_listing -v
```
Expected: PASS. Manual: with `--replays-dir` pointed at a dir containing a `playback_seed_*/index.html`, the Replay tab lists it and the link opens the existing viewer.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/web/static/replay.js mcts_study/catan_mcts/web/server.py mcts_study/tests/test_web_api.py
git commit -m "feat(web): replay tab — list + serve existing playback outputs"
```

---

## Phase 7: Wiring & docs

### Task 21: Full suite + optional Playwright smoke

**Files:**
- Test (optional): `mcts_study/tests/test_web_smoke.py`

- [ ] **Step 1: Run the entire web + playback test suite**

Run (from `mcts_study/`):
```bash
pytest tests/test_board_layout.py tests/test_serializers.py tests/test_serializers_golden.py tests/test_playback.py tests/test_bot_registry.py tests/test_action_decode.py tests/test_trade_logic.py tests/test_game_session.py tests/test_web_api.py -v
```
Expected: all PASS, no regressions in `test_playback.py`.

- [ ] **Step 2 (optional): Playwright smoke test**

If the `webapp-testing` (Playwright) tooling is available in WSL, create `mcts_study/tests/test_web_smoke.py` marked `@pytest.mark.slow` that: launches the server on a free port, loads `/`, starts an all-Random game as P0, clicks End Turn / a build target a few times, and asserts the board SVG has children and the status text updates. If Playwright is not wired for WSL, skip this step — the Python API tests in Step 1 are the contract gate. Record the decision in the commit message.

- [ ] **Step 3: Commit (if smoke test added)**

```bash
git add mcts_study/tests/test_web_smoke.py
git commit -m "test(web): optional Playwright smoke for the play screen"
```

### Task 22: README / run docs

**Files:**
- Create: `mcts_study/catan_mcts/web/README.md`

- [ ] **Step 1: Write the run doc**

Create `mcts_study/catan_mcts/web/README.md`:

```markdown
# Interactive Play-vs-Bots

A local web app to play Catan as one seat against selectable bots, with
human-in-the-loop trade responses. Shares board rendering with the offline
replay viewer (`catan_mcts.playback`).

## Run (WSL, mcts-study venv active, from `mcts_study/`)

    pip install -e ".[web]"        # one-time: fastapi/uvicorn
    python -m catan_mcts.web \
        --checkpoints-dir /path/to/gnn/checkpoints \
        --replays-dir     /path/to/playback/outputs \
        --port 8000

Open http://localhost:8000 in your browser.

- **Play tab:** pick your seat, choose each opponent bot (GNN bots let you
  select any `.pt` from `--checkpoints-dir`), set rules, Start.
- **Replay tab:** lists `playback_seed_*/index.html` dirs under `--replays-dir`.

## How it works

- `server.py` — FastAPI app (REST + SSE). Paths are CLI args; no hardcoded paths.
- `game_session.py` — one live game; drives chance + bot turns, intercepts a
  bot `ProposeTrade` that would auto-match you and pauses for Accept/Reject
  (the Rust engine resolves trades instantly, so this lives in Python — see
  `docs/superpowers/specs/2026-05-31-interactive-play-vs-bots-design.md` §2).
- `bot_registry.py` — bot types + `.pt` discovery + construction.
- `board_layout.py` / `serializers.py` — shared with `playback.py`.

## Deploy note

Local-first but deploy-ready: clean REST/SSE API, configurable paths, one game
per server-side session. Multi-user/auth/scaling are out of scope (this iteration).
```

- [ ] **Step 2: Commit**

```bash
git add mcts_study/catan_mcts/web/README.md
git commit -m "docs(web): run + architecture README for play-vs-bots"
```

- [ ] **Step 3: Finish the branch**

Use the `superpowers:finishing-a-development-branch` skill to present merge/PR options for the `worktree-interactive-play` branch.

---

## Notes for the implementer

- **No engine changes / no maturin rebuild** anywhere in this plan — confirm before any `cargo`/`maturin` reflex.
- **WSL execution:** all `pytest` and `python -m` commands run in WSL with the mcts-study venv active (project memory: "WSL setup for MCTS-study"). The Windows-side editable install will not find `catan_bot`.
- **Checkpoints** live on the WSL filesystem; pass their dir via `--checkpoints-dir`. Never hardcode a path (deploy goal).
- **Skip-trivial-turn** parity: `advance()` mirrors `common.py`'s single-legal-action shortcut so forced moves don't stall the human or burn bot compute.
- **The golden test (Task 1)** is the safety net for the Phase-1 refactor — if it ever fails after Tasks 2-3, the extraction changed replay output and must be fixed before proceeding.
```
