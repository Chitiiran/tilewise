# Batched GNN Evaluator + Async Self-Play Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-process asyncio self-play engine that batches MCTS leaf evaluations across many concurrent games into one GPU forward pass (~10–50× speedup over the current batch=1 `GnnEvaluator`).

**Architecture:** N game-coroutines each drive a minimal async MCTS; at every leaf they `await` a shared `BatchedGnnEvaluator`, which parks the request on a queue and fires one batched forward pass when the batch fills, a time window elapses, or all live games are parked. Finished games are written via the existing `SelfPlayRecorder` (e9 parquet schema, unchanged).

**Tech Stack:** Python `asyncio`, PyTorch + PyTorch-Geometric (`GnnModel` returns `(value, policy_logits)`), the existing `CatanGame`/`CatanState` engine adapter, `pytest` + `pytest-asyncio`.

**Spec:** `docs/superpowers/specs/2026-05-30-batched-gnn-evaluator-design.md`

---

## Environment notes (read before starting)

- **Run Python in WSL Ubuntu**, venv at `~/catan_mcts_venvs/mcts-study/bin/python`. From Windows: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest ...'`.
- **No engine (Rust) changes in this plan** — pure Python. `maturin develop` is NOT needed.
- **`pytest-asyncio` may need installing.** Task 0 checks/installs it.
- All paths below are relative to `mcts_study/` unless stated.

## Key API facts (verified against the codebase — do not re-derive)

- `GnnModel(batch) -> (value: Tensor[B,4], policy_logits: Tensor[B,ACTION_SPACE_SIZE])`. Import `ACTION_SPACE_SIZE` from `catan_mcts`.
- `state_to_pyg(obs: dict) -> HeteroData`. Get `obs` via `state._engine.observation()`.
- `Batch.from_data_list([HeteroData, ...])` from `torch_geometric.data` makes one batched graph; the model returns per-graph rows in order.
- `CatanState` methods: `.is_terminal()`, `.current_player()` (returns int, or `pyspiel.PlayerId.CHANCE`/`.TERMINAL`), `.is_chance_node()`, `.chance_outcomes() -> list[(int,float)]`, `.returns() -> list[float]` (±1.0 winner / -1.0 others / 0.0 no-winner), `.legal_actions() -> list[int]`, `.apply_action(int)`, `.clone()`, `._engine.observation()`.
- `CatanGame(vp_target=10, bonuses=True).new_initial_state(seed=int)` makes a fresh game.
- **Terminal value semantics for the GNN path:** terminal leaf value = `state.returns()` (NOT length-discounted; the `DECAY^steps` discount lives only in the Rust random-rollout path, which we do not use here).
- `SelfPlayRecorder(out_dir, config)` with `.game(seed)` context manager → `_GameRecorder` exposing `.record_move(*, current_player, move_index, legal_action_mask, mcts_visit_counts, action_taken, mcts_root_value)` and `.finalize(*, winner, final_vp, length_in_moves, action_history, timed_out=False)`. Also `.skip_game(...)`, `.done_seeds()`, `.mark_done(seed)`, `.checkpoint(label)`, `.flush()`.
- `legal_action_mask` and `mcts_visit_counts` passed to `record_move` must be length-`ACTION_SPACE_SIZE` arrays.

## File structure

| File | Responsibility |
|---|---|
| `catan_mcts/batched_evaluator.py` (create) | `BatchedGnnEvaluator`: async `eval()`, the batcher coroutine, flush logic, OOM handling |
| `catan_mcts/async_mcts.py` (create) | `AsyncMcts` + `Node`: UCB selection, expansion, `await`-at-leaf, backup, visit-count extraction |
| `catan_mcts/experiments/self_play_async.py` (create) | Orchestrator: spawn N game-coroutines, per-game RNG, watchdog, memory-budget cap, recording, resume |
| `catan_mcts/experiments/e10e_async.py` (create) | Acceptance Gate 2: clean e10e re-run on the async stack |
| `tests/test_batched_evaluator.py` (create) | Unit tests for the evaluator |
| `tests/test_async_mcts.py` (create) | Unit tests for the async MCTS + equivalence-on-toy |
| `tests/test_self_play_async.py` (create) | Integration: parquet output, resume, RNG reproducibility, budget cap |

---

## Task 0: Environment prep

**Files:** none (dependency check)

- [ ] **Step 1: Verify pytest-asyncio is available**

Run: `wsl.exe -d Ubuntu -- bash -c '~/catan_mcts_venvs/mcts-study/bin/python -c "import pytest_asyncio; print(pytest_asyncio.__version__)"'`
Expected: a version string. If `ModuleNotFoundError`, run:
`wsl.exe -d Ubuntu -- bash -c '~/catan_mcts_venvs/mcts-study/bin/pip install pytest-asyncio'`

- [ ] **Step 2: Add asyncio_mode to pytest config**

Modify `mcts_study/pytest.ini` (or `pyproject.toml` `[tool.pytest.ini_options]` if that's where config lives — check which exists first). Add under the pytest section:

```ini
asyncio_mode = auto
```

This lets `async def test_*` run without a per-test `@pytest.mark.asyncio` decorator.

- [ ] **Step 3: Commit**

```bash
git add mcts_study/pytest.ini
git commit -m "chore: enable pytest-asyncio auto mode for async self-play tests"
```

---

## Task 1: BatchedGnnEvaluator — single-request path

**Files:**
- Create: `catan_mcts/batched_evaluator.py`
- Test: `tests/test_batched_evaluator.py`

- [ ] **Step 1: Write the failing test — one request resolves correctly**

```python
# tests/test_batched_evaluator.py
import asyncio
import numpy as np
import torch
from catan_gnn.gnn_model import GnnModel
from catan_mcts.adapter import CatanGame
from catan_mcts.batched_evaluator import BatchedGnnEvaluator


def _untrained_model():
    torch.manual_seed(0)
    return GnnModel(hidden_dim=8, num_layers=2)


def _leaf_state():
    game = CatanGame(vp_target=10, bonuses=True)
    state = game.new_initial_state(seed=42)
    while state.is_chance_node():
        state.apply_action(int(state.chance_outcomes()[0][0]))
    return state


async def test_single_eval_returns_value_and_policy():
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=8, window_ms=5)
    ev.start()
    try:
        value, policy = await ev.eval(_leaf_state())
        assert isinstance(value, np.ndarray) and value.shape == (4,)
        assert isinstance(policy, np.ndarray)
        assert (value >= -1.0).all() and (value <= 1.0).all()
    finally:
        await ev.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_batched_evaluator.py::test_single_eval_returns_value_and_policy -x -q'`
Expected: FAIL — `ModuleNotFoundError: catan_mcts.batched_evaluator`.

- [ ] **Step 3: Write minimal implementation**

```python
# catan_mcts/batched_evaluator.py
"""Async GNN evaluator that batches MCTS leaf evals across concurrent games.

Each eval() call parks a Future on a pending queue and suspends. A background
batcher coroutine drains the queue and runs ONE forward pass per batch, then
resolves all the parked Futures. See spec 2026-05-30-batched-gnn-evaluator.
"""
from __future__ import annotations

import asyncio
import numpy as np
import torch
from torch_geometric.data import Batch

from catan_gnn.gnn_model import GnnModel
from catan_gnn.state_to_pyg import state_to_pyg


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


class BatchedGnnEvaluator:
    def __init__(self, model: GnnModel, device: str = "cpu",
                 max_batch: int = 64, window_ms: float = 5.0) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.max_batch = int(max_batch)
        self.window_s = float(window_ms) / 1000.0
        self._pending: list[tuple] = []   # (features, future)
        self._wakeup: asyncio.Event | None = None
        self._batcher_task: asyncio.Task | None = None
        self._stopped = False
        # active_game_count is set by the orchestrator each step; default huge
        # so the all-parked flush clause never fires spuriously in unit tests.
        self.active_game_count = 10 ** 9
        # Stats (health metric).
        self.total_batches = 0
        self.total_requests = 0

    def start(self) -> None:
        self._wakeup = asyncio.Event()
        self._batcher_task = asyncio.ensure_future(self._batcher_loop())

    async def stop(self) -> None:
        self._stopped = True
        if self._wakeup is not None:
            self._wakeup.set()
        if self._batcher_task is not None:
            await self._batcher_task

    @torch.no_grad()
    def _run_forward(self, features_list):
        batch = Batch.from_data_list(features_list).to(self.device)
        v, logits = self.model(batch)
        v_np = v.cpu().numpy().astype(np.float32)
        l_np = logits.cpu().numpy().astype(np.float32)
        return v_np, l_np

    async def eval(self, state):
        # Features built on the caller side (cheap, CPU).
        obs = state._engine.observation()
        features = state_to_pyg(obs)
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._pending.append((features, fut))
        self.total_requests += 1
        if self._wakeup is not None:
            self._wakeup.set()
        return await fut

    async def _batcher_loop(self):
        while not self._stopped:
            if not self._pending:
                await self._wakeup.wait()
                self._wakeup.clear()
                continue
            drained = self._pending[: self.max_batch]
            self._pending = self._pending[self.max_batch :]
            feats = [f for f, _ in drained]
            v_np, l_np = self._run_forward(feats)
            self.total_batches += 1
            for i, (_, fut) in enumerate(drained):
                if not fut.done():
                    fut.set_result((v_np[i], l_np[i]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_batched_evaluator.py::test_single_eval_returns_value_and_policy -x -q'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add catan_mcts/batched_evaluator.py tests/test_batched_evaluator.py
git commit -m "feat(batched-eval): async evaluator single-request path"
```

---

## Task 2: BatchedGnnEvaluator — batching, window, and all-parked flush

**Files:**
- Modify: `catan_mcts/batched_evaluator.py`
- Test: `tests/test_batched_evaluator.py`

- [ ] **Step 1: Write the failing tests — batch fill, window-partial, all-parked**

```python
# append to tests/test_batched_evaluator.py

async def test_batch_fills_to_max_in_one_forward():
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=8, window_ms=50)
    ev.start()
    try:
        states = [_leaf_state() for _ in range(8)]
        results = await asyncio.gather(*[ev.eval(s) for s in states])
        assert len(results) == 8
        # 8 requests, max_batch 8 -> exactly one batch.
        assert ev.total_batches == 1
    finally:
        await ev.stop()


async def test_window_fires_partial_batch():
    # 3 requests < max_batch 8: must still resolve via the time window, not hang.
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=8, window_ms=20)
    ev.start()
    try:
        states = [_leaf_state() for _ in range(3)]
        results = await asyncio.wait_for(
            asyncio.gather(*[ev.eval(s) for s in states]), timeout=2.0)
        assert len(results) == 3
    finally:
        await ev.stop()


async def test_all_parked_flushes_immediately():
    # When pending >= active_game_count, flush without waiting for the window.
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=64, window_ms=10_000)  # huge window
    ev.active_game_count = 3
    ev.start()
    try:
        states = [_leaf_state() for _ in range(3)]
        # If the all-parked clause works, this resolves well under the 10s window.
        results = await asyncio.wait_for(
            asyncio.gather(*[ev.eval(s) for s in states]), timeout=2.0)
        assert len(results) == 3
    finally:
        await ev.stop()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_batched_evaluator.py -x -q'`
Expected: `test_window_fires_partial_batch` and `test_all_parked_flushes_immediately` FAIL/HANG (current loop fires immediately on any pending, so `test_batch_fills_to_max` may already pass but the window/all-parked semantics aren't correct). The point is the loop has no window/flush-condition logic yet.

- [ ] **Step 3: Replace the batcher loop with proper flush conditions**

Replace `_batcher_loop` in `catan_mcts/batched_evaluator.py` with:

```python
    async def _batcher_loop(self):
        while not self._stopped:
            if not self._pending:
                await self._wakeup.wait()
                self._wakeup.clear()
                continue
            # Decide whether to flush now or wait for more requests.
            first_arrival = asyncio.get_event_loop().time()
            while not self._stopped:
                n = len(self._pending)
                flush_now = (
                    n >= self.max_batch
                    or n >= self.active_game_count
                )
                if flush_now:
                    break
                elapsed = asyncio.get_event_loop().time() - first_arrival
                if elapsed >= self.window_s:
                    break  # window fired -> flush partial
                # Sleep a short slice to let more requests arrive.
                try:
                    await asyncio.wait_for(self._wakeup.wait(),
                                           timeout=self.window_s - elapsed)
                except asyncio.TimeoutError:
                    pass
                self._wakeup.clear()
            if not self._pending:
                continue
            drained = self._pending[: self.max_batch]
            self._pending = self._pending[self.max_batch :]
            feats = [f for f, _ in drained]
            v_np, l_np = self._run_forward(feats)
            self.total_batches += 1
            for i, (_, fut) in enumerate(drained):
                if not fut.done():
                    fut.set_result((v_np[i], l_np[i]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_batched_evaluator.py -x -q'`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add catan_mcts/batched_evaluator.py tests/test_batched_evaluator.py
git commit -m "feat(batched-eval): flush on batch-full, window timeout, or all-parked"
```

---

## Task 3: BatchedGnnEvaluator — chance/terminal short-circuit + OOM retry

**Files:**
- Modify: `catan_mcts/batched_evaluator.py`
- Test: `tests/test_batched_evaluator.py`

- [ ] **Step 1: Write the failing test — terminal/chance never hit the model**

```python
# append to tests/test_batched_evaluator.py

async def test_eval_leaf_helper_skips_model_for_terminal():
    # eval_leaf() is the MCTS-facing entry: returns state.returns() for terminals
    # WITHOUT enqueuing a GPU request.
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=8, window_ms=5)
    ev.start()
    try:
        game = CatanGame(vp_target=10, bonuses=True)
        # Drive a game to terminal by always taking the first legal action.
        state = game.new_initial_state(seed=7)
        steps = 0
        while not state.is_terminal() and steps < 200000:
            if state.is_chance_node():
                state.apply_action(int(state.chance_outcomes()[0][0]))
            else:
                state.apply_action(int(state.legal_actions()[0]))
            steps += 1
        assert state.is_terminal()
        before = ev.total_requests
        value, priors = await ev.eval_leaf(state)
        assert ev.total_requests == before  # no GPU request enqueued
        assert priors is None
        assert list(value) == state.returns()
    finally:
        await ev.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_batched_evaluator.py::test_eval_leaf_helper_skips_model_for_terminal -x -q'`
Expected: FAIL — `AttributeError: 'BatchedGnnEvaluator' object has no attribute 'eval_leaf'`.

- [ ] **Step 3: Add eval_leaf() and OOM-resilient forward**

Add to `BatchedGnnEvaluator` in `catan_mcts/batched_evaluator.py`:

```python
    async def eval_leaf(self, state):
        """MCTS-facing leaf evaluation.

        Returns (value: np.ndarray[4], priors: list[(action,prob)] | None).
        - terminal  -> (state.returns(), None), no GPU
        - otherwise -> (value_head, policy-over-legal), via the batched model
        Chance nodes are handled by AsyncMcts itself (it expands outcomes), so
        eval_leaf is never called on a chance node.
        """
        if state.is_terminal():
            return np.asarray(state.returns(), dtype=np.float32), None
        value, logits = await self.eval(state)
        legal = state.legal_actions()
        if not legal:
            return value, []
        legal_arr = np.asarray(legal, dtype=np.int64)
        probs = _softmax(logits[legal_arr])
        priors = [(int(a), float(p)) for a, p in zip(legal, probs)]
        return value, priors
```

Then make `_run_forward` OOM-resilient by replacing it with:

```python
    @torch.no_grad()
    def _run_forward(self, features_list):
        try:
            batch = Batch.from_data_list(features_list).to(self.device)
            v, logits = self.model(batch)
            return (v.cpu().numpy().astype(np.float32),
                    logits.cpu().numpy().astype(np.float32))
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or len(features_list) <= 1:
                raise
            # Halve and retry once (recurse on each half).
            torch.cuda.empty_cache()
            half = len(features_list) // 2
            v1, l1 = self._run_forward(features_list[:half])
            v2, l2 = self._run_forward(features_list[half:])
            return np.concatenate([v1, v2]), np.concatenate([l1, l2])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_batched_evaluator.py -x -q'`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add catan_mcts/batched_evaluator.py tests/test_batched_evaluator.py
git commit -m "feat(batched-eval): eval_leaf chance/terminal short-circuit + OOM halve-retry"
```

---

## Task 4: AsyncMcts — node + UCB selection + expansion + backup

**Files:**
- Create: `catan_mcts/async_mcts.py`
- Test: `tests/test_async_mcts.py`

- [ ] **Step 1: Write the failing test — one full search returns visit counts**

```python
# tests/test_async_mcts.py
import asyncio
import numpy as np
import torch
from catan_gnn.gnn_model import GnnModel
from catan_mcts.adapter import CatanGame
from catan_mcts.batched_evaluator import BatchedGnnEvaluator
from catan_mcts.async_mcts import AsyncMcts
from catan_mcts import ACTION_SPACE_SIZE


def _untrained_model():
    torch.manual_seed(0)
    return GnnModel(hidden_dim=8, num_layers=2)


def _leaf_state(seed=42):
    game = CatanGame(vp_target=10, bonuses=True)
    state = game.new_initial_state(seed=seed)
    while state.is_chance_node():
        state.apply_action(int(state.chance_outcomes()[0][0]))
    return state


async def test_search_returns_visit_counts():
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=4, window_ms=5)
    ev.start()
    try:
        mcts = AsyncMcts(evaluator=ev, c=1.4, rng=np.random.default_rng(0))
        state = _leaf_state()
        visits = await mcts.search(state, n_sims=16)
        assert visits.shape == (ACTION_SPACE_SIZE,)
        assert 0 < int(visits.sum()) <= 16
        legal = set(state.legal_actions())
        assert all(visits[a] == 0 for a in range(ACTION_SPACE_SIZE) if a not in legal)
    finally:
        await ev.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_async_mcts.py -x -q'`
Expected: FAIL — `ModuleNotFoundError: catan_mcts.async_mcts`.

- [ ] **Step 3: Write the async MCTS**

```python
# catan_mcts/async_mcts.py
"""Minimal async MCTS for batched self-play.

PUCT/UCB over a CatanState tree. The ONLY await is the leaf evaluation, which
suspends the coroutine so other games' leaves can batch. Matches the OpenSpiel
MCTSBot semantics we rely on: uct_c=1.4, priors from the policy head, value
from the value head (or state.returns() at terminals), 4-player per-seat backup,
argmax-visit final move. See spec 2026-05-30-batched-gnn-evaluator.
"""
from __future__ import annotations

import math
import numpy as np

from catan_mcts import ACTION_SPACE_SIZE


class Node:
    __slots__ = ("state", "to_play", "is_expanded", "children", "prior",
                 "visit_count", "value_sum")

    def __init__(self, state, prior: float = 0.0) -> None:
        self.state = state
        self.to_play = state.current_player() if not state.is_terminal() else -1
        self.is_expanded = False
        self.children: dict = {}
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0


class AsyncMcts:
    def __init__(self, evaluator, c: float = 1.4, rng=None) -> None:
        self.ev = evaluator
        self.c = float(c)
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def _ucb_score(self, parent: "Node", child: "Node") -> float:
        q = (child.value_sum / child.visit_count) if child.visit_count else 0.0
        u = self.c * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        return q + u

    def _select_child(self, node: "Node"):
        best_score, best_a, best_child = -float("inf"), None, None
        for a, child in node.children.items():
            s = self._ucb_score(node, child)
            if s > best_score:
                best_score, best_a, best_child = s, a, child
        return best_a, best_child

    async def _expand_and_evaluate(self, node: "Node"):
        state = node.state
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            r = float(self.rng.random())
            cum, chosen = 0.0, outcomes[-1][0]
            for v, p in outcomes:
                cum += p
                if r <= cum:
                    chosen = v
                    break
            nxt = state.clone()
            nxt.apply_action(int(chosen))
            node.state = nxt
            node.to_play = nxt.current_player() if not nxt.is_terminal() else -1
            return await self._expand_and_evaluate(node)
        value, priors = await self.ev.eval_leaf(state)
        if priors is not None:
            for a, p in priors:
                child_state = state.clone()
                child_state.apply_action(int(a))
                node.children[a] = Node(child_state, prior=p)
            node.is_expanded = True
        return np.asarray(value, dtype=np.float32)

    def _backup(self, path, value_vec: np.ndarray) -> None:
        for node in path:
            node.visit_count += 1
            if node.to_play >= 0:
                node.value_sum += float(value_vec[node.to_play])

    async def search(self, root_state, n_sims: int) -> np.ndarray:
        root = Node(root_state.clone())
        root_value = await self._expand_and_evaluate(root)
        root.visit_count += 1
        if root.to_play >= 0:
            root.value_sum += float(root_value[root.to_play])
        for _ in range(n_sims - 1):
            node, path = root, [root]
            while node.is_expanded and node.children and not node.state.is_terminal():
                _, node = self._select_child(node)
                path.append(node)
            value_vec = await self._expand_and_evaluate(node)
            self._backup(path, value_vec)
        out = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
        for a, child in root.children.items():
            out[a] = child.visit_count
        return out

    def best_action(self, visit_counts: np.ndarray) -> int:
        return int(np.argmax(visit_counts))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_async_mcts.py -x -q'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add catan_mcts/async_mcts.py tests/test_async_mcts.py
git commit -m "feat(async-mcts): UCB selection, expand-on-eval, 4-player backup"
```

---

## Task 5: AsyncMcts — play a full game to terminal

**Files:**
- Modify: `catan_mcts/async_mcts.py`
- Test: `tests/test_async_mcts.py`

- [ ] **Step 1: Write the failing test — drive a full game**

```python
# append to tests/test_async_mcts.py

async def test_play_full_game_terminates_and_records():
    from catan_mcts.async_mcts import play_one_async_game
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=4, window_ms=5)
    ev.start()
    try:
        game = CatanGame(vp_target=10, bonuses=True)
        result = await play_one_async_game(
            game=game, seed=123, evaluator=ev, n_sims=8,
            rng=np.random.default_rng(123), max_steps=200000)
        assert result.terminal is True
        assert -1 <= result.winner <= 3
        assert result.length_in_moves > 0
        assert len(result.moves) > 0
        m = result.moves[0]
        assert m.visit_counts.shape == (ACTION_SPACE_SIZE,)
        assert m.legal_mask.shape == (ACTION_SPACE_SIZE,)
    finally:
        await ev.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_async_mcts.py::test_play_full_game_terminates_and_records -x -q'`
Expected: FAIL — `ImportError: cannot import name 'play_one_async_game'`.

- [ ] **Step 3: Add the game-driver and result dataclasses**

Add to `catan_mcts/async_mcts.py`:

```python
from dataclasses import dataclass, field


@dataclass
class RecordedMove:
    current_player: int
    move_index: int
    legal_mask: np.ndarray
    visit_counts: np.ndarray
    action_taken: int
    root_value: float


@dataclass
class GameResult:
    seed: int
    terminal: bool
    winner: int
    final_vp: list
    length_in_moves: int
    action_history: list
    moves: list = field(default_factory=list)


async def play_one_async_game(*, game, seed: int, evaluator, n_sims: int,
                              rng, max_steps: int = 200000):
    state = game.new_initial_state(seed=seed)
    mcts = AsyncMcts(evaluator=evaluator, c=1.4, rng=rng)
    moves: list = []
    move_index = 0
    steps = 0
    while not state.is_terminal() and steps < max_steps:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            r = float(rng.random())
            cum, chosen = 0.0, outcomes[-1][0]
            for v, p in outcomes:
                cum += p
                if r <= cum:
                    chosen = v
                    break
            state.apply_action(int(chosen))
            steps += 1
            continue
        legal = state.legal_actions()
        if len(legal) == 1:
            state.apply_action(int(legal[0]))
            steps += 1
            continue
        visit_counts = await mcts.search(state, n_sims=n_sims)
        action = mcts.best_action(visit_counts)
        legal_mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        legal_mask[np.asarray(legal, dtype=np.int64)] = 1
        moves.append(RecordedMove(
            current_player=int(state.current_player()), move_index=move_index,
            legal_mask=legal_mask, visit_counts=visit_counts,
            action_taken=int(action), root_value=0.0))
        state.apply_action(int(action))
        move_index += 1
        steps += 1
    terminal = state.is_terminal()
    if terminal:
        rets = state.returns()
        winner = int(np.argmax(rets)) if max(rets) > 0 else -1
    else:
        winner = -1
    final_vp = [0, 0, 0, 0]
    try:
        stats = state._engine.stats()
        final_vp = [int(x) for x in stats.get("final_vp", final_vp)]
    except Exception:
        pass
    return GameResult(seed=seed, terminal=terminal, winner=winner,
                      final_vp=final_vp, length_in_moves=steps,
                      action_history=state.history(), moves=moves)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_async_mcts.py -x -q'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add catan_mcts/async_mcts.py tests/test_async_mcts.py
git commit -m "feat(async-mcts): play_one_async_game full-game driver"
```

---

## Task 6: Per-game RNG reproducibility

**Files:**
- Test: `tests/test_async_mcts.py`

- [ ] **Step 1: Write the test — same seed -> same action sequence**

```python
# append to tests/test_async_mcts.py

async def test_per_game_rng_reproducible():
    from catan_mcts.async_mcts import play_one_async_game
    async def run():
        ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                                 max_batch=4, window_ms=5)
        ev.start()
        try:
            res = await play_one_async_game(
                game=CatanGame(vp_target=10, bonuses=True), seed=999,
                evaluator=ev, n_sims=8, rng=np.random.default_rng(999),
                max_steps=200000)
            return res.action_history
        finally:
            await ev.stop()
    h1 = await run()
    h2 = await run()
    assert h1 == h2, "same seed produced different play"
```

- [ ] **Step 2: Run test**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_async_mcts.py::test_per_game_rng_reproducible -x -q'`
Expected: PASS. If FAIL: shared RNG state is leaking — ensure `AsyncMcts.rng` and `play_one_async_game`'s `rng` are the SAME per-game generator and nothing draws from a module-global RNG.

- [ ] **Step 3: Commit**

```bash
git add tests/test_async_mcts.py
git commit -m "test(async-mcts): per-game RNG reproducibility under async batching"
```

---

## Task 7: Self-play orchestrator — N coroutines, recording, resume

**Files:**
- Create: `catan_mcts/experiments/self_play_async.py`
- Test: `tests/test_self_play_async.py`

- [ ] **Step 1: Write the failing tests — parquet output + resume**

```python
# tests/test_self_play_async.py
import numpy as np
import torch
import pandas as pd
from catan_gnn.gnn_model import GnnModel
from catan_mcts.experiments.self_play_async import run_self_play


def _save_ckpt(tmp_path):
    torch.manual_seed(0)
    m = GnnModel(hidden_dim=8, num_layers=2)
    p = tmp_path / "ckpt.pt"
    torch.save({"model_state": m.state_dict()}, p)
    return p


def test_self_play_writes_valid_parquet(tmp_path):
    ckpt = _save_ckpt(tmp_path)
    out = run_self_play(
        out_root=tmp_path / "runs", checkpoint=ckpt, num_games=4, n_sims=4,
        n_concurrent=4, hidden_dim=8, num_layers=2, vp_target=10, bonuses=True,
        device="cpu", max_batch=4, window_ms=5, seed_base=5_000_000)
    assert out.exists()
    parquets = list(out.rglob("games*.parquet"))
    assert parquets
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    assert len(df) == 4
    assert {"seed", "winner", "length_in_moves"}.issubset(df.columns)


def test_resume_skips_done_seeds(tmp_path):
    ckpt = _save_ckpt(tmp_path)
    common = dict(checkpoint=ckpt, n_sims=4, n_concurrent=2, hidden_dim=8,
                  num_layers=2, vp_target=10, bonuses=True, device="cpu",
                  max_batch=2, window_ms=5, seed_base=5_100_000)
    out_root = tmp_path / "runs"
    out = run_self_play(out_root=out_root, num_games=2, **common)
    out2 = run_self_play(out_root=out_root, num_games=4, resume_dir=out, **common)
    df = pd.concat([pd.read_parquet(p) for p in out2.rglob("games*.parquet")],
                   ignore_index=True)
    assert df["seed"].nunique() == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_self_play_async.py -x -q'`
Expected: FAIL — `ModuleNotFoundError: catan_mcts.experiments.self_play_async`.

- [ ] **Step 3: Write the orchestrator**

```python
# catan_mcts/experiments/self_play_async.py
"""Single-process asyncio self-play data generator.

Runs N game-coroutines concurrently against one BatchedGnnEvaluator, writes
e9-schema parquets via SelfPlayRecorder, resumable via done.txt. See spec
2026-05-30-batched-gnn-evaluator.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import torch

from catan_gnn.gnn_model import GnnModel
from ..adapter import CatanGame
from ..batched_evaluator import BatchedGnnEvaluator
from ..async_mcts import play_one_async_game
from ..recorder import SelfPlayRecorder
from .common import make_run_dir


def _load_model(checkpoint: Path, hidden_dim: int, num_layers: int, device: str):
    model = GnnModel(hidden_dim=hidden_dim, num_layers=num_layers)
    obj = torch.load(checkpoint, map_location=device, weights_only=False)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)
    return model.to(device).eval()


async def _play_and_record(*, game, seed, evaluator, n_sims, rec, sem, active):
    async with sem:
        active["n"] += 1
        evaluator.active_game_count = active["n"]
        try:
            result = await play_one_async_game(
                game=game, seed=seed, evaluator=evaluator, n_sims=n_sims,
                rng=np.random.default_rng(seed))
            with rec.game(seed=seed) as g_rec:
                for m in result.moves:
                    g_rec.record_move(
                        current_player=m.current_player, move_index=m.move_index,
                        legal_action_mask=m.legal_mask,
                        mcts_visit_counts=m.visit_counts,
                        action_taken=m.action_taken, mcts_root_value=m.root_value)
                g_rec.finalize(winner=result.winner, final_vp=result.final_vp,
                               length_in_moves=result.length_in_moves,
                               action_history=result.action_history,
                               timed_out=not result.terminal)
            rec.mark_done(seed)
        finally:
            active["n"] -= 1
            evaluator.active_game_count = max(1, active["n"])


async def _run_async(*, out, checkpoint, num_games, n_sims, n_concurrent,
                     hidden_dim, num_layers, vp_target, bonuses, device,
                     max_batch, window_ms, seed_base, resume,
                     ram_budget_mb, per_game_mb):
    if ram_budget_mb is not None:
        cap = max(1, int(ram_budget_mb / per_game_mb))
        if cap < n_concurrent:
            print(f"[self_play] concurrency capped: {n_concurrent} -> {cap} "
                  f"(ram_budget={ram_budget_mb}MB / {per_game_mb}MB per game)")
            n_concurrent = cap
    model = _load_model(checkpoint, hidden_dim, num_layers, device)
    evaluator = BatchedGnnEvaluator(model=model, device=device,
                                    max_batch=max_batch, window_ms=window_ms)
    evaluator.start()
    rec = SelfPlayRecorder(out, config={
        "experiment": "self_play_async", "n_sims": n_sims,
        "n_concurrent": n_concurrent, "vp_target": vp_target, "bonuses": bonuses,
        "max_batch": max_batch, "window_ms": window_ms, "device": device,
        "seed_base": seed_base})
    done = rec.done_seeds() if resume else set()
    sem = asyncio.Semaphore(n_concurrent)
    active = {"n": 0}
    game = CatanGame(vp_target=vp_target, bonuses=bonuses)
    seeds = [seed_base + i for i in range(num_games) if (seed_base + i) not in done]
    tasks = [_play_and_record(game=game, seed=s, evaluator=evaluator,
                              n_sims=n_sims, rec=rec, sem=sem, active=active)
             for s in seeds]
    await asyncio.gather(*tasks, return_exceptions=True)
    print(f"[self_play] done: {len(seeds)} games, "
          f"mean_batch={evaluator.mean_batch_size():.1f}, "
          f"total_batches={evaluator.total_batches}")
    await evaluator.stop()
    rec.flush()


def run_self_play(*, out_root: Path, checkpoint: Path, num_games: int = 64,
                  n_sims: int = 200, n_concurrent: int = 64,
                  hidden_dim: int = 128, num_layers: int = 4,
                  vp_target: int = 10, bonuses: bool = True, device: str = "cpu",
                  max_batch: int = 64, window_ms: float = 5.0,
                  max_seconds: float = 900.0, seed_base: int = 20_000_000,
                  resume_dir: Path | None = None,
                  ram_budget_mb: float | None = None,
                  per_game_mb: float = 50.0) -> Path:
    out = resume_dir if resume_dir is not None else make_run_dir(out_root, "self_play_async")
    asyncio.run(_run_async(
        out=out, checkpoint=checkpoint, num_games=num_games, n_sims=n_sims,
        n_concurrent=n_concurrent, hidden_dim=hidden_dim, num_layers=num_layers,
        vp_target=vp_target, bonuses=bonuses, device=device, max_batch=max_batch,
        window_ms=window_ms, seed_base=seed_base, resume=resume_dir is not None,
        ram_budget_mb=ram_budget_mb, per_game_mb=per_game_mb))
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--num-games", type=int, default=64)
    p.add_argument("--n-sims", type=int, default=200)
    p.add_argument("--n-concurrent", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--vp-target", type=int, default=10)
    p.add_argument("--bonuses", action="store_true", default=True)
    p.add_argument("--no-bonuses", dest="bonuses", action="store_false")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--window-ms", type=float, default=5.0)
    p.add_argument("--seed-base", type=int, default=20_000_000)
    p.add_argument("--ram-budget-mb", type=float, default=None)
    args = p.parse_args()
    out = run_self_play(
        out_root=args.out_root, checkpoint=args.checkpoint, num_games=args.num_games,
        n_sims=args.n_sims, n_concurrent=args.n_concurrent, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, vp_target=args.vp_target, bonuses=args.bonuses,
        device=args.device, max_batch=args.max_batch, window_ms=args.window_ms,
        seed_base=args.seed_base, ram_budget_mb=args.ram_budget_mb)
    print(f"self_play_async wrote to {out}")


if __name__ == "__main__":
    cli_main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_self_play_async.py -x -q'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add catan_mcts/experiments/self_play_async.py tests/test_self_play_async.py
git commit -m "feat(self-play): asyncio orchestrator with recording + resume"
```

---

## Task 8: Observability — mean-batch metric + memory-budget cap test

**Files:**
- Modify: `catan_mcts/batched_evaluator.py`
- Test: `tests/test_self_play_async.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_self_play_async.py

def test_mean_batch_size_reported(tmp_path, capsys):
    ckpt = _save_ckpt(tmp_path)
    run_self_play(out_root=tmp_path / "runs", checkpoint=ckpt, num_games=8,
                  n_sims=4, n_concurrent=8, hidden_dim=8, num_layers=2,
                  vp_target=10, bonuses=True, device="cpu", max_batch=8,
                  window_ms=5, seed_base=5_200_000)
    assert "mean_batch" in capsys.readouterr().out


def test_memory_budget_caps_concurrency(tmp_path, capsys):
    ckpt = _save_ckpt(tmp_path)
    run_self_play(out_root=tmp_path / "runs", checkpoint=ckpt, num_games=4,
                  n_sims=4, n_concurrent=1000, hidden_dim=8, num_layers=2,
                  vp_target=10, bonuses=True, device="cpu", max_batch=8,
                  window_ms=5, seed_base=5_300_000, ram_budget_mb=64)
    assert "concurrency capped" in capsys.readouterr().out.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_self_play_async.py -k "mean_batch or memory_budget" -x -q'`
Expected: `test_mean_batch_size_reported` FAILS (no `mean_batch_size` method on the evaluator → AttributeError in the print). `test_memory_budget` should already pass (the cap logic is in Task 7's orchestrator). If mean_batch already prints, this test passes too; the point is to add the method.

- [ ] **Step 3: Add the mean_batch_size method**

Add to `BatchedGnnEvaluator` in `catan_mcts/batched_evaluator.py`:

```python
    def mean_batch_size(self) -> float:
        return (self.total_requests / self.total_batches) if self.total_batches else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_self_play_async.py -x -q'`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add catan_mcts/batched_evaluator.py tests/test_self_play_async.py
git commit -m "feat(batched-eval): mean_batch_size health metric"
```

---

## Task 8b: Stuck-game watchdog (hardening #1)

The mean-batch metric reports aggregate health *after* a run. The watchdog
catches a *single stuck game mid-run* — a coroutine alive but not producing eval
requests (engine stall, non-awaiting loop), which silently degrades batches to
window-fired partials. Spec hardening #1.

**Files:**
- Modify: `catan_mcts/batched_evaluator.py`
- Test: `tests/test_batched_evaluator.py`

- [ ] **Step 1: Write the failing test — watchdog flags a stalled game**

```python
# append to tests/test_batched_evaluator.py

async def test_watchdog_flags_stuck_game(capsys):
    # active_game_count says 2 games alive, but only 1 ever enqueues a request.
    # The watchdog must log a warning naming the stall after K idle windows.
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=8, window_ms=5, watchdog_windows=3)
    ev.active_game_count = 2
    ev.start()
    try:
        # Only ONE request ever arrives; the "second game" never asks.
        await ev.eval(_leaf_state())
        # Give the watchdog a few windows to notice running>0 with tiny batches.
        await asyncio.sleep(0.1)
    finally:
        await ev.stop()
    out = capsys.readouterr().out.lower()
    assert "watchdog" in out and "stuck" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_batched_evaluator.py::test_watchdog_flags_stuck_game -x -q'`
Expected: FAIL — `BatchedGnnEvaluator.__init__` has no `watchdog_windows` param (TypeError), and no "watchdog" output.

- [ ] **Step 3: Add the watchdog**

In `BatchedGnnEvaluator.__init__`, add `watchdog_windows: int = 0` param and store `self.watchdog_windows = int(watchdog_windows)` plus a counter `self._idle_windows = 0`. In the batcher loop, after computing whether to flush, when the window fires with a partial batch AND `len(self._pending) < self.active_game_count` (i.e. some live game is NOT parked), increment `self._idle_windows`; reset it to 0 on a full or all-parked flush. When `self.watchdog_windows > 0 and self._idle_windows >= self.watchdog_windows`, print once:

```python
        print(f"[watchdog] stuck game suspected: {self.active_game_count - len(self._pending)} "
              f"live game(s) not parked across {self._idle_windows} windows "
              f"(mean_batch={self.mean_batch_size():.1f})")
        self._idle_windows = 0  # don't spam every window
```

- [ ] **Step 4: Run test to verify it passes**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_batched_evaluator.py -x -q'`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add catan_mcts/batched_evaluator.py tests/test_batched_evaluator.py
git commit -m "feat(batched-eval): stuck-game watchdog (hardening #1)"
```

Note: pass `watchdog_windows=10` from the orchestrator (`_run_async`) when constructing the evaluator so it's active in real runs but off (0) by default in unit tests that don't exercise it.

---

## Task 9: Full suite green + Gate 1 throughput probe

**Files:** none (validation) + journal

- [ ] **Step 1: Run all new tests**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_batched_evaluator.py tests/test_async_mcts.py tests/test_self_play_async.py -q'`
Expected: all PASS.

- [ ] **Step 2: Regression — existing tests still green**

Run: `wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m pytest tests/test_gnn_evaluator.py tests/test_e10e_gnn_mcts.py -q'`
Expected: all PASS (no existing code changed).

- [ ] **Step 3: Gate 1 throughput probe (GPU, real Cell 6 net)**

Run 16 games:
```bash
wsl.exe -d Ubuntu -- bash -c 'cd /mnt/c/dojo/catan_bot/.claude/worktrees/v4/mcts_study && ~/catan_mcts_venvs/mcts-study/bin/python -m catan_mcts.experiments.self_play_async --out-root /tmp/sp_probe --checkpoint /home/chitii/catan_data/runs/v3/training/loss_aug/06_cand11_cand8_cand10_h128_l4/training_h128_l4/checkpoint_epoch10.pt --num-games 16 --n-sims 200 --n-concurrent 16 --vp-target 10 --bonuses --device cuda --max-batch 16 --window-ms 5'
```
Record wall-clock, `mean_batch`, s/game.
**Gate 1 PASS:** mean_batch ≥ 8 (of 16) AND ≤ 24 s/game. If mean_batch ≈ 2: STOP and diagnose — likely the semaphore (`n_concurrent`) is below `max_batch`, or `active_game_count` isn't tracking, so games don't park together.

- [ ] **Step 4: Journal + commit**

Create `docs/superpowers/journals/2026-05-30-batched-eval-gate1.md` with measured mean_batch, s/game, and the speedup vs the 256 s/game single-worker baseline.
```bash
git add docs/superpowers/journals/2026-05-30-batched-eval-gate1.md
git commit -m "docs(batched-eval): Gate 1 throughput probe results"
```

---

## Task 10: Gate 2 — clean e10e re-run on the async stack

**Files:**
- Create: `catan_mcts/experiments/e10e_async.py`
- Journal: `docs/superpowers/journals/2026-05-30-e10e-clean-rerun.md`

- [ ] **Step 1: Write e10e_async by adapting e10e_gnn_mcts**

Create `catan_mcts/experiments/e10e_async.py` modeled on `e10e_gnn_mcts.py`. ONE change: the GnnMcts slot uses `AsyncMcts` + a shared `BatchedGnnEvaluator` instead of `os_mcts.MCTSBot`. Each game is a coroutine; the GnnMcts seat's turn does `await mcts.search(state, n_sims)` then `best_action`, while the two PureGnn seats and the LookV3 seat call their synchronous `.step(state)` directly. Run N games concurrently (one shared evaluator) so GnnMcts evals batch across games. Reuse the rotation + seating + recording from `e10e_gnn_mcts.py` verbatim. Provide a `--n-concurrent` arg and an async orchestrator like `self_play_async`'s.

- [ ] **Step 2: CPU smoke test (tiny params)**

Run with the real Cell6/Cell1 checkpoints, `--num-games-per-seating 1 --gnn-mcts-sims 4 --base-sims-v3 50 --device cpu --hidden-dim 128 --num-layers 4` into `/tmp/e10e_async_smoke`.
Expected: completes, writes a games parquet, no exceptions.

- [ ] **Step 3: Gate 2 run (GPU, 120 games)**

Run `--num-games-per-seating 30` (120 total), `--gnn-mcts-sims 200`, `--n-concurrent 32`, `--device cuda`, `--max-seconds 1800`, into `/home/chitii/catan_data/runs/v3/tournaments/e10e_async_120_2026_05_30`. Run in the background; watch `mean_batch` and timeout rate.

- [ ] **Step 4: Aggregate + verify Gate 2**

Map slot→role per rotation (same logic as the prior e10e aggregation: `_BASE_SEATING[rot:]+[:rot]`, seed encodes rotation via `(seed-seed_base)//10000`). Compute winrate by role.
**Gate 2 PASS:** timeout rate < 5% AND GnnMcts winrate consistent (within CI) with the prior bias-corrected finding (GNN+MCTS ≤ PureGnn). If the clean number CONTRADICTS the old one, document as a finding (old result was contention-driven, OR async MCTS diverges from OpenSpiel — flag for follow-up). Journal it in `docs/superpowers/journals/2026-05-30-e10e-clean-rerun.md`.

- [ ] **Step 5: Commit**

```bash
git add catan_mcts/experiments/e10e_async.py docs/superpowers/journals/2026-05-30-e10e-clean-rerun.md
git commit -m "feat(e10e-async): clean batched re-run of the GNN+MCTS diagnostic (Gate 2)"
```

---

## Done criteria

- All unit + integration tests green (Tasks 1–8).
- Gate 1: mean_batch ≥ 8/16 and ≤ 24 s/game (Task 9).
- Gate 2: timeout rate < 5%, GnnMcts winrate consistent with prior finding (Task 10).
- Both gate results journaled and committed.

The AlphaZero training loop is the next spec (out of scope here).
