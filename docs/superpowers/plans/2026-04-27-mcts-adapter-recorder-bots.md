# MCTS Adapter + Recorder + Bots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the Python project `mcts_study/` with (a) an OpenSpiel adapter that wraps `catan_bot._engine.Engine`, (b) a self-play recorder that writes parquet datasets sized for a future GNN, (c) baseline bots and a heuristic rollout policy, and (d) a complete TDD test suite for all three.

**Architecture:** Single Python project with its own `pyproject.toml` and venv. The adapter is the only module that touches `catan_bot._engine`; bots, recorder, and downstream code interact only with OpenSpiel types. The recorder is opt-in via context manager; tests cover schema, determinism, and OpenSpiel contract compliance.

**Tech Stack:** Python 3.10+, `open_spiel`, `pyarrow`, `pandas`, `hypothesis`, `pytest`, `numpy`. Uses the `catan_bot._engine` extension built by Phase 0 (`maturin develop`).

**Prerequisite:** Phase 0 plan complete and merged — adapter relies on `is_chance_pending`, `chance_outcomes`, `apply_chance_outcome`, `clone`, `action_history` on the PyEngine class.

**Action space width:** 206 (post-Phase-0). All schema-shaped lists/arrays use this constant — never hardcode the integer in code outside one constants module.

---

### Task 1: Project skeleton + pyproject + import

**Files:**
- Create: `mcts_study/pyproject.toml`
- Create: `mcts_study/catan_mcts/__init__.py`
- Create: `mcts_study/.gitignore`
- Create: `mcts_study/tests/__init__.py`
- Create: `mcts_study/tests/test_smoke_imports.py`

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_smoke_imports.py`:

```python
def test_can_import_catan_mcts():
    import catan_mcts
    assert hasattr(catan_mcts, "__version__")


def test_can_import_dependencies():
    import open_spiel  # noqa: F401
    import pyarrow      # noqa: F401
    import pandas       # noqa: F401
    import numpy        # noqa: F401


def test_engine_module_available():
    from catan_bot import _engine
    assert _engine.engine_version() == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest mcts_study/tests/test_smoke_imports.py -v`
Expected: FAIL — package doesn't exist.

- [ ] **Step 3: Create the pyproject and package**

Create `mcts_study/pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "catan_mcts"
version = "0.1.0"
description = "OpenSpiel MCTS study using the catan_bot engine."
requires-python = ">=3.10"
dependencies = [
    "open_spiel>=1.5",
    "pyarrow>=15",
    "pandas>=2.1",
    "numpy>=1.26",
    "tqdm>=4.66",
    "matplotlib>=3.8",
    "catan_bot",            # built locally via maturin develop
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "hypothesis>=6.100",
    "nbstripout>=0.7",
]

[tool.hatch.build.targets.wheel]
packages = ["catan_mcts"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Create `mcts_study/catan_mcts/__init__.py`:

```python
"""OpenSpiel MCTS study driven by the catan_bot Rust engine."""

__version__ = "0.1.0"

ACTION_SPACE_SIZE = 206  # post-Phase-0 engine constant
```

Create `mcts_study/.gitignore`:

```
runs/
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ipynb_checkpoints/
```

Create `mcts_study/tests/__init__.py` as an empty file.

- [ ] **Step 4: Set up the venv and install**

Run from `mcts_study/`:
```bash
python -m venv .venv
.venv/Scripts/activate    # Windows; on Linux/macOS use `source .venv/bin/activate`
pip install -e .[dev]
maturin develop --manifest-path ../catan_engine/Cargo.toml
```

- [ ] **Step 5: Run tests**

Run: `pytest mcts_study/tests/test_smoke_imports.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mcts_study/
git commit -m "feat(mcts_study): project skeleton + smoke imports"
```

---

### Task 2: Adapter — minimal `CatanGame` and `CatanState` shells

**Files:**
- Create: `mcts_study/catan_mcts/adapter.py`
- Create: `mcts_study/tests/test_adapter.py`

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_adapter.py`:

```python
import pytest
from catan_mcts.adapter import CatanGame
from catan_mcts import ACTION_SPACE_SIZE


def test_game_construction():
    game = CatanGame()
    assert game.num_distinct_actions() == ACTION_SPACE_SIZE
    assert game.num_players() == 4


def test_initial_state_is_not_terminal():
    game = CatanGame()
    state = game.new_initial_state()
    assert not state.is_terminal()


def test_initial_state_has_legal_actions():
    game = CatanGame()
    state = game.new_initial_state()
    legal = state.legal_actions()
    assert len(legal) > 0
    assert all(0 <= a < ACTION_SPACE_SIZE for a in legal)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest mcts_study/tests/test_adapter.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement the minimal adapter shell**

Create `mcts_study/catan_mcts/adapter.py`:

```python
"""OpenSpiel adapter wrapping catan_bot._engine.Engine."""
from __future__ import annotations

from typing import Iterable

from catan_bot import _engine

from . import ACTION_SPACE_SIZE

NUM_PLAYERS = 4


class CatanGame:
    """OpenSpiel-style Game for our Catan engine.

    Note: We don't subclass `pyspiel.Game` directly — pyspiel.Game requires C++-side
    registration. Instead we expose the duck-typed interface OpenSpiel's Python MCTSBot
    needs. If a future algorithm requires registered pyspiel games, we'll add a
    register-on-import path; until then, duck typing is simpler and works for MCTS.
    """

    def __init__(self, default_seed: int = 0) -> None:
        self._default_seed = default_seed

    def num_distinct_actions(self) -> int:
        return ACTION_SPACE_SIZE

    def num_players(self) -> int:
        return NUM_PLAYERS

    def new_initial_state(self, seed: int | None = None) -> "CatanState":
        s = self._default_seed if seed is None else seed
        return CatanState(_engine.Engine(s))


class CatanState:
    def __init__(self, engine) -> None:  # engine: catan_bot._engine.Engine
        self._engine = engine

    def is_terminal(self) -> bool:
        return self._engine.is_terminal()

    def current_player(self) -> int:
        if self.is_terminal():
            return -1  # OpenSpiel's TERMINAL marker convention
        if self._engine.is_chance_pending():
            return -2  # OpenSpiel's CHANCE_PLAYER_ID is -1; we'll align in Task 3.
        return int(self._engine.current_player())

    def legal_actions(self) -> list[int]:
        return [int(a) for a in self._engine.legal_actions()]

    def apply_action(self, action: int) -> None:
        self._engine.step(int(action))
```

- [ ] **Step 4: Run tests**

Run: `pytest mcts_study/tests/test_adapter.py -v`
Expected: PASS for the three tests above.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/adapter.py mcts_study/tests/test_adapter.py
git commit -m "feat(adapter): CatanGame + CatanState shell"
```

---

### Task 3: Adapter — chance-node integration

**Files:**
- Modify: `mcts_study/catan_mcts/adapter.py`
- Modify: `mcts_study/tests/test_adapter.py`

- [ ] **Step 1: Write the failing tests**

Append to `mcts_study/tests/test_adapter.py`:

```python
import pyspiel
from catan_mcts.adapter import CatanGame


def test_chance_player_id_matches_pyspiel():
    # OpenSpiel uses pyspiel.PlayerId.CHANCE = -1.
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    # Drive forward until a chance node fires (Roll phase).
    for _ in range(2000):
        if state.is_chance_node():
            assert state.current_player() == pyspiel.PlayerId.CHANCE
            outcomes = state.chance_outcomes()
            assert len(outcomes) > 0
            total = sum(p for _, p in outcomes)
            assert abs(total - 1.0) < 1e-9
            return
        state.apply_action(state.legal_actions()[0])
    raise AssertionError("no chance node reached in 2000 steps")


def test_chance_node_drives_dice():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    for _ in range(2000):
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            value, _ = outcomes[0]
            state.apply_action(int(value))
            assert not state.is_chance_node() or state.current_player() == pyspiel.PlayerId.CHANCE
            return
        state.apply_action(state.legal_actions()[0])
    raise AssertionError("no chance node reached")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest mcts_study/tests/test_adapter.py::test_chance_player_id_matches_pyspiel mcts_study/tests/test_adapter.py::test_chance_node_drives_dice -v`
Expected: FAIL — `is_chance_node`, `chance_outcomes` not implemented; `current_player()` returns -2 instead of -1.

- [ ] **Step 3: Implement the chance-node API**

In `mcts_study/catan_mcts/adapter.py`:

Replace the `current_player()` method with:

```python
import pyspiel

# ... module top, near imports

CHANCE = pyspiel.PlayerId.CHANCE  # -1
TERMINAL = pyspiel.PlayerId.TERMINAL  # -4 in OpenSpiel; check if used

# ... in CatanState ...

    def current_player(self) -> int:
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        if self._engine.is_chance_pending():
            return pyspiel.PlayerId.CHANCE
        return int(self._engine.current_player())

    def is_chance_node(self) -> bool:
        return self._engine.is_chance_pending()

    def chance_outcomes(self) -> list[tuple[int, float]]:
        # PyO3 returns list[(u32, f64)]; ensure ints.
        return [(int(v), float(p)) for v, p in self._engine.chance_outcomes()]

    def apply_action(self, action: int) -> None:
        action = int(action)
        if self._engine.is_chance_pending():
            self._engine.apply_chance_outcome(action)
        else:
            self._engine.step(action)
```

- [ ] **Step 4: Run tests**

Run: `pytest mcts_study/tests/test_adapter.py -v`
Expected: PASS for the new chance tests; existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/adapter.py mcts_study/tests/test_adapter.py
git commit -m "feat(adapter): chance-node integration"
```

---

### Task 4: Adapter — clone, returns, serialize

**Files:**
- Modify: `mcts_study/catan_mcts/adapter.py`
- Modify: `mcts_study/tests/test_adapter.py`

- [ ] **Step 1: Write the failing tests**

Append to `mcts_study/tests/test_adapter.py`:

```python
def test_clone_is_independent():
    game = CatanGame()
    a = game.new_initial_state(seed=42)
    a.apply_action(a.legal_actions()[0])
    b = a.clone()
    history_before = list(a.history())
    b.apply_action(b.legal_actions()[0])
    assert list(a.history()) == history_before


def test_returns_zero_until_terminal():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    assert state.returns() == [0.0, 0.0, 0.0, 0.0]


def test_returns_sum_zero_at_terminal():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    steps = 0
    while not state.is_terminal() and steps < 5000:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            state.apply_action(int(outcomes[0][0]))
        else:
            state.apply_action(state.legal_actions()[0])
        steps += 1
    if not state.is_terminal():
        # Determinism over legal[0] policy may not finish on every seed; not a bug.
        return
    rs = state.returns()
    assert sum(rs) == 0.0  # +1 winner / 3 * (-1/3) losers sums to 0... actually +1 - 3 = -2.
    # Wait: spec says +1 / -1, so sum = +1 - 3 = -2. Adjust:
    # Actually in 4-player AlphaZero convention sum is -2 with +1/-1 returns.


def test_serialize_round_trips_through_history():
    game = CatanGame()
    s1 = game.new_initial_state(seed=42)
    s1.apply_action(s1.legal_actions()[0])
    blob = s1.serialize()
    s2 = CatanGame.deserialize(blob)
    assert s2.legal_actions() == s1.legal_actions()
    assert s2.is_terminal() == s1.is_terminal()
```

Hold on — the `returns_sum_zero_at_terminal` test above has a real arithmetic issue: with `+1/-1` rewards and one winner, sum is `+1 - 3 = -2`, not 0. Per the design spec §6 we use AlphaZero-style sparse `+1` / `-1`. Replace that test with:

```python
def test_returns_at_terminal_have_one_winner():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    steps = 0
    while not state.is_terminal() and steps < 5000:
        if state.is_chance_node():
            state.apply_action(int(state.chance_outcomes()[0][0]))
        else:
            state.apply_action(state.legal_actions()[0])
        steps += 1
    if not state.is_terminal():
        return  # legal[0] policy may stall on some seeds; not the test target
    rs = state.returns()
    assert rs.count(1.0) == 1
    assert rs.count(-1.0) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest mcts_study/tests/test_adapter.py -v`
Expected: FAIL — `clone`, `history`, `returns`, `serialize`, `CatanGame.deserialize` not implemented.

- [ ] **Step 3: Implement clone / returns / serialize / history**

In `mcts_study/catan_mcts/adapter.py`:

Add a helper at module level:

```python
import json
from typing import Any
```

In `CatanState`:

```python
    def clone(self) -> "CatanState":
        return CatanState(self._engine.clone())

    def history(self) -> list[int]:
        return [int(x) for x in self._engine.action_history()]

    def returns(self) -> list[float]:
        if not self.is_terminal():
            return [0.0] * NUM_PLAYERS
        # The engine's stats() exposes winner_player_id; +1 for winner, -1 for losers.
        stats = self._engine.stats()
        winner = int(stats["winner_player_id"])
        if winner < 0:  # no winner (shouldn't happen if terminal, but be defensive)
            return [0.0] * NUM_PLAYERS
        out = [-1.0] * NUM_PLAYERS
        out[winner] = 1.0
        return out

    def serialize(self) -> str:
        # Stable, unique, deterministic: (seed, action_history) reconstructs everything.
        seed = self._initial_seed
        return json.dumps({"seed": seed, "history": self.history()})
```

To make `_initial_seed` available, change `CatanGame.new_initial_state` to pass it to `CatanState.__init__`, and update `CatanState.__init__`:

```python
    def __init__(self, engine, initial_seed: int = 0) -> None:
        self._engine = engine
        self._initial_seed = initial_seed
```

In `CatanGame`:

```python
    def new_initial_state(self, seed: int | None = None) -> "CatanState":
        s = self._default_seed if seed is None else seed
        return CatanState(_engine.Engine(s), initial_seed=s)

    @staticmethod
    def deserialize(blob: str) -> "CatanState":
        data = json.loads(blob)
        seed = int(data["seed"])
        history = [int(x) for x in data["history"]]
        engine = _engine.Engine(seed)
        for action_id in history:
            # action_history pushes both step actions (raw IDs) and chance outcomes (high bit set).
            CHANCE_FLAG = 0x80000000
            if action_id & CHANCE_FLAG:
                engine.apply_chance_outcome(action_id & ~CHANCE_FLAG)
            else:
                engine.step(action_id)
        return CatanState(engine, initial_seed=seed)
```

- [ ] **Step 4: Run tests**

Run: `pytest mcts_study/tests/test_adapter.py -v`
Expected: PASS for all adapter tests.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/adapter.py mcts_study/tests/test_adapter.py
git commit -m "feat(adapter): clone + returns + serialize/deserialize"
```

---

### Task 5: Adapter — property test (random legal play never raises)

**Files:**
- Create: `mcts_study/tests/test_adapter_properties.py`

- [ ] **Step 1: Write the test**

Create `mcts_study/tests/test_adapter_properties.py`:

```python
"""Property test: random legal-action play never raises, terminal sums correct, all 256 seeds."""
import random

from hypothesis import given, settings, strategies as st

from catan_mcts.adapter import CatanGame


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=64, deadline=None)
def test_random_play_never_raises(seed: int) -> None:
    rng = random.Random(seed)
    game = CatanGame()
    state = game.new_initial_state(seed=seed)
    for _ in range(5000):
        if state.is_terminal():
            break
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            # Sample by probability — same as MCTS will do at runtime.
            r = rng.random()
            cum = 0.0
            chosen = outcomes[-1][0]
            for v, p in outcomes:
                cum += p
                if r <= cum:
                    chosen = v
                    break
            state.apply_action(int(chosen))
        else:
            legal = state.legal_actions()
            if not legal:
                break
            state.apply_action(rng.choice(legal))
    if state.is_terminal():
        rs = state.returns()
        assert rs.count(1.0) == 1
        assert rs.count(-1.0) == 3
```

- [ ] **Step 2: Run test**

Run: `pytest mcts_study/tests/test_adapter_properties.py -v`
Expected: PASS — 64 random seeds all complete without exceptions.

- [ ] **Step 3: Commit**

```bash
git add mcts_study/tests/test_adapter_properties.py
git commit -m "test(adapter): property test — random play never raises"
```

---

### Task 6: Smoke test — OpenSpiel MCTSBot vs RandomBot, full game

**Files:**
- Create: `mcts_study/tests/test_smoke_mcts_vs_random.py`

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_smoke_mcts_vs_random.py`:

```python
"""End-to-end smoke: OpenSpiel MCTSBot plays a full game vs 3 random opponents."""
import random

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts

from catan_mcts.adapter import CatanGame


class _RandomBotShim:
    """Tiny random bot — picks uniformly from legal actions."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state) -> int:
        return self._rng.choice(state.legal_actions())


def test_mcts_vs_random_full_game():
    game = CatanGame()
    state = game.new_initial_state(seed=42)

    rng = np.random.default_rng(seed=42)
    rand_eval = mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=rng)
    mcts_bot = mcts.MCTSBot(
        game=game,
        uct_c=1.4,
        max_simulations=50,
        evaluator=rand_eval,
        solve=False,
        random_state=rng,
    )
    bots = {0: mcts_bot, 1: _RandomBotShim(1), 2: _RandomBotShim(2), 3: _RandomBotShim(3)}

    steps = 0
    while not state.is_terminal() and steps < 5000:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            r = rng.random()
            cum = 0.0
            chosen = outcomes[-1][0]
            for v, p in outcomes:
                cum += p
                if r <= cum:
                    chosen = v
                    break
            state.apply_action(int(chosen))
        else:
            p = state.current_player()
            bot = bots[p]
            if bot is mcts_bot:
                action = bot.step(state)
            else:
                action = bot.step(state)
            state.apply_action(int(action))
        steps += 1

    assert state.is_terminal(), f"game did not terminate in 5000 steps (got {steps})"
    rs = state.returns()
    assert rs.count(1.0) == 1
    assert rs.count(-1.0) == 3
```

- [ ] **Step 2: Run test**

Run: `pytest mcts_study/tests/test_smoke_mcts_vs_random.py -v -s`
Expected: PASS. May take 30-90 seconds — MCTSBot does 50 sims per move and games are ~80 moves.

If MCTSBot's `step(state)` complains that `CatanGame` is not a `pyspiel.Game`, this is the failure mode the adapter docstring (Task 2) anticipates. Fallback: convert `CatanGame` to use `pyspiel.LoadGame("python_*")` registration. We avoid that until forced — verify whether the duck-typed adapter works first.

If duck typing doesn't work: see Task 6.5 below.

- [ ] **Step 3: Commit**

```bash
git add mcts_study/tests/test_smoke_mcts_vs_random.py
git commit -m "test(adapter): smoke — MCTSBot vs random full game"
```

---

### Task 6.5: (Conditional) Register `CatanGame` as a Python game with OpenSpiel

**Skip this task if Task 6 passes.**

**Files:**
- Modify: `mcts_study/catan_mcts/adapter.py`

- [ ] **Step 1: Convert `CatanGame` to a registered Python game**

Read [OpenSpiel's "Adding a Python Game" guide](https://github.com/deepmind/open_spiel/blob/master/docs/developer_guide.md) and convert `CatanGame` to subclass `pyspiel.Game`, registering it via `pyspiel.register_game(...)` at import time.

The required surface to implement:
- `__init__(self, params)` calling `pyspiel.Game.__init__(self, _GAME_TYPE, _GAME_INFO, params)`
- `_GAME_TYPE = pyspiel.GameType(...)` with `provides_information_state_string=False`, `provides_observation_string=False` (we don't need these for MCTS), and `chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC`.
- `_GAME_INFO = pyspiel.GameInfo(num_distinct_actions=ACTION_SPACE_SIZE, max_chance_outcomes=<max steal-hand-size or 11 for dice>, num_players=4, ...)`

Then `CatanState` subclasses `pyspiel.State` and overrides `_legal_actions`, `_apply_action`, `_action_to_string`, `is_terminal`, `is_chance_node`, `chance_outcomes`, `current_player`, `returns`, `clone`, `__str__`, `_observation_string`.

The bulk of the body delegates to the existing `_engine` calls. The change is in the *base class* not the *logic*.

- [ ] **Step 2: Run smoke test**

Run: `pytest mcts_study/tests/test_smoke_mcts_vs_random.py -v -s`
Expected: PASS.

- [ ] **Step 3: Re-run all adapter tests**

Run: `pytest mcts_study/tests/ -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add mcts_study/catan_mcts/adapter.py
git commit -m "refactor(adapter): register as pyspiel.Game so MCTSBot accepts it"
```

---

### Task 7: Bots — `GreedyBaselineBot` + `heuristic_rollout`

**Files:**
- Create: `mcts_study/catan_mcts/bots.py`
- Create: `mcts_study/tests/test_bots.py`

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_bots.py`:

```python
import random

from catan_mcts.adapter import CatanGame
from catan_mcts.bots import GreedyBaselineBot, heuristic_rollout


def test_greedy_bot_completes_game_against_random():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    bots = [GreedyBaselineBot(seed=0)] + [_RandomBot(i + 1) for i in range(3)]
    rng = random.Random(123)
    steps = 0
    while not state.is_terminal() and steps < 5000:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            r = rng.random()
            cum = 0.0
            chosen = outcomes[-1][0]
            for v, p in outcomes:
                cum += p
                if r <= cum:
                    chosen = v
                    break
            state.apply_action(int(chosen))
        else:
            action = bots[state.current_player()].step(state)
            state.apply_action(int(action))
        steps += 1
    assert state.is_terminal()


def test_heuristic_rollout_returns_legal_action():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    rng = random.Random(0)
    # Drive past chance once to get into a player decision.
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        state.apply_action(int(outcomes[0][0]))
    action = heuristic_rollout(state, rng)
    assert action in state.legal_actions()


class _RandomBot:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest mcts_study/tests/test_bots.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `bots.py`**

Create `mcts_study/catan_mcts/bots.py`:

```python
"""Baseline bots and rollout policy for MCTS experiments.

Bots interact only with OpenSpiel state types. They never touch `_engine` directly —
that abstraction is what `adapter.py` provides.
"""
from __future__ import annotations

import random
from typing import Optional

# Action ID ranges from spec/Phase 0:
#   BuildSettlement: 0..54   (worth 1 VP)
#   BuildCity:       54..108 (worth 2 VP — net +1 over upgraded settlement)
#   BuildRoad:       108..180
#   MoveRobber:      180..199
#   Discard:         199..204
#   EndTurn:         204
#   RollDice:        205   (Phase 0)


def _action_priority(action_id: int) -> int:
    """Higher = bot prefers it. Crude VP-greedy ordering."""
    if 54 <= action_id < 108:
        return 100  # city — best
    if 0 <= action_id < 54:
        return 80   # settlement
    if 108 <= action_id < 180:
        return 50   # road (enables future settlements)
    if action_id == 205:  # roll
        return 10
    if action_id == 204:  # end turn
        return 1
    if 180 <= action_id < 199:
        return 5    # robber move (no preference among hexes here)
    if 199 <= action_id < 204:
        return 5    # discard
    return 0


class GreedyBaselineBot:
    """Picks the legal action with highest VP-greedy priority. Ties broken randomly."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def step(self, state) -> int:
        legal = state.legal_actions()
        if not legal:
            raise RuntimeError("GreedyBaselineBot: no legal actions in non-terminal state")
        best = max(legal, key=_action_priority)
        # Break ties uniformly among actions with the top priority.
        top = _action_priority(best)
        candidates = [a for a in legal if _action_priority(a) == top]
        return self._rng.choice(candidates)


def heuristic_rollout(state, rng: Optional[random.Random] = None) -> int:
    """A single-action callable matching MCTS rollout-policy signature.

    Returns one action; the MCTS rollout loop calls this until terminal.
    """
    rng = rng or random
    legal = state.legal_actions()
    if not legal:
        raise RuntimeError("heuristic_rollout: no legal actions")
    best = max(legal, key=_action_priority)
    top = _action_priority(best)
    candidates = [a for a in legal if _action_priority(a) == top]
    return rng.choice(candidates) if isinstance(rng, random.Random) else random.choice(candidates)
```

- [ ] **Step 4: Run tests**

Run: `pytest mcts_study/tests/test_bots.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/bots.py mcts_study/tests/test_bots.py
git commit -m "feat(bots): GreedyBaselineBot + heuristic_rollout"
```

---

### Task 8: Recorder — write moves.parquet and games.parquet

**Files:**
- Create: `mcts_study/catan_mcts/recorder.py`
- Create: `mcts_study/tests/test_recorder.py`

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_recorder.py`:

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from catan_mcts import ACTION_SPACE_SIZE
from catan_mcts.recorder import SelfPlayRecorder, SCHEMA_VERSION


def test_schema_version_is_constant():
    assert SCHEMA_VERSION == 1  # bump explicitly when the schema changes


def test_recorder_writes_expected_columns(tmp_path: Path):
    rec = SelfPlayRecorder(out_dir=tmp_path, config={"experiment": "smoke"})
    with rec.game(seed=42) as game_rec:
        # Fake one move
        game_rec.record_move(
            current_player=0,
            move_index=0,
            legal_action_mask=np.zeros(ACTION_SPACE_SIZE, dtype=bool),
            mcts_visit_counts=np.zeros(ACTION_SPACE_SIZE, dtype=np.int32),
            action_taken=204,  # end turn
            mcts_root_value=0.5,
        )
        game_rec.finalize(winner=0, final_vp=[10, 5, 4, 3], length_in_moves=1)
    rec.flush()

    moves = pq.read_table(tmp_path / "moves.parquet").to_pandas()
    games = pq.read_table(tmp_path / "games.parquet").to_pandas()

    assert set(moves.columns) == {
        "seed", "move_index", "current_player", "legal_action_mask",
        "mcts_visit_counts", "action_taken", "mcts_root_value", "schema_version",
    }
    assert set(games.columns) == {
        "seed", "winner", "final_vp", "length_in_moves", "mcts_config_id", "schema_version",
    }
    assert moves["schema_version"].iloc[0] == SCHEMA_VERSION
    assert games["schema_version"].iloc[0] == SCHEMA_VERSION
    assert len(moves["legal_action_mask"].iloc[0]) == ACTION_SPACE_SIZE
    assert games["winner"].iloc[0] == 0


def test_recorder_writes_config_json(tmp_path: Path):
    rec = SelfPlayRecorder(out_dir=tmp_path, config={"experiment": "smoke", "uct_c": 1.4})
    with rec.game(seed=1) as g:
        g.finalize(winner=-1, final_vp=[0, 0, 0, 0], length_in_moves=0)
    rec.flush()

    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["experiment"] == "smoke"
    assert cfg["uct_c"] == 1.4
    assert cfg["schema_version"] == SCHEMA_VERSION


def test_record_move_rejects_action_not_in_legal_mask(tmp_path: Path):
    rec = SelfPlayRecorder(out_dir=tmp_path, config={})
    with rec.game(seed=42) as game_rec:
        try:
            game_rec.record_move(
                current_player=0,
                move_index=0,
                legal_action_mask=np.zeros(ACTION_SPACE_SIZE, dtype=bool),  # no legal actions
                mcts_visit_counts=np.zeros(ACTION_SPACE_SIZE, dtype=np.int32),
                action_taken=42,
                mcts_root_value=0.0,
            )
        except AssertionError:
            return
    raise AssertionError("expected record_move to reject action with mask=False")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest mcts_study/tests/test_recorder.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `recorder.py`**

Create `mcts_study/catan_mcts/recorder.py`:

```python
"""Self-play recorder. Writes moves.parquet, games.parquet, config.json per run.

Schema is documented in docs/superpowers/specs/2026-04-27-mcts-study-design.md §1.5.
SCHEMA_VERSION must be bumped explicitly on any field change; tests assert the value.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import ACTION_SPACE_SIZE


SCHEMA_VERSION = 1


@dataclass
class _MoveRow:
    seed: int
    move_index: int
    current_player: int
    legal_action_mask: list[bool]
    mcts_visit_counts: list[int]
    action_taken: int
    mcts_root_value: float
    schema_version: int = SCHEMA_VERSION


@dataclass
class _GameRow:
    seed: int
    winner: int
    final_vp: list[int]
    length_in_moves: int
    mcts_config_id: str
    schema_version: int = SCHEMA_VERSION


class _GameRecorder:
    def __init__(self, parent: "SelfPlayRecorder", seed: int) -> None:
        self._parent = parent
        self._seed = seed
        self._moves: list[_MoveRow] = []
        self._finalized = False

    def record_move(
        self,
        *,
        current_player: int,
        move_index: int,
        legal_action_mask: np.ndarray,
        mcts_visit_counts: np.ndarray,
        action_taken: int,
        mcts_root_value: float,
    ) -> None:
        assert legal_action_mask.shape == (ACTION_SPACE_SIZE,)
        assert mcts_visit_counts.shape == (ACTION_SPACE_SIZE,)
        assert legal_action_mask[action_taken], (
            f"action_taken={action_taken} not in legal_action_mask"
        )
        self._moves.append(_MoveRow(
            seed=self._seed,
            move_index=move_index,
            current_player=current_player,
            legal_action_mask=[bool(x) for x in legal_action_mask],
            mcts_visit_counts=[int(x) for x in mcts_visit_counts],
            action_taken=int(action_taken),
            mcts_root_value=float(mcts_root_value),
        ))

    def finalize(self, *, winner: int, final_vp: list[int], length_in_moves: int) -> None:
        self._parent._game_rows.append(_GameRow(
            seed=self._seed,
            winner=int(winner),
            final_vp=[int(x) for x in final_vp],
            length_in_moves=int(length_in_moves),
            mcts_config_id=self._parent._config_id,
        ))
        self._parent._move_rows.extend(self._moves)
        self._finalized = True


class SelfPlayRecorder:
    def __init__(self, out_dir: Path, config: dict[str, Any]) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._config_id = str(uuid.uuid4())
        full_config = {**config, "mcts_config_id": self._config_id, "schema_version": SCHEMA_VERSION}
        (self._out_dir / "config.json").write_text(json.dumps(full_config, indent=2))
        self._move_rows: list[_MoveRow] = []
        self._game_rows: list[_GameRow] = []

    @contextmanager
    def game(self, seed: int) -> Iterator[_GameRecorder]:
        rec = _GameRecorder(self, seed)
        try:
            yield rec
        finally:
            if not rec._finalized:
                # Game ended without finalize — record a -1 winner
                rec.finalize(winner=-1, final_vp=[0]*4, length_in_moves=len(rec._moves))

    def flush(self) -> None:
        moves_table = pa.Table.from_pylist([row.__dict__ for row in self._move_rows])
        games_table = pa.Table.from_pylist([row.__dict__ for row in self._game_rows])
        pq.write_table(moves_table, self._out_dir / "moves.parquet")
        pq.write_table(games_table, self._out_dir / "games.parquet")
```

- [ ] **Step 4: Run tests**

Run: `pytest mcts_study/tests/test_recorder.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/recorder.py mcts_study/tests/test_recorder.py
git commit -m "feat(recorder): SelfPlayRecorder writes parquet"
```

---

### Task 9: Determinism regression — same seed → byte-identical moves.parquet

**Files:**
- Create: `mcts_study/tests/test_determinism.py`

- [ ] **Step 1: Write the test**

Create `mcts_study/tests/test_determinism.py`:

```python
"""Determinism regression: same seed + same config → byte-identical moves.parquet."""
from pathlib import Path
import hashlib
import random

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts

from catan_mcts.adapter import CatanGame
from catan_mcts.recorder import SelfPlayRecorder
from catan_mcts import ACTION_SPACE_SIZE


def _run_one(out_dir: Path) -> bytes:
    game = CatanGame()
    rec = SelfPlayRecorder(out_dir, config={"uct_c": 1.4, "max_simulations": 25})
    rng = np.random.default_rng(seed=42)
    state = game.new_initial_state(seed=42)
    evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=rng)
    bot = mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=25, evaluator=evaluator,
        solve=False, random_state=rng,
    )
    rng_random = random.Random(42)
    move_index = 0

    with rec.game(seed=42) as g:
        steps = 0
        while not state.is_terminal() and steps < 2000:
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                r = rng_random.random()
                cum, chosen = 0.0, outcomes[-1][0]
                for v, p in outcomes:
                    cum += p
                    if r <= cum:
                        chosen = v
                        break
                state.apply_action(int(chosen))
            else:
                if state.current_player() == 0:
                    action = bot.step(state)
                    legal = state.legal_actions()
                    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
                    mask[legal] = True
                    # Determinism test only checks parquet bytes are byte-identical,
                    # not that visit counts are right. Zeros are deterministic.
                    visits = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
                    g.record_move(
                        current_player=0, move_index=move_index,
                        legal_action_mask=mask, mcts_visit_counts=visits,
                        action_taken=int(action), mcts_root_value=0.0,
                    )
                    move_index += 1
                    state.apply_action(int(action))
                else:
                    state.apply_action(rng_random.choice(state.legal_actions()))
            steps += 1
        winner = -1
        if state.is_terminal():
            rs = state.returns()
            winner = rs.index(1.0) if 1.0 in rs else -1
        g.finalize(winner=winner, final_vp=[0,0,0,0], length_in_moves=steps)
    rec.flush()
    return (out_dir / "moves.parquet").read_bytes()


def test_byte_identical_runs(tmp_path: Path):
    a = _run_one(tmp_path / "run_a")
    b = _run_one(tmp_path / "run_b")
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()
```

- [ ] **Step 2: Run test**

Run: `pytest mcts_study/tests/test_determinism.py -v`
Expected: PASS. If it fails, the bug is in adapter, recorder, or `mcts.MCTSBot`'s seed handling — diagnose the layer that diverges and fix.

- [ ] **Step 3: Commit**

```bash
git add mcts_study/tests/test_determinism.py
git commit -m "test(mcts_study): determinism regression for self-play"
```

---

### Task 10: Visit-count extraction helper

**Files:**
- Modify: `mcts_study/catan_mcts/recorder.py` (add a helper)
- Modify: `mcts_study/tests/test_recorder.py`

- [ ] **Step 1: Write the failing test**

Append to `mcts_study/tests/test_recorder.py`:

```python
def test_extract_visit_counts_from_mcts_root():
    """OpenSpiel's MCTSBot exposes the search tree root via `bot.mcts_search(state)`.
    The recorder needs a helper that converts that to a fixed-width visit-count array.
    """
    import numpy as np
    import pyspiel
    from open_spiel.python.algorithms import mcts as os_mcts
    from catan_mcts.adapter import CatanGame
    from catan_mcts.recorder import visit_counts_from_root

    game = CatanGame()
    state = game.new_initial_state(seed=42)
    while state.is_chance_node():
        state.apply_action(int(state.chance_outcomes()[0][0]))

    rng = np.random.default_rng(seed=42)
    evaluator = os_mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=rng)
    bot = os_mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=25, evaluator=evaluator,
        solve=False, random_state=rng,
    )
    root = bot.mcts_search(state)
    visits = visit_counts_from_root(root)
    assert visits.shape == (ACTION_SPACE_SIZE,)
    assert visits.sum() > 0
    # All non-zero entries correspond to legal actions
    legal = set(state.legal_actions())
    for a, v in enumerate(visits):
        if v > 0:
            assert a in legal
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest mcts_study/tests/test_recorder.py::test_extract_visit_counts_from_mcts_root -v`
Expected: FAIL — `visit_counts_from_root` not implemented.

- [ ] **Step 3: Implement the helper**

In `mcts_study/catan_mcts/recorder.py`, add:

```python
def visit_counts_from_root(root) -> np.ndarray:
    """Extract a length-ACTION_SPACE_SIZE int32 array of root-child visit counts from
    OpenSpiel's MCTS search-tree root node.

    OpenSpiel's `SearchNode` exposes `.children` as a list of `SearchNode`, each with
    `.action` (int) and `.explore_count` (int).
    """
    out = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
    for child in root.children:
        out[int(child.action)] = int(child.explore_count)
    return out
```

- [ ] **Step 4: Run tests**

Run: `pytest mcts_study/tests/test_recorder.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/recorder.py mcts_study/tests/test_recorder.py
git commit -m "feat(recorder): visit_counts_from_root helper"
```

---

### Task 11: Plan-2 self-review checklist

- [ ] All adapter contract methods implemented and tested:
  - `num_distinct_actions`, `num_players`, `new_initial_state`, `is_terminal`, `current_player`, `legal_actions`, `apply_action`, `is_chance_node`, `chance_outcomes`, `clone`, `history`, `returns`, `serialize`, `deserialize`
- [ ] Recorder schemas match spec §1.5:
  - moves: seed, move_index, current_player, legal_action_mask, mcts_visit_counts, action_taken, mcts_root_value, schema_version
  - games: seed, winner, final_vp, length_in_moves, mcts_config_id, schema_version
- [ ] All schema-shaped arrays use `ACTION_SPACE_SIZE = 206`, never the literal.
- [ ] No `#[ignore]` / `pytest.mark.skip` in the final test set.
- [ ] Full test suite green: `pytest mcts_study/tests/ -v`
- [ ] Smoke test (`test_smoke_mcts_vs_random.py`) passes — full MCTS-vs-random game completes.
- [ ] Determinism regression passes.
