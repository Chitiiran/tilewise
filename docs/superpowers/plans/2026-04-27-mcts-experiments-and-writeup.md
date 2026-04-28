# MCTS Experiments + Writeup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the four planned MCTS experiments end-to-end, generating reproducible parquet datasets and a ~5-page writeup with at least three concrete reusable learnings about MCTS as a tool.

**Architecture:** Each experiment is a self-contained script under `mcts_study/catan_mcts/experiments/`, sharing a thin `common.py` for output-directory plumbing and game-loop boilerplate. A `cli.py` dispatcher exposes them as `python -m catan_mcts run <name>`. Analysis is a single Jupyter notebook reading from `runs/`; the writeup is two markdown files in `mcts_study/docs/`.

**Tech Stack:** Python 3.10+, OpenSpiel MCTSBot, the adapter + recorder + bots from Plan 2, `pandas`, `matplotlib`, `tqdm`, `nbstripout`. No new engine work.

**Prerequisite:** Plans 1 (Phase 0 engine) and 2 (adapter + recorder + bots) complete and merged.

---

### Task 1: `experiments/common.py` — game loop and output plumbing

**Files:**
- Create: `mcts_study/catan_mcts/experiments/__init__.py`
- Create: `mcts_study/catan_mcts/experiments/common.py`
- Create: `mcts_study/tests/test_experiments_common.py`

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_experiments_common.py`:

```python
from pathlib import Path

from catan_mcts.experiments.common import (
    make_run_dir,
    play_one_game,
    GameOutcome,
)


def test_make_run_dir_creates_timestamped_subdir(tmp_path: Path):
    d = make_run_dir(parent=tmp_path, name="e0_smoke")
    assert d.parent == tmp_path
    assert d.name.endswith("e0_smoke") or "e0_smoke" in d.name
    assert d.is_dir()


def test_play_one_game_returns_outcome():
    """Smoke: play_one_game with all-random bots terminates and returns a GameOutcome."""
    from catan_mcts.adapter import CatanGame
    from catan_mcts.bots import GreedyBaselineBot
    import random

    game = CatanGame()
    bots = {i: _RandomBot(i) for i in range(4)}
    outcome = play_one_game(
        game=game,
        bots=bots,
        seed=42,
        chance_rng=random.Random(42),
        recorded_player=None,    # don't record anything
        recorder_game=None,
    )
    assert isinstance(outcome, GameOutcome)
    assert outcome.winner in {-1, 0, 1, 2, 3}
    assert outcome.length_in_moves > 0


class _RandomBot:
    def __init__(self, seed: int) -> None:
        import random
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest mcts_study/tests/test_experiments_common.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `common.py`**

Create `mcts_study/catan_mcts/experiments/__init__.py` as empty.

Create `mcts_study/catan_mcts/experiments/common.py`:

```python
"""Shared experiment infrastructure: run-directory creation, the canonical game loop."""
from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .. import ACTION_SPACE_SIZE


@dataclass
class GameOutcome:
    seed: int
    winner: int
    final_vp: list[int]
    length_in_moves: int


def make_run_dir(parent: Path, name: str) -> Path:
    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M")
    d = parent / f"{ts}-{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sample_chance_outcome(state, rng: random.Random) -> int:
    outcomes = state.chance_outcomes()
    r = rng.random()
    cum = 0.0
    for v, p in outcomes:
        cum += p
        if r <= cum:
            return int(v)
    return int(outcomes[-1][0])


def play_one_game(
    *,
    game,
    bots: dict,
    seed: int,
    chance_rng: random.Random,
    recorded_player: Optional[int] = None,
    recorder_game=None,    # an active _GameRecorder context
    mcts_bot=None,         # only required if recording (visit count extraction)
    max_steps: int = 5000,
) -> GameOutcome:
    """Drive one game to completion. Optionally record `recorded_player`'s MCTS moves.

    For an MCTS player, `bots[recorded_player]` should be the OpenSpiel `MCTSBot`,
    AND `mcts_bot` must point at the same instance — the recorder needs to call
    `mcts_bot.mcts_search(state)` to extract visit counts (Plan 2 Task 10's helper).
    """
    from ..recorder import visit_counts_from_root  # local import to avoid cycle

    state = game.new_initial_state(seed=seed)
    move_index = 0
    steps = 0

    while not state.is_terminal() and steps < max_steps:
        if state.is_chance_node():
            state.apply_action(_sample_chance_outcome(state, chance_rng))
        else:
            p = state.current_player()
            bot = bots[p]
            if recorder_game is not None and p == recorded_player and mcts_bot is not None:
                # Search once to populate the tree (and grab its root for visit counts);
                # then take the search's recommended action.
                root = mcts_bot.mcts_search(state)
                visits = visit_counts_from_root(root)
                # Best action = argmax visits among legal actions
                legal = state.legal_actions()
                best = max(legal, key=lambda a: int(visits[a]))
                mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
                mask[legal] = True
                recorder_game.record_move(
                    current_player=p,
                    move_index=move_index,
                    legal_action_mask=mask,
                    mcts_visit_counts=visits,
                    action_taken=int(best),
                    mcts_root_value=float(getattr(root, "total_reward", 0.0)) /
                                    max(1, int(getattr(root, "explore_count", 1))),
                )
                move_index += 1
                state.apply_action(int(best))
            else:
                action = bot.step(state)
                state.apply_action(int(action))
        steps += 1

    winner = -1
    final_vp = [0, 0, 0, 0]
    if state.is_terminal():
        rs = state.returns()
        if 1.0 in rs:
            winner = rs.index(1.0)
        # Pull VP from engine stats. The adapter exposes the underlying engine via _engine.
        stats = state._engine.stats()
        final_vp = [int(stats["players"][p]["vp_final"]) for p in range(4)]
    return GameOutcome(seed=seed, winner=winner, final_vp=final_vp, length_in_moves=steps)
```

- [ ] **Step 4: Run tests**

Run: `pytest mcts_study/tests/test_experiments_common.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/experiments/__init__.py mcts_study/catan_mcts/experiments/common.py mcts_study/tests/test_experiments_common.py
git commit -m "feat(experiments): common.py — run dir + canonical game loop"
```

---

### Task 2: Experiment 1 — `winrate_vs_random.py`

**Files:**
- Create: `mcts_study/catan_mcts/experiments/e1_winrate_vs_random.py`
- Create: `mcts_study/tests/test_e1_runs.py`

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_e1_runs.py`:

```python
from pathlib import Path

from catan_mcts.experiments.e1_winrate_vs_random import main


def test_e1_smoke_run(tmp_path: Path):
    out = main(
        out_root=tmp_path,
        num_games=3,            # tiny — just check it runs end-to-end
        sims_per_move_grid=[10, 25],
        seed_base=1000,
    )
    moves = (out / "moves.parquet")
    games = (out / "games.parquet")
    assert moves.exists() and games.exists()

    import pyarrow.parquet as pq
    games_df = pq.read_table(games).to_pandas()
    # 2 budgets * 3 games = 6 game rows
    assert len(games_df) == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest mcts_study/tests/test_e1_runs.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement e1**

Create `mcts_study/catan_mcts/experiments/e1_winrate_vs_random.py`:

```python
"""Experiment 1: MCTS win-rate vs 3 RandomBots, sweeping simulation budget."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from open_spiel.python.algorithms import mcts as os_mcts
from tqdm import tqdm

from ..adapter import CatanGame
from ..recorder import SelfPlayRecorder
from .common import make_run_dir, play_one_game


def _build_mcts_bot(game, sims: int, seed: int):
    rng = np.random.default_rng(seed)
    evaluator = os_mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=rng)
    return os_mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=sims,
        evaluator=evaluator, solve=False, random_state=rng,
    )


class _RandomBot:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())


def main(
    *,
    out_root: Path,
    num_games: int = 100,
    sims_per_move_grid: list[int] = (50, 200, 1000, 4000),
    seed_base: int = 1_000_000,
) -> Path:
    out = make_run_dir(out_root, "e1_winrate_vs_random")
    rec = SelfPlayRecorder(out, config={
        "experiment": "e1_winrate_vs_random",
        "uct_c": 1.4,
        "sims_per_move_grid": list(sims_per_move_grid),
        "num_games": num_games,
    })

    game = CatanGame()
    for sims in sims_per_move_grid:
        for i in tqdm(range(num_games), desc=f"sims={sims}", leave=False):
            seed = seed_base + sims * 1_000 + i
            chance_rng = random.Random(seed)
            mcts_bot = _build_mcts_bot(game, sims=sims, seed=seed)
            bots = {0: mcts_bot, 1: _RandomBot(seed+1), 2: _RandomBot(seed+2), 3: _RandomBot(seed+3)}
            with rec.game(seed=seed) as g_rec:
                outcome = play_one_game(
                    game=game, bots=bots, seed=seed, chance_rng=chance_rng,
                    recorded_player=0, recorder_game=g_rec, mcts_bot=mcts_bot,
                )
                g_rec.finalize(
                    winner=outcome.winner,
                    final_vp=outcome.final_vp,
                    length_in_moves=outcome.length_in_moves,
                )
    rec.flush()
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("mcts_study/runs"))
    p.add_argument("--num-games", type=int, default=100)
    p.add_argument("--sims-grid", type=int, nargs="+", default=[50, 200, 1000, 4000])
    p.add_argument("--seed-base", type=int, default=1_000_000)
    args = p.parse_args()
    out = main(
        out_root=args.out_root, num_games=args.num_games,
        sims_per_move_grid=args.sims_grid, seed_base=args.seed_base,
    )
    print(f"e1 wrote to {out}")


if __name__ == "__main__":
    cli_main()
```

- [ ] **Step 4: Run test**

Run: `pytest mcts_study/tests/test_e1_runs.py -v`
Expected: PASS. Will be slow (~30s) due to MCTS simulation.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/experiments/e1_winrate_vs_random.py mcts_study/tests/test_e1_runs.py
git commit -m "feat(e1): MCTS win-rate vs random sims-per-move sweep"
```

---

### Task 3: Experiment 2 — `ucb_c_sweep.py`

**Files:**
- Create: `mcts_study/catan_mcts/experiments/e2_ucb_c_sweep.py`
- Create: `mcts_study/tests/test_e2_runs.py`

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_e2_runs.py`:

```python
from pathlib import Path

from catan_mcts.experiments.e2_ucb_c_sweep import main


def test_e2_smoke_run(tmp_path: Path):
    out = main(
        out_root=tmp_path, num_games=2, c_grid=[0.5, 1.4, 4.0], sims=25, seed_base=2000,
    )
    import pyarrow.parquet as pq
    games_df = pq.read_table(out / "games.parquet").to_pandas()
    # 3 c-values * 2 games = 6
    assert len(games_df) == 6
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement e2**

Create `mcts_study/catan_mcts/experiments/e2_ucb_c_sweep.py`:

```python
"""Experiment 2: MCTS UCB exploration constant `c` sweep at fixed simulation budget."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from open_spiel.python.algorithms import mcts as os_mcts
from tqdm import tqdm

from ..adapter import CatanGame
from ..recorder import SelfPlayRecorder
from .common import make_run_dir, play_one_game


class _RandomBot:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())


def _build_mcts_bot(game, c: float, sims: int, seed: int):
    rng = np.random.default_rng(seed)
    evaluator = os_mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=rng)
    return os_mcts.MCTSBot(
        game=game, uct_c=c, max_simulations=sims,
        evaluator=evaluator, solve=False, random_state=rng,
    )


def main(
    *,
    out_root: Path,
    num_games: int = 100,
    c_grid: list[float] = (0.5, 1.0, 1.4, 2.0, 4.0),
    sims: int = 200,
    seed_base: int = 2_000_000,
) -> Path:
    out = make_run_dir(out_root, "e2_ucb_c_sweep")
    rec = SelfPlayRecorder(out, config={
        "experiment": "e2_ucb_c_sweep",
        "c_grid": list(c_grid),
        "sims_per_move": sims,
        "num_games": num_games,
    })

    game = CatanGame()
    for c in c_grid:
        for i in tqdm(range(num_games), desc=f"c={c}", leave=False):
            seed = seed_base + int(c * 100) * 1_000 + i
            chance_rng = random.Random(seed)
            mcts_bot = _build_mcts_bot(game, c=c, sims=sims, seed=seed)
            bots = {0: mcts_bot, 1: _RandomBot(seed+1), 2: _RandomBot(seed+2), 3: _RandomBot(seed+3)}
            with rec.game(seed=seed) as g_rec:
                outcome = play_one_game(
                    game=game, bots=bots, seed=seed, chance_rng=chance_rng,
                    recorded_player=0, recorder_game=g_rec, mcts_bot=mcts_bot,
                )
                g_rec.finalize(
                    winner=outcome.winner,
                    final_vp=outcome.final_vp,
                    length_in_moves=outcome.length_in_moves,
                )
    rec.flush()
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("mcts_study/runs"))
    p.add_argument("--num-games", type=int, default=100)
    p.add_argument("--c-grid", type=float, nargs="+", default=[0.5, 1.0, 1.4, 2.0, 4.0])
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--seed-base", type=int, default=2_000_000)
    args = p.parse_args()
    out = main(
        out_root=args.out_root, num_games=args.num_games,
        c_grid=args.c_grid, sims=args.sims, seed_base=args.seed_base,
    )
    print(f"e2 wrote to {out}")


if __name__ == "__main__":
    cli_main()
```

- [ ] **Step 4: Run test**

Run: `pytest mcts_study/tests/test_e2_runs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/experiments/e2_ucb_c_sweep.py mcts_study/tests/test_e2_runs.py
git commit -m "feat(e2): UCB c sweep experiment"
```

---

### Task 4: Experiment 3 — `rollout_policy.py`

**Files:**
- Create: `mcts_study/catan_mcts/experiments/e3_rollout_policy.py`
- Create: `mcts_study/tests/test_e3_runs.py`

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_e3_runs.py`:

```python
from pathlib import Path

from catan_mcts.experiments.e3_rollout_policy import main


def test_e3_smoke_run(tmp_path: Path):
    out = main(out_root=tmp_path, num_games=2, sims_grid=[25, 100], seed_base=3000)
    import pyarrow.parquet as pq
    games_df = pq.read_table(out / "games.parquet").to_pandas()
    # 2 rollout policies * 2 sim budgets * 2 games = 8 game rows
    assert len(games_df) == 8
```

- [ ] **Step 2: Run test (expected fail)**

Expected: FAIL.

- [ ] **Step 3: Implement e3**

Create `mcts_study/catan_mcts/experiments/e3_rollout_policy.py`:

```python
"""Experiment 3: MCTS rollout policy comparison — random vs heuristic, across sim budgets."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from open_spiel.python.algorithms import mcts as os_mcts
from tqdm import tqdm

from ..adapter import CatanGame
from ..bots import heuristic_rollout
from ..recorder import SelfPlayRecorder
from .common import make_run_dir, play_one_game


class _RandomBot:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())


class _HeuristicEvaluator(os_mcts.Evaluator):
    """OpenSpiel evaluator that rolls out using `heuristic_rollout`."""

    def __init__(self, n_rollouts: int, rng: random.Random) -> None:
        self._n = n_rollouts
        self._rng = rng

    def evaluate(self, state):
        wins = np.zeros(state.num_players(), dtype=np.float32)
        for _ in range(self._n):
            sim = state.clone()
            steps = 0
            while not sim.is_terminal() and steps < 1000:
                if sim.is_chance_node():
                    outcomes = sim.chance_outcomes()
                    r = self._rng.random()
                    cum, chosen = 0.0, outcomes[-1][0]
                    for v, p in outcomes:
                        cum += p
                        if r <= cum:
                            chosen = v
                            break
                    sim.apply_action(int(chosen))
                else:
                    sim.apply_action(int(heuristic_rollout(sim, self._rng)))
                steps += 1
            wins += np.array(sim.returns(), dtype=np.float32)
        return wins / self._n

    def prior(self, state):
        legal = state.legal_actions()
        p = 1.0 / len(legal)
        return [(a, p) for a in legal]


def _bot(game, sims: int, evaluator, seed: int):
    rng = np.random.default_rng(seed)
    return os_mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=sims,
        evaluator=evaluator, solve=False, random_state=rng,
    )


def main(
    *,
    out_root: Path,
    num_games: int = 100,
    sims_grid: list[int] = (50, 200, 1000),
    seed_base: int = 3_000_000,
) -> Path:
    out = make_run_dir(out_root, "e3_rollout_policy")
    rec = SelfPlayRecorder(out, config={
        "experiment": "e3_rollout_policy",
        "uct_c": 1.4,
        "sims_grid": list(sims_grid),
        "num_games": num_games,
        "rollout_policies": ["random", "heuristic"],
    })

    game = CatanGame()
    for policy_name in ("random", "heuristic"):
        for sims in sims_grid:
            for i in tqdm(range(num_games), desc=f"{policy_name} sims={sims}", leave=False):
                seed = seed_base + (0 if policy_name == "random" else 1) * 1_000_000 + sims * 1_000 + i
                chance_rng = random.Random(seed)
                rng = np.random.default_rng(seed)
                if policy_name == "random":
                    evaluator = os_mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=rng)
                else:
                    evaluator = _HeuristicEvaluator(n_rollouts=1, rng=random.Random(seed))
                mcts_bot = _bot(game, sims=sims, evaluator=evaluator, seed=seed)
                bots = {0: mcts_bot, 1: _RandomBot(seed+1), 2: _RandomBot(seed+2), 3: _RandomBot(seed+3)}
                with rec.game(seed=seed) as g_rec:
                    outcome = play_one_game(
                        game=game, bots=bots, seed=seed, chance_rng=chance_rng,
                        recorded_player=0, recorder_game=g_rec, mcts_bot=mcts_bot,
                    )
                    g_rec.finalize(
                        winner=outcome.winner,
                        final_vp=outcome.final_vp,
                        length_in_moves=outcome.length_in_moves,
                    )
    rec.flush()
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("mcts_study/runs"))
    p.add_argument("--num-games", type=int, default=100)
    p.add_argument("--sims-grid", type=int, nargs="+", default=[50, 200, 1000])
    p.add_argument("--seed-base", type=int, default=3_000_000)
    args = p.parse_args()
    out = main(
        out_root=args.out_root, num_games=args.num_games,
        sims_grid=args.sims_grid, seed_base=args.seed_base,
    )
    print(f"e3 wrote to {out}")


if __name__ == "__main__":
    cli_main()
```

- [ ] **Step 4: Run test**

Run: `pytest mcts_study/tests/test_e3_runs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/experiments/e3_rollout_policy.py mcts_study/tests/test_e3_runs.py
git commit -m "feat(e3): rollout policy comparison (random vs heuristic)"
```

---

### Task 5: Experiment 4 — `tournament.py`

**Files:**
- Create: `mcts_study/catan_mcts/experiments/e4_tournament.py`
- Create: `mcts_study/tests/test_e4_runs.py`

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_e4_runs.py`:

```python
from pathlib import Path

from catan_mcts.experiments.e4_tournament import main


def test_e4_smoke_run(tmp_path: Path):
    out = main(out_root=tmp_path, num_games_per_seating=2, mcts_sims=25, seed_base=4000)
    import pyarrow.parquet as pq
    games_df = pq.read_table(out / "games.parquet").to_pandas()
    # We sample a small fixed set of seatings (4 rotations) * 2 games = 8.
    assert len(games_df) == 8
```

- [ ] **Step 2: Run test (expected fail)**

- [ ] **Step 3: Implement e4**

Create `mcts_study/catan_mcts/experiments/e4_tournament.py`:

```python
"""Experiment 4: round-robin-ish tournament. MCTS vs Greedy vs Random.

Seating is rotated through the 4 cyclic rotations (not all 24 permutations — that's
a lot of compute for limited statistical extra payoff for a study at this scope).
Each rotation is run `num_games_per_seating` times with different seeds.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from open_spiel.python.algorithms import mcts as os_mcts
from tqdm import tqdm

from ..adapter import CatanGame
from ..bots import GreedyBaselineBot
from ..recorder import SelfPlayRecorder
from .common import make_run_dir, play_one_game


class _RandomBot:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())


def _build_mcts(game, sims: int, seed: int):
    rng = np.random.default_rng(seed)
    evaluator = os_mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=rng)
    return os_mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=sims,
        evaluator=evaluator, solve=False, random_state=rng,
    )


def main(
    *,
    out_root: Path,
    num_games_per_seating: int = 25,
    mcts_sims: int = 1000,
    seed_base: int = 4_000_000,
) -> Path:
    """4 cyclic rotations of [MCTS, Greedy, Random, Random]."""
    out = make_run_dir(out_root, "e4_tournament")
    rec = SelfPlayRecorder(out, config={
        "experiment": "e4_tournament",
        "mcts_sims": mcts_sims,
        "num_games_per_seating": num_games_per_seating,
    })

    game = CatanGame()
    base_seating = ["MCTS", "Greedy", "Random", "Random"]
    rotations = [base_seating[i:] + base_seating[:i] for i in range(4)]

    for rot_idx, seating in enumerate(rotations):
        for i in tqdm(range(num_games_per_seating), desc=f"rot={rot_idx} {'-'.join(seating)}", leave=False):
            seed = seed_base + rot_idx * 10_000 + i
            chance_rng = random.Random(seed)
            mcts_bot = _build_mcts(game, sims=mcts_sims, seed=seed)
            bots = {}
            for slot, role in enumerate(seating):
                if role == "MCTS":
                    bots[slot] = mcts_bot
                    mcts_slot = slot
                elif role == "Greedy":
                    bots[slot] = GreedyBaselineBot(seed=seed + 100 + slot)
                else:
                    bots[slot] = _RandomBot(seed + 200 + slot)

            with rec.game(seed=seed) as g_rec:
                outcome = play_one_game(
                    game=game, bots=bots, seed=seed, chance_rng=chance_rng,
                    recorded_player=mcts_slot, recorder_game=g_rec, mcts_bot=mcts_bot,
                )
                g_rec.finalize(
                    winner=outcome.winner,
                    final_vp=outcome.final_vp,
                    length_in_moves=outcome.length_in_moves,
                )
    rec.flush()
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("mcts_study/runs"))
    p.add_argument("--num-games-per-seating", type=int, default=25)
    p.add_argument("--mcts-sims", type=int, default=1000)
    p.add_argument("--seed-base", type=int, default=4_000_000)
    args = p.parse_args()
    out = main(
        out_root=args.out_root,
        num_games_per_seating=args.num_games_per_seating,
        mcts_sims=args.mcts_sims, seed_base=args.seed_base,
    )
    print(f"e4 wrote to {out}")


if __name__ == "__main__":
    cli_main()
```

- [ ] **Step 4: Run test**

Run: `pytest mcts_study/tests/test_e4_runs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/experiments/e4_tournament.py mcts_study/tests/test_e4_runs.py
git commit -m "feat(e4): MCTS vs Greedy vs Random tournament"
```

---

### Task 6: CLI dispatcher

**Files:**
- Create: `mcts_study/catan_mcts/cli.py`
- Create: `mcts_study/catan_mcts/__main__.py`
- Create: `mcts_study/tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Create `mcts_study/tests/test_cli.py`:

```python
import subprocess
import sys


def test_cli_help_lists_experiments():
    r = subprocess.run(
        [sys.executable, "-m", "catan_mcts", "run", "--help"],
        capture_output=True, text=True, check=True,
    )
    out = r.stdout + r.stderr
    for name in ("e1", "e2", "e3", "e4", "all"):
        assert name in out
```

- [ ] **Step 2: Run test (expected fail)**

- [ ] **Step 3: Implement CLI**

Create `mcts_study/catan_mcts/cli.py`:

```python
"""CLI dispatcher: `python -m catan_mcts run <experiment> [args...]`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .experiments import (
    e1_winrate_vs_random,
    e2_ucb_c_sweep,
    e3_rollout_policy,
    e4_tournament,
)


_EXPERIMENTS = {
    "e1": e1_winrate_vs_random,
    "e2": e2_ucb_c_sweep,
    "e3": e3_rollout_policy,
    "e4": e4_tournament,
}


def _run_all(out_root: Path) -> None:
    for name, mod in _EXPERIMENTS.items():
        print(f"=== {name} ===", flush=True)
        mod.main(out_root=out_root)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="catan_mcts")
    sub = p.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run", help="run an experiment (e1, e2, e3, e4, or all)")
    runp.add_argument("name", choices=list(_EXPERIMENTS) + ["all"])
    runp.add_argument("--out-root", type=Path, default=Path("mcts_study/runs"))
    args, rest = p.parse_known_args(argv)

    if args.cmd != "run":
        p.error(f"unknown command {args.cmd}")
        return 2

    if args.name == "all":
        _run_all(args.out_root)
    else:
        mod = _EXPERIMENTS[args.name]
        # Forward unknown args to the experiment's CLI parser by pushing them onto sys.argv.
        sys.argv = [f"catan_mcts.{args.name}", *rest]
        mod.cli_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `mcts_study/catan_mcts/__main__.py`:

```python
from .cli import main

raise SystemExit(main())
```

- [ ] **Step 4: Run test**

Run: `pytest mcts_study/tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcts_study/catan_mcts/cli.py mcts_study/catan_mcts/__main__.py mcts_study/tests/test_cli.py
git commit -m "feat(cli): python -m catan_mcts run <name>"
```

---

### Task 7: Run the experiments at production scale (manual, not a test)

**Files:**
- (no source changes — this task produces data)

- [ ] **Step 1: Run e1 at full scale**

Run: `python -m catan_mcts run e1 --num-games 200`
Expected: takes ~30-60 minutes; writes a new dir under `mcts_study/runs/`.

- [ ] **Step 2: Run e2 at full scale**

Run: `python -m catan_mcts run e2 --num-games 200`
Expected: similar.

- [ ] **Step 3: Run e3 at full scale**

Run: `python -m catan_mcts run e3 --num-games 200`
Expected: similar; possibly longer due to heuristic-rollout evaluator overhead.

- [ ] **Step 4: Run e4 at full scale**

Run: `python -m catan_mcts run e4 --num-games-per-seating 50`
Expected: 4 rotations × 50 games × 1000 sims/move ≈ 1-2 hours.

- [ ] **Step 5: Commit a small README of the production run**

Create `mcts_study/runs/README.md`:

```markdown
# Production runs

This directory holds the parquet outputs of `python -m catan_mcts run <name>`.
Subdirectories are timestamped; the analysis notebook reads from the latest
of each experiment by default. Override with `RUN_E1=...` etc.

Production-run dates and seeds:

| Experiment | Date | Num games | Notes |
|---|---|---|---|
| e1 | (fill in) | 200 | sims grid 50/200/1000/4000 |
| e2 | (fill in) | 200 | c grid 0.5..4.0 at sims=200 |
| e3 | (fill in) | 200 | sims grid 50/200/1000, two rollout policies |
| e4 | (fill in) | 50 / seating | 4 cyclic rotations |
```

```bash
git add mcts_study/runs/README.md
git commit -m "docs(runs): document the canonical production run"
```

---

### Task 8: Analysis notebook

**Files:**
- Create: `mcts_study/notebooks/analysis.ipynb`
- Create: `mcts_study/notebooks/.gitignore` (strip outputs)

- [ ] **Step 1: Configure nbstripout**

Run from repo root:
```bash
pip install nbstripout
nbstripout --install
```

Add to `mcts_study/.gitignore` (already exists from Plan 2):
```
.ipynb_checkpoints/
```

- [ ] **Step 2: Write the notebook outline**

Create `mcts_study/notebooks/analysis.ipynb` with cells corresponding to:

1. **Setup cell** — imports, `runs_root = Path("../runs")`, helper `latest_run(name)`.
2. **e1 plot** — load `runs/<latest e1>/games.parquet`, group by `mcts_config_id` (which encodes sims), compute `win_rate = mean(winner == 0)`, plot with error bars over `num_games`.
3. **e2 plot** — same shape, x-axis = `c`.
4. **e3 plot** — two lines, win_rate vs sims, one per rollout policy. The crossover (if any) is the headline learning.
5. **e4 plot** — heatmap or grouped bar of win-rate per role × rotation.
6. **Discussion cell (markdown)** — pre-populated with prompts: "What surprised you?" "What was the cost-vs-strength curve in e1?" "Did c≈√2 hold?" "Did heuristic rollouts win at low sims?"

(The notebook is exploratory — write the cells while looking at real data; the structure above is the skeleton.)

- [ ] **Step 3: Smoke-run the notebook**

Run: `jupyter nbconvert --to notebook --execute mcts_study/notebooks/analysis.ipynb --output analysis.ipynb`
Expected: completes without errors, produces all four plots.

- [ ] **Step 4: Strip outputs and commit**

```bash
nbstripout mcts_study/notebooks/analysis.ipynb
git add mcts_study/notebooks/analysis.ipynb
git commit -m "feat(notebook): analysis notebook reading runs/ parquet"
```

---

### Task 9: Writeup — `docs/writeup.md` and `docs/learnings.md`

**Files:**
- Create: `mcts_study/docs/writeup.md`
- Create: `mcts_study/docs/learnings.md`

- [ ] **Step 1: Write the writeup skeleton**

Create `mcts_study/docs/writeup.md` with these sections (~5 pages total when filled):

```markdown
# MCTS on Catan — Study Writeup

## 1. Setup
- Engine version (`engine_version()` output, git SHA).
- OpenSpiel version.
- Hardware (laptop CPU, single-threaded).
- Action space: 206 IDs.
- Chance nodes: 2d6 dice (11 outcomes), robber-steal (1 per card in victim's hand).
- Opponents in e1-e3: 3 RandomBots. Opponents in e4: see e4 section.

## 2. Experiment 1 — Win-rate vs simulation budget
*Embedded plot from notebook.*
- Headline number: at sims=4000, MCTS wins X% of games vs random.
- Cost: ~Y ms / move at this budget.
- Discussion: shape of the curve. Does it plateau? Where?

## 3. Experiment 2 — UCB exploration constant
*Embedded plot.*
- Best `c` for our setup (likely near √2 ≈ 1.4 by classical results).
- How sensitive is win-rate to `c`? Wide plateau or sharp peak?

## 4. Experiment 3 — Rollout policy
*Embedded plot.*
- Random vs heuristic rollouts at three sim budgets.
- Headline finding: does the crossover (heuristic wins low-sim, random wins high-sim) appear, and where?

## 5. Experiment 4 — Tournament
*Embedded plot.*
- MCTS, Greedy, Random win-rates by rotation.
- Seat-advantage check: does win-rate change much with rotation?

## 6. Caveats and threats to validity
- Fixed board (Tier 1 of engine).
- 3 random opponents are weak — strong opponents would change every conclusion.
- Sample sizes (200 games per cell). Confidence intervals computed in notebook.
- Heuristic rollout is intentionally crude; a stronger heuristic could shift e3.

## 7. What's next
- Hand off the self-play parquet to the GNN project.
- Possible next study: replace UCB rollouts with a learned value function (AlphaZero step).
```

Fill in the actual numbers and discussion after Task 8 completes.

- [ ] **Step 2: Write `learnings.md`**

Create `mcts_study/docs/learnings.md`:

```markdown
# Reusable learnings from the MCTS study

This file is the primary deliverable of the project. Each entry is something I'd
take to a *different* MCTS application without rerunning experiments.

## 1. ...
*(headline learning — TBD after running)*

## 2. ...

## 3. ...

## (more if relevant)
```

The success criterion (spec §1.4) requires **at least 3 entries**. Examples of the *kind* of learning we expect:

- "On stochastic games, modelling chance correctly was the difference between meaningful UCB statistics and noise. Worth the engine refactor."
- "MCTS strength curves are concave in compute — most of the strength gain happens by ~1000 sims/move; doubling beyond is rarely worth it for a baseline."
- "Heuristic rollouts beat random rollouts at low budgets but the gap closes by ~Nx sims. The right rollout policy depends on your compute target."

(Don't ship placeholders — fill these in from the actual data.)

- [ ] **Step 3: Commit the skeletons**

```bash
git add mcts_study/docs/writeup.md mcts_study/docs/learnings.md
git commit -m "docs(writeup): skeletons for writeup + learnings"
```

- [ ] **Step 4: Fill in writeup + learnings from real data**

Open the notebook, run it, copy plots into `writeup.md`, write the discussion in each section, and write at least 3 concrete learnings in `learnings.md`. Aim for ~5 pages total in `writeup.md`.

- [ ] **Step 5: Commit the filled writeup**

```bash
git add mcts_study/docs/writeup.md mcts_study/docs/learnings.md
git commit -m "docs(writeup): fill in writeup + learnings from production runs"
```

---

### Task 10: Project README

**Files:**
- Create: `mcts_study/README.md`
- Modify: top-level `README.md` (add a pointer)

- [ ] **Step 1: Write the project README**

Create `mcts_study/README.md`:

```markdown
# mcts_study

A scientific study of MCTS on Catan, using the v1 engine + OpenSpiel.

## Quick start

```bash
cd mcts_study
python -m venv .venv && source .venv/bin/activate     # or .venv/Scripts/activate on Windows
pip install -e .[dev]
maturin develop --manifest-path ../catan_engine/Cargo.toml

pytest                       # all green
python -m catan_mcts run e1  # runs experiment 1
python -m catan_mcts run all # runs all four
```

## Layout

- `catan_mcts/adapter.py` — OpenSpiel adapter wrapping `catan_bot._engine`.
- `catan_mcts/bots.py` — baseline bots and rollout policy.
- `catan_mcts/recorder.py` — parquet self-play dataset writer.
- `catan_mcts/experiments/` — four experiment scripts (e1–e4).
- `catan_mcts/cli.py` — dispatcher.
- `notebooks/analysis.ipynb` — reads `runs/`, produces all plots.
- `docs/writeup.md` — the study writeup.
- `docs/learnings.md` — reusable learnings about MCTS.

## Outputs

- `runs/<timestamp>-<experiment>/{moves,games}.parquet` + `config.json`
- `docs/writeup.md` and `docs/learnings.md`

## Companion to a future GNN project

`runs/*/moves.parquet` is the AlphaZero-style training dataset — `(seed, MCTS visit
distribution, eventual outcome)` triples — for a future GNN policy/value network.
The GNN project's data loader replays each `(seed, action_history_up_to_move_index)`
through the engine to materialize observation tensors on demand (the engine is
deterministic; spec v1 §9).
```

- [ ] **Step 2: Update the repo-root README**

In top-level `README.md`, append to the Roadmap section:

```markdown
- **MCTS Study** ([`mcts_study/`](mcts_study/)) — first project on top of v1.
  OpenSpiel-driven MCTS, four-experiment scientific study, self-play parquet for
  the future GNN project. See [its README](mcts_study/README.md).
```

- [ ] **Step 3: Commit**

```bash
git add mcts_study/README.md README.md
git commit -m "docs: project README + repo-root pointer"
```

---

## Plan-3 self-review checklist

- [ ] All four experiments run end-to-end via `python -m catan_mcts run <name>`.
- [ ] `python -m catan_mcts run all` runs all four with sensible defaults.
- [ ] Notebook produces a plot for each experiment.
- [ ] `writeup.md` is filled (not skeleton) with real numbers and discussion.
- [ ] `learnings.md` has ≥3 concrete reusable learnings — not placeholders.
- [ ] Each experiment's smoke test passes (`pytest mcts_study/tests/test_eN_runs.py`).
- [ ] Determinism regression from Plan 2 still passes.
- [ ] No `pytest.mark.skip` or unused experiments shipped.
- [ ] Top-level README has a pointer to `mcts_study/`.
