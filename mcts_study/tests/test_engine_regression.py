"""Engine regression test — game-output fingerprints from the post-fix smoke.

Locks in the *engine's* deterministic-replay behavior (winner, final_vp,
length_in_moves) for 5 known-good action histories captured on
2026-05-01 after P1+P2+P3 landed (commit 74eaedb).

Why this exists: the M1/M2/M3 refactor pyramid (combined PyO3 calls,
Rust pyspiel.State, Rust MCTS) all touch hot-path engine surfaces.
This test confirms that REPLAYING a known action_history through the
engine still produces the same outcome — i.e. the rules engine is
unchanged. It does NOT assert that MCTS produces the same trajectory
(MCTS RNG is sensitive to legal_actions ordering); for that you need
the upstream play_one_game + a fixed seed.

If this test ever flips, ask: did the rules change? If so, regenerate
the baseline at runs/v2_smoke_postfix/.../games.sims=25.parquet and
update FINGERPRINTS.

This is a free smoke too: it exercises Engine.{step, apply_chance_outcome,
is_chance_pending, is_terminal, stats} against a real long history.
"""
from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from catan_bot import _engine

CHANCE_BIT = 0x8000_0000

# Captured 2026-05-01 from post-fix smoke (commit 74eaedb), via:
#   pq.read_table("runs/v2_smoke_postfix/.../games.sims=25.parquet")
# These are the EXPECTED outcomes when replaying each seed's
# action_history step-by-step.
FINGERPRINTS = [
    {"seed": 725001, "winner": 2, "final_vp": [6, 2, 10, 2],  "length": 930},
    {"seed": 725002, "winner": 3, "final_vp": [6, 9, 7, 10],  "length": 1940},
    {"seed": 725003, "winner": 1, "final_vp": [5, 10, 4, 6],  "length": 1393},
    {"seed": 725004, "winner": 1, "final_vp": [2, 10, 8, 8],  "length": 1659},
    {"seed": 725005, "winner": 2, "final_vp": [5, 7, 10, 3],  "length": 2016},
]

# Versioned baseline parquet committed alongside the test (not in runs/ which
# is gitignored). Regenerate via the e1 smoke sweep + cp.
PARQUET = "baselines/postfix_smoke_games.parquet"


def _load_action_history(seed: int) -> list[int]:
    """Read the captured action_history for `seed` from the baseline parquet."""
    from pathlib import Path

    here = Path(__file__).parent  # mcts_study/tests/
    table = pq.read_table(here / PARQUET)
    df = table.to_pandas()
    row = df[df.seed == seed]
    if len(row) == 0:
        pytest.skip(f"baseline parquet missing seed {seed} (regenerate?)")
    return [int(x) for x in row.iloc[0].action_history]


def _replay(seed: int, history: list[int]) -> tuple[int, list[int], int]:
    """Replay action_history through a fresh engine. Returns (winner, vp, steps)."""
    e = _engine.Engine(seed)
    steps = 0
    for a in history:
        a = int(a)
        if e.is_terminal():
            break
        if a & CHANCE_BIT:
            e.apply_chance_outcome(a & 0x7FFF_FFFF)
        else:
            e.step(a)
        steps += 1
    stats = e.stats()
    winner = int(stats["winner_player_id"])
    vp = [int(stats["players"][p]["vp_final"]) for p in range(4)]
    return winner, vp, steps


@pytest.mark.parametrize("fp", FINGERPRINTS, ids=lambda f: f"seed{f['seed']}")
def test_engine_replay_matches_baseline(fp: dict) -> None:
    """Replaying the baseline action_history through the engine must produce
    the recorded (winner, final_vp, length). Catches any rules-engine drift."""
    history = _load_action_history(fp["seed"])
    assert len(history) == fp["length"], (
        f"seed {fp['seed']}: history length {len(history)} != baseline {fp['length']}"
    )
    winner, vp, steps = _replay(fp["seed"], history)
    assert winner == fp["winner"], (
        f"seed {fp['seed']}: winner {winner} != baseline {fp['winner']}"
    )
    assert vp == fp["final_vp"], (
        f"seed {fp['seed']}: final_vp {vp} != baseline {fp['final_vp']}"
    )
    assert steps == fp["length"], (
        f"seed {fp['seed']}: steps {steps} != baseline {fp['length']}"
    )


def test_baseline_parquet_present() -> None:
    """Sanity: the baseline parquet exists. Without it the regression test
    is a no-op skip, which would silently mask drift."""
    from pathlib import Path
    here = Path(__file__).parent
    p = here / PARQUET
    assert p.exists(), (
        f"baseline parquet missing at {p}; "
        "regenerate via `python -m catan_mcts.experiments.e1_winrate_vs_random "
        "--out-root runs/v2_smoke_postfix --num-games 5 --sims-grid 25 "
        "--seed-base 700001 --max-seconds 300.0 --workers 1` and copy "
        "`games.sims=25.parquet` -> `tests/baselines/postfix_smoke_games.parquet`"
    )
