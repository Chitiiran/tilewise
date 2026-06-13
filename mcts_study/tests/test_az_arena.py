"""Arena gate logic (spec §2 ARENA): seating, verdict, serialization.

The async game driver itself is exercised by the micro-iteration
integration test (test_az_integration.py), not here — these tests cover
every branch that doesn't need a GPU.
"""
from __future__ import annotations

import pytest


def test_seating_rotations_cover_all_seats():
    from catan_az.arena import seating_for_rotation
    # Base: candidate at seats 0+2, champion at 1+3; rotations shift.
    seen_cand_seats = set()
    for rot in range(4):
        seating = seating_for_rotation(rot)
        assert len(seating) == 4
        assert seating.count("cand") == 2 and seating.count("champ") == 2
        seen_cand_seats |= {i for i, r in enumerate(seating) if r == "cand"}
    assert seen_cand_seats == {0, 1, 2, 3}   # candidate visits every seat


def test_seed_plan_unique_and_shared(tmp_path):
    from catan_az.arena import seed_plan
    plan = seed_plan(seed_base=500, games=120)
    assert len(plan) == 120
    assert len({(rot, seed) for rot, seed in plan}) == 120
    rots = [rot for rot, _ in plan]
    assert all(rots.count(r) == 30 for r in range(4))


def test_should_promote_threshold_strictly_greater():
    from catan_az.arena import ArenaResult, should_promote
    from catan_az.config import AzConfig
    cfg = AzConfig()
    # 66/120 = 55.0% exactly -> hold (strictly greater promotes)
    r = ArenaResult(wins_cand=66, wins_champ=54, draws=0, timeouts=0)
    assert should_promote(r, cfg) == "hold"
    # 67/120 = 55.8% -> promote
    r = ArenaResult(wins_cand=67, wins_champ=53, draws=0, timeouts=0)
    assert should_promote(r, cfg) == "promote"


def test_should_promote_timeout_guard_invalidates():
    from catan_az.arena import ArenaResult, should_promote
    from catan_az.config import AzConfig
    cfg = AzConfig()
    # Great winrate but 8/120 = 6.7% timeouts > 5% cap -> invalid (e5 lesson:
    # wall-clock-censored winrates are not real).
    r = ArenaResult(wins_cand=80, wins_champ=32, draws=0, timeouts=8)
    assert should_promote(r, cfg) == "invalid"


def test_draws_count_in_denominator():
    from catan_az.arena import ArenaResult
    r = ArenaResult(wins_cand=60, wins_champ=40, draws=20, timeouts=0)
    assert r.games == 120
    assert r.winrate_cand == pytest.approx(60 / 120)


def test_result_json_round_trip(tmp_path):
    from catan_az.arena import ArenaResult
    r = ArenaResult(wins_cand=67, wins_champ=53, draws=0, timeouts=2,
                    per_rotation=[20, 15, 17, 15])
    p = tmp_path / "arena.json"
    r.to_json(p)
    back = ArenaResult.from_json(p)
    assert back == r


def test_play_arena_game_wall_clock_cap(monkeypatch):
    """A game that never terminates must hit the wall-clock cap and return
    timed_out=True, instead of grinding to the 200k step cap (2026-06-13
    straggler held the whole gate hostage for 45min+)."""
    import asyncio

    from catan_az import arena as arena_mod

    class NeverEndingState:
        def is_terminal(self):
            return False

        def is_chance_node(self):
            return False

        def legal_actions(self):
            return [0, 1]   # >1 so it always calls the (stubbed) search

        def current_player(self):
            return 0

        def apply_action(self, a):
            pass

    class FakeGame:
        def new_initial_state(self, seed):
            return NeverEndingState()

    class FakeMcts:
        async def search(self, state, n_sims):
            return {0: 1}

        def best_action(self, vc):
            return 0

    winner, timed_out = asyncio.run(arena_mod._play_arena_game(
        game=FakeGame(), seed=1, seating=["cand"] * 4,
        mcts_cand=FakeMcts(), mcts_champ=FakeMcts(), sims=1,
        max_seconds=0.2))
    assert timed_out is True
    assert winner == -1


def test_run_arena_sets_active_game_count(tmp_path, monkeypatch):
    """Regression: each evaluator's active_game_count must be set below the
    10**9 default during the run, else the all-parked flush clause never
    fires and batches degrade to window-only (the 2026-06-13 arena stall:
    98% CPU, 32% GPU, watchdog spam, 0 results). Drives run_arena with fakes
    for the GPU/engine pieces."""
    import catan_az.arena as arena_mod
    from catan_az.config import AzConfig

    seen_active = {"min": 10 ** 9}

    class FakeEvaluator:
        def __init__(self, **kw):
            object.__setattr__(self, "active_game_count", 10 ** 9)

        def start(self):
            pass

        async def stop(self):
            pass

        def __setattr__(self, k, v):
            object.__setattr__(self, k, v)
            if k == "active_game_count":
                seen_active["min"] = min(seen_active["min"], v)

    class FakeGame:
        def __init__(self, **kw):
            pass

        def new_initial_state(self, seed):
            return None

    class FakeMcts:
        def __init__(self, **kw):
            pass

    async def fake_play(*, game, seed, seating, mcts_cand, mcts_champ, sims,
                        max_seconds=None):
        return 0, False   # candidate (seat 0 in rot 0) wins, not timed out

    # run_arena imports these locally (keeps the module torch-free); patch at
    # their source modules so the local imports resolve to the fakes.
    monkeypatch.setattr("catan_mcts.batched_evaluator.BatchedGnnEvaluator",
                        FakeEvaluator)
    monkeypatch.setattr("catan_mcts.adapter.CatanGame", FakeGame)
    monkeypatch.setattr("catan_mcts.async_mcts.AsyncMcts", FakeMcts)
    monkeypatch.setattr(arena_mod, "_load_model", lambda *a, **k: object())
    monkeypatch.setattr(arena_mod, "_play_arena_game", fake_play)

    cfg = AzConfig(arena_games=8, arena_sims=2)
    arena_mod.run_arena(candidate_ckpt=tmp_path / "c.pt",
                        champion_ckpt=tmp_path / "x.pt", cfg=cfg,
                        out_dir=tmp_path / "arena", seed_base=1, device="cpu",
                        n_concurrent=4)
    # With >=1 live game, active_game_count must drop to the live count,
    # never stay at the 10**9 sentinel.
    assert seen_active["min"] < 10 ** 9
    assert seen_active["min"] >= 1
