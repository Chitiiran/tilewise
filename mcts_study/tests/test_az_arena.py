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
