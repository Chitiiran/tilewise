import pytest
from pathlib import Path

SPIKE = Path(__file__).resolve().parents[1] / "spike"

@pytest.mark.skipif(not (SPIKE / "wrapper_batched.ts").exists(), reason="fixture missing")
def test_batched_kwargs_same_schema_and_dedup_key():
    import catan_mcts_rs
    pairs = [(0, 9001), (1, 9002), (2, 9003), (3, 9004)]
    b1 = str(SPIKE / "wrapper_traced.ts")
    bb = str(SPIKE / "wrapper_batched.ts")
    recs = catan_mcts_rs.run_arena_games(
        b1, b1, pairs, 8, 10, True,
        batched_cand_ts=bb, batched_champ_ts=bb, b_max=8)
    assert [r["seed"] for r in recs] == [s for _, s in pairs]
    for r in recs:
        assert set(r) == {"seed", "rot", "winner_seat", "winner_role",
                          "timed_out", "vp_margin"}
        assert r["winner_role"] in ("cand", "champ", None)


def test_partial_batched_kwargs_raises_value_error():
    # Supplying only some of the three batched kwargs must raise, not
    # silently fall back to the slow serial B=1 path (review finding).
    # Validation happens before any net is loaded, so this doesn't need the
    # spike .ts fixtures — bogus paths are fine, the ValueError fires first.
    import catan_mcts_rs
    pairs = [(0, 9001)]
    with pytest.raises(ValueError):
        catan_mcts_rs.run_arena_games(
            "bogus_cand.ts", "bogus_champ.ts", pairs, 8, 10, True,
            batched_cand_ts="bogus_batched_cand.ts", b_max=8)
