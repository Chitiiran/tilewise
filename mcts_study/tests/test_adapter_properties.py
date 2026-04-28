"""Property test: random legal-action play never raises, terminal returns correct, across 64 seeds."""
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
