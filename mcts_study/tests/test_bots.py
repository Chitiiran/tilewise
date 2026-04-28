import random

from catan_mcts.adapter import CatanGame
from catan_mcts.bots import GreedyBaselineBot, heuristic_rollout


def test_greedy_bot_completes_game_against_random():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    bots = [GreedyBaselineBot(seed=0)] + [_RandomBot(i + 1) for i in range(3)]
    rng = random.Random(123)
    steps = 0
    # Empirically: greedy P0 + 3 random opponents on (engine_seed=42,
    # chance_seed=123) terminates around step ~11.7k. Budget 15000 leaves
    # comfortable headroom while still catching infinite-loop bugs.
    while not state.is_terminal() and steps < 15000:
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
