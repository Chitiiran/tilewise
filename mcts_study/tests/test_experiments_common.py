from pathlib import Path

from catan_mcts.experiments.common import (
    make_run_dir,
    play_one_game,
    GameOutcome,
)


def test_make_run_dir_creates_timestamped_subdir(tmp_path: Path):
    d = make_run_dir(parent=tmp_path, name="e0_smoke")
    assert d.parent == tmp_path
    assert "e0_smoke" in d.name
    assert d.is_dir()


def test_play_one_game_returns_outcome():
    """Smoke: play_one_game with all-random bots terminates and returns a GameOutcome."""
    from catan_mcts.adapter import CatanGame
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
