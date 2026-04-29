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


def test_play_one_game_respects_wall_clock_cap():
    """v2 hardening: max_seconds aborts the game and returns a 'timed_out=True'
    outcome. Protects sweeps against pathological MCTS-rollout slowdowns blocking
    the entire batch."""
    import random
    import time
    from catan_mcts.adapter import CatanGame

    game = CatanGame()
    bots = {i: _RandomBot(i) for i in range(4)}
    t0 = time.perf_counter()
    outcome = play_one_game(
        game=game, bots=bots, seed=42,
        chance_rng=random.Random(42),
        recorded_player=None, recorder_game=None,
        max_seconds=0.05,  # 50 ms — far less than any real game
    )
    elapsed = time.perf_counter() - t0
    # Should give up shortly after the cap, not run for many seconds.
    assert elapsed < 1.0, f"max_seconds=0.05 but elapsed {elapsed:.2f}s"
    # Outcome marks a timeout: winner=-1 AND timed_out flag.
    assert outcome.winner == -1
    assert outcome.timed_out is True
    assert outcome.length_in_moves > 0


def test_play_one_game_natural_finish_not_marked_timed_out():
    """A normally-completing game has timed_out=False, so the recorder can
    distinguish 'genuine draw' from 'we abandoned this one'."""
    import random
    from catan_mcts.adapter import CatanGame

    game = CatanGame()
    bots = {i: _RandomBot(i) for i in range(4)}
    outcome = play_one_game(
        game=game, bots=bots, seed=42,
        chance_rng=random.Random(42),
        recorded_player=None, recorder_game=None,
        max_seconds=600.0,  # generous
    )
    assert outcome.timed_out is False


class _RandomBot:
    def __init__(self, seed: int) -> None:
        import random
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())
