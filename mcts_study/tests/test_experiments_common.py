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


def test_play_one_game_skips_bot_on_trivial_turns():
    """Optimization: when len(legal_actions)==1, play_one_game should apply the
    forced action directly without invoking the bot. We verify by giving every
    bot a step() that *raises* — if even one trivial turn is delegated to a bot,
    the game crashes. With len==1 turns common in Catan (forced Roll, forced
    EndTurn after no resources), the game should still drive forward, terminating
    or hitting the timeout cleanly."""
    import random
    from catan_mcts.adapter import CatanGame

    class _TrivialOnlyBot:
        """Bot that crashes if asked to choose between >1 legal actions —
        i.e., asserts every call is a 'meaningful' decision. Conversely,
        the test passes if play_one_game NEVER calls bot.step on a trivial
        turn AND every multi-option turn it does call us on, we resolve.
        For this test we just want it to not crash on len==1 turns."""
        calls_with_multiple_legals = 0

        def step(self, state):
            legal = state.legal_actions()
            if len(legal) > 1:
                _TrivialOnlyBot.calls_with_multiple_legals += 1
            else:
                raise AssertionError(
                    "bot.step called on trivial turn (len(legal)==1) — "
                    "play_one_game should auto-apply forced actions"
                )
            return legal[0]

    _TrivialOnlyBot.calls_with_multiple_legals = 0
    game = CatanGame()
    bots = {i: _TrivialOnlyBot() for i in range(4)}
    outcome = play_one_game(
        game=game, bots=bots, seed=42,
        chance_rng=random.Random(42),
        recorded_player=None, recorder_game=None,
        max_seconds=10.0,
    )
    # Either the game finished or hit the wall clock; either way no bot.step
    # was ever called on a forced turn.
    assert outcome.length_in_moves > 0
    # And we DID make at least some real decisions (otherwise the test is vacuous).
    assert _TrivialOnlyBot.calls_with_multiple_legals > 0


class _RandomBot:
    def __init__(self, seed: int) -> None:
        import random
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())
