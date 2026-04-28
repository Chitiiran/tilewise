import numpy as np
import pytest
from catan_bot import CatanEnv, ACTION_SPACE_SIZE


def test_action_space_size_constant():
    assert ACTION_SPACE_SIZE == 206  # Phase-0 added Action::RollDice (id 205)


def _sample_chance(env, rng):
    """Sample a chance outcome weighted by its probability."""
    outcomes = env.chance_outcomes()
    values = [int(v) for v, _ in outcomes]
    probs = np.array([p for _, p in outcomes], dtype=float)
    probs = probs / probs.sum()  # guard against rounding
    return int(rng.choice(values, p=probs))


def test_random_self_play_completes():
    rng = np.random.default_rng(42)
    env = CatanEnv(seed=42)
    obs = env.reset(seed=42)
    steps = 0
    while not env.is_terminal():
        if env.is_chance_pending():
            env.apply_chance_outcome(_sample_chance(env, rng))
        else:
            mask = env.legal_mask()
            legal_ids = np.flatnonzero(mask)
            assert len(legal_ids) > 0, "no legal actions in non-terminal state"
            action = int(rng.choice(legal_ids))
            obs, reward, done, info = env.step(action)
        steps += 1
        assert steps < 20000
    assert env.is_terminal()
    s = env.stats()
    assert s["winner_player_id"] >= 0
    assert s["turns_played"] > 0


def test_observation_shapes():
    env = CatanEnv(seed=1)
    obs = env._observation()
    assert obs["hex_features"].shape == (19, 8)
    assert obs["vertex_features"].shape == (54, 7)
    assert obs["edge_features"].shape == (72, 6)
    assert obs["legal_mask"].shape == (ACTION_SPACE_SIZE,)


def test_chance_api_roundtrip():
    from catan_bot import _engine
    e = _engine.Engine(42)
    # Drive to a Roll phase by taking legal[0] until is_chance_pending.
    for _ in range(2000):
        if e.is_chance_pending():
            break
        legal = e.legal_actions()
        if len(legal) == 0:
            break
        e.step(int(legal[0]))
    assert e.is_chance_pending()
    outcomes = e.chance_outcomes()  # list of (value, probability)
    assert len(outcomes) == 11      # 2d6 sums
    total = sum(p for _, p in outcomes)
    assert abs(total - 1.0) < 1e-9
    e.apply_chance_outcome(int(outcomes[0][0]))


def test_clone_independence():
    from catan_bot import _engine
    a = _engine.Engine(7)
    a.step(int(a.legal_actions()[0]))
    b = a.clone()
    a_history_before = list(a.action_history())
    b.step(int(b.legal_actions()[0]))
    assert list(a.action_history()) == a_history_before  # untouched


def test_action_history():
    from catan_bot import _engine
    e = _engine.Engine(7)
    a0 = int(e.legal_actions()[0])
    e.step(a0)
    hist = e.action_history()
    assert hist[0] == a0


def test_replay_roundtrip(tmp_path):
    import numpy as np
    from catan_bot import Replay
    rng = np.random.default_rng(7)
    env = CatanEnv(seed=7)
    while not env.is_terminal():
        if env.is_chance_pending():
            env.apply_chance_outcome(_sample_chance(env, rng))
        else:
            mask = env.legal_mask()
            legal_ids = np.flatnonzero(mask)
            a = int(rng.choice(legal_ids))
            env.step(a)
    # Use the engine's authoritative action_history so chance outcomes
    # (encoded with the 0x8000_0000 high-bit flag) are preserved.
    actions = list(env.action_history())
    rep = Replay(schema_version=1, seed=7, actions=actions,
                 engine_version="0.1.0", rules_tier=1)
    p = tmp_path / "rep.json"
    rep.save(p)
    rep2 = Replay.load(p)
    env2 = rep2.reconstruct()
    assert env2.is_terminal()
    assert env2.stats()["winner_player_id"] == env.stats()["winner_player_id"]
