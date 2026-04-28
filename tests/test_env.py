import numpy as np
from catan_bot import CatanEnv, ACTION_SPACE_SIZE


def test_action_space_size_constant():
    assert ACTION_SPACE_SIZE == 205


def test_random_self_play_completes():
    rng = np.random.default_rng(42)
    env = CatanEnv(seed=42)
    obs = env.reset(seed=42)
    steps = 0
    while not env.is_terminal():
        mask = env.legal_mask()
        legal_ids = np.flatnonzero(mask)
        assert len(legal_ids) > 0, "no legal actions in non-terminal state"
        action = int(rng.choice(legal_ids))
        obs, reward, done, info = env.step(action)
        steps += 1
        assert steps < 5000
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
    assert obs["legal_mask"].shape == (205,)
