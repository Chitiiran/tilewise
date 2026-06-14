"""AzConfig: spec §7 defaults + JSON round-trip with typo guard."""
from __future__ import annotations

import pytest


def test_defaults_match_spec():
    from catan_az.config import AzConfig
    cfg = AzConfig()
    assert cfg.games_per_iter == 400
    assert cfg.window_games == 1200
    assert cfg.sims == 200
    assert cfg.dirichlet_alpha == 0.8
    assert cfg.dirichlet_eps == 0.25
    assert cfg.temp_moves == 30
    assert cfg.lr == 2e-4
    assert cfg.max_epochs == 4
    assert cfg.early_stop is True
    assert cfg.policy_sharpen == 1.0
    assert cfg.arena_games == 120
    assert cfg.promote_threshold == 0.55
    assert cfg.arena_timeout_rate_max == 0.05
    assert cfg.arena_game_max_seconds == 600.0
    assert cfg.arena_max_draw_rate == 0.40
    assert cfg.arena_min_decisive == 40
    assert cfg.anchor_every == 5
    assert cfg.anchor_games == 60
    assert cfg.vp_target == 10
    assert cfg.bonuses is True
    assert cfg.n_procs == 5
    assert cfg.hidden_dim == 128
    assert cfg.num_layers == 4


def test_json_round_trip(tmp_path):
    from catan_az.config import AzConfig
    cfg = AzConfig(games_per_iter=8, sims=16)
    p = tmp_path / "cfg.json"
    cfg.to_json(p)
    back = AzConfig.from_json(p)
    assert back == cfg
    assert back.games_per_iter == 8 and back.sims == 16


def test_unknown_key_rejected(tmp_path):
    from catan_az.config import AzConfig
    p = tmp_path / "cfg.json"
    p.write_text('{"games_per_iter": 8, "smis": 16}')  # typo'd "sims"
    with pytest.raises(TypeError):
        AzConfig.from_json(p)
