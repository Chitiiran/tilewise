"""Redesign config knobs (spec 2026-06-14 §5)."""
from __future__ import annotations


def test_redesign_config_values():
    from catan_az.config import AzConfig
    c = AzConfig()
    assert c.games_per_iter == 1000
    assert c.arena_games == 300
    assert c.arena_games % 4 == 0          # 4 rotations
    assert c.promote_threshold == 0.65
    assert c.max_iters_per_model == 10
