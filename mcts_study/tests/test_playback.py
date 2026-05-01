"""Smoke test for v2 playback: synthesize a parquet via e1 then render."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from catan_mcts import playback


@pytest.fixture(scope="module")
def minimal_run_dir(tmp_path_factory):
    """Spawn one tiny e1 game for the playback test."""
    from catan_mcts.experiments.e1_winrate_vs_random import main
    out_root = tmp_path_factory.mktemp("playback_runs")
    return main(
        out_root=out_root,
        num_games=1, sims_per_move_grid=[2],
        seed_base=4242, max_seconds=300.0,
    )


def test_action_desc_covers_full_v2_action_space():
    # Spot-check each block.
    assert "BuildSettlement" in playback._action_desc(0)
    assert "BuildCity" in playback._action_desc(54)
    assert "BuildRoad" in playback._action_desc(108)
    assert "MoveRobber" in playback._action_desc(180)
    assert "Discard" in playback._action_desc(199)
    assert "EndTurn" == playback._action_desc(204)
    assert "RollDice" == playback._action_desc(205)
    assert "TradeBank" in playback._action_desc(206)
    assert "BuyDevCard" == playback._action_desc(226)
    assert "PlayKnight" == playback._action_desc(227)
    assert "PlayRoadBuilding" == playback._action_desc(228)
    assert "PlayMonopoly" in playback._action_desc(229)
    assert "PlayYearOfPlenty" in playback._action_desc(234)
    assert "PlayVpCard" == playback._action_desc(259)
    assert "ProposeTrade" in playback._action_desc(260)
    assert playback._action_desc(279).startswith("ProposeTrade")
    assert "<unknown" in playback._action_desc(280)


def test_render_emits_html(minimal_run_dir, tmp_path):
    seed = 4242 + 2 * 1_000  # e1 uses seed_base + sims*1000
    out = playback.render(minimal_run_dir, seed, tmp_path / "out")
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    # Smoke: contains the seed marker, embeds a board PNG, and has overlays.
    assert f"seed {seed}" in html
    assert "data:image/png;base64," in html
    assert '"states"' in html
    # board.png artefact also produced as a side-effect of rendering.
    assert (tmp_path / "out" / "board.png").exists()


def test_render_includes_polish_markers(minimal_run_dir, tmp_path):
    """Polish-pass markers — see specs/2026-05-01-playback-ui-polish-design.md.
    Each assertion corresponds to one visual region of the polish pass."""
    seed = 4242 + 2 * 1_000
    out = playback.render(minimal_run_dir, seed, tmp_path / "out_polish")
    html = out.read_text(encoding="utf-8")

    # Seat header strip (Task 2)
    assert 'class="seat-chip"' in html
    assert 'class="seat-strip"' in html

    # Dev card emojis (Task 3) — codepoints embedded in the JS DEV_EMOJI literal
    for emoji in ['⚔️', '🛣️', '📜', '🌽', '⭐']:
        assert emoji in html, f"missing dev-card emoji {emoji!r}"

    # Dark narration bar with dice chip (Task 7). seat-tag-N classes are
    # emitted dynamically via JS template literals, so we assert the CSS
    # selector text in the static <style> block rather than the rendered
    # class="..." attribute.
    assert 'class="dice-chip"' in html
    assert '.seat-tag-0' in html
    assert '.seat-tag-3' in html

    # Building / road glyph polish (Task 6) — darker palette identifier.
    assert 'PLAYER_COLORS_DARK' in html
