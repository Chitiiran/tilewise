"""Tests for the extracted board geometry + PNG renderer."""
from __future__ import annotations

from pathlib import Path


def test_build_layout_returns_geometry():
    from catan_mcts.web import board_layout
    vertex_xy, edges, hex_centers = board_layout.build_layout()
    assert len(vertex_xy) == 54
    assert len(edges) == 72
    assert len(hex_centers) == 19


def test_layout_dict_is_json_ready():
    from catan_mcts.web import board_layout
    d = board_layout.layout_dict()
    assert set(d.keys()) == {"xlim", "ylim", "vertices", "edges", "hex_centers"}
    assert len(d["vertices"]) == 54
    assert len(d["edges"]) == 72


def test_render_board_png_writes_file(tmp_path):
    from catan_mcts.web import board_layout
    out = tmp_path / "board.png"
    board_layout.render_board_png(seed=4242, out_path=out)
    assert out.exists() and out.stat().st_size > 0
