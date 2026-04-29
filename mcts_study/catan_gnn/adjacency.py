"""Static board topology tables for the standard 19-hex Catan board.

Mirrors catan_engine/src/board.rs `hex_to_vertices` and `edge_to_vertices`.
The values come from the engine's `standard_hex_to_vertices()` and
`standard_edge_to_vertices()`. Hardcoded here (rather than fetched via PyO3)
because they never change -- paying a PyO3 call per state_to_pyg() conversion
is wasteful for static data.

Cross-checked against engine via tests/test_adjacency.py.
"""
from __future__ import annotations

import numpy as np

NUM_HEXES = 19
NUM_VERTICES = 54
NUM_EDGES = 72


# Each hex's 6 vertex IDs in clockwise order from the top vertex.
# Source: catan_engine/src/board.rs `standard_hex_to_vertices()`.
HEX_TO_VERTICES: list[list[int]] = [
    [0, 4, 8, 12, 7, 3],       # hex  0
    [1, 5, 9, 13, 8, 4],       # hex  1
    [2, 6, 10, 14, 9, 5],      # hex  2
    [7, 12, 17, 22, 16, 11],   # hex  3
    [8, 13, 18, 23, 17, 12],   # hex  4
    [9, 14, 19, 24, 18, 13],   # hex  5
    [10, 15, 20, 25, 19, 14],  # hex  6
    [16, 22, 28, 33, 27, 21],  # hex  7
    [17, 23, 29, 34, 28, 22],  # hex  8
    [18, 24, 30, 35, 29, 23],  # hex  9
    [19, 25, 31, 36, 30, 24],  # hex 10
    [20, 26, 32, 37, 31, 25],  # hex 11
    [28, 34, 39, 43, 38, 33],  # hex 12
    [29, 35, 40, 44, 39, 34],  # hex 13
    [30, 36, 41, 45, 40, 35],  # hex 14
    [31, 37, 42, 46, 41, 36],  # hex 15
    [39, 44, 48, 51, 47, 43],  # hex 16
    [40, 45, 49, 52, 48, 44],  # hex 17
    [41, 46, 50, 53, 49, 45],  # hex 18
]

# Each edge's 2 vertex IDs.
# Source: catan_engine/src/board.rs `standard_edge_to_vertices()`.
EDGE_TO_VERTICES: list[list[int]] = [
    [0, 3],    # edge  0
    [0, 4],    # edge  1
    [1, 4],    # edge  2
    [1, 5],    # edge  3
    [2, 5],    # edge  4
    [2, 6],    # edge  5
    [3, 7],    # edge  6
    [4, 8],    # edge  7
    [5, 9],    # edge  8
    [6, 10],   # edge  9
    [7, 11],   # edge 10
    [7, 12],   # edge 11
    [8, 12],   # edge 12
    [8, 13],   # edge 13
    [9, 13],   # edge 14
    [9, 14],   # edge 15
    [10, 14],  # edge 16
    [10, 15],  # edge 17
    [11, 16],  # edge 18
    [12, 17],  # edge 19
    [13, 18],  # edge 20
    [14, 19],  # edge 21
    [15, 20],  # edge 22
    [16, 21],  # edge 23
    [16, 22],  # edge 24
    [17, 22],  # edge 25
    [17, 23],  # edge 26
    [18, 23],  # edge 27
    [18, 24],  # edge 28
    [19, 24],  # edge 29
    [19, 25],  # edge 30
    [20, 25],  # edge 31
    [20, 26],  # edge 32
    [21, 27],  # edge 33
    [22, 28],  # edge 34
    [23, 29],  # edge 35
    [24, 30],  # edge 36
    [25, 31],  # edge 37
    [26, 32],  # edge 38
    [27, 33],  # edge 39
    [28, 33],  # edge 40
    [28, 34],  # edge 41
    [29, 34],  # edge 42
    [29, 35],  # edge 43
    [30, 35],  # edge 44
    [30, 36],  # edge 45
    [31, 36],  # edge 46
    [31, 37],  # edge 47
    [32, 37],  # edge 48
    [33, 38],  # edge 49
    [34, 39],  # edge 50
    [35, 40],  # edge 51
    [36, 41],  # edge 52
    [37, 42],  # edge 53
    [38, 43],  # edge 54
    [39, 43],  # edge 55
    [39, 44],  # edge 56
    [40, 44],  # edge 57
    [40, 45],  # edge 58
    [41, 45],  # edge 59
    [41, 46],  # edge 60
    [42, 46],  # edge 61
    [43, 47],  # edge 62
    [44, 48],  # edge 63
    [45, 49],  # edge 64
    [46, 50],  # edge 65
    [47, 51],  # edge 66
    [48, 51],  # edge 67
    [48, 52],  # edge 68
    [49, 52],  # edge 69
    [49, 53],  # edge 70
    [50, 53],  # edge 71
]


def _build_hex_vertex_edge_index() -> np.ndarray:
    """Returns [2, NUM_HEXES*6*2] long-tensor-shaped array. First half is
    hex->vertex direction (src=hex, dst=vertex); second half is vertex->hex
    (src=vertex, dst=hex). PyG expects this format for HeteroConv with both
    edge types, so we stack them and split by direction in state_to_pyg.py."""
    src_h2v, dst_h2v = [], []
    for h, vs in enumerate(HEX_TO_VERTICES):
        for v in vs:
            src_h2v.append(h)
            dst_h2v.append(v)
    src_v2h = list(dst_h2v)
    dst_v2h = list(src_h2v)
    return np.array([src_h2v + src_v2h, dst_h2v + dst_v2h], dtype=np.int64)


def _build_vertex_edge_edge_index() -> np.ndarray:
    """Returns [2, NUM_EDGES*2*2]. First half: vertex->edge (src=vertex, dst=edge_id).
    Second half: edge->vertex (reverse)."""
    src_v2e, dst_v2e = [], []
    for e, vs in enumerate(EDGE_TO_VERTICES):
        for v in vs:
            src_v2e.append(v)
            dst_v2e.append(e)
    src_e2v = list(dst_v2e)
    dst_e2v = list(src_v2e)
    return np.array([src_v2e + src_e2v, dst_v2e + dst_e2v], dtype=np.int64)


HEX_VERTEX_EDGE_INDEX: np.ndarray = _build_hex_vertex_edge_index()
VERTEX_EDGE_EDGE_INDEX: np.ndarray = _build_vertex_edge_edge_index()
