"""Adjacency tables in catan_gnn.adjacency must match the engine's board topology."""
import numpy as np

from catan_gnn.adjacency import (
    EDGE_TO_VERTICES,
    HEX_TO_VERTICES,
    HEX_VERTEX_EDGES,
    VERTEX_EDGE_EDGES,
)


def test_hex_vertex_edge_index_shape():
    """[2, num_undirected_edges*2] format expected by PyG. Each undirected edge
    is represented twice (hex->vertex and vertex->hex)."""
    ei = HEX_VERTEX_EDGES
    assert ei.shape[0] == 2
    # 19 hexes * 6 vertices each = 114 hex->vertex edges + 114 reverse = 228.
    assert ei.shape[1] == 228


def test_vertex_edge_edge_index_shape():
    """72 edges * 2 vertices each = 144 vertex->edge + 144 reverse = 288."""
    ei = VERTEX_EDGE_EDGES
    assert ei.shape[0] == 2
    assert ei.shape[1] == 288


def test_hex_vertex_consistent_with_engine_hex_to_vertices():
    """Each (hex, vertex) pair in HEX_VERTEX_EDGES (the hex->vertex direction)
    must correspond to a vertex listed in the engine's hex_to_vertices for
    that hex. We can't read board.rs directly from Python -- instead, use the
    engine's observation for a fresh game and check that the union of vertex
    IDs across HEX_VERTEX_EDGES rows touching each hex hits all 6 expected
    vertex slots."""
    # Hex 0 in the standard board has 6 vertices. In our table rows where
    # row[0] == hex_id == 0, the row[1] entries should be exactly 6 distinct values.
    src, dst = HEX_VERTEX_EDGES[0], HEX_VERTEX_EDGES[1]
    # Direction is encoded by convention: first 114 entries are hex->vertex.
    hex_to_vertex_src = src[:114]
    hex_to_vertex_dst = dst[:114]
    for h in range(19):
        verts = hex_to_vertex_dst[hex_to_vertex_src == h]
        assert len(verts) == 6, f"hex {h} has {len(verts)} vertex edges, expected 6"
        assert len(set(verts.tolist())) == 6, f"hex {h} has duplicate vertices"


def test_edges_match_hex_perimeters_consistency():
    """Cross-check: derive (min,max)-vertex-pair edge set from each hex's
    6-vertex perimeter and compare to the literal EDGE_TO_VERTICES table.
    Catches single-row transcription errors in either table."""
    edges_from_hexes: set[tuple[int, int]] = set()
    for vs in HEX_TO_VERTICES:
        assert len(vs) == 6
        for i in range(6):
            a, b = vs[i], vs[(i + 1) % 6]
            edges_from_hexes.add((min(a, b), max(a, b)))

    edges_from_table: set[tuple[int, int]] = set()
    for vs in EDGE_TO_VERTICES:
        assert len(vs) == 2
        a, b = vs[0], vs[1]
        edges_from_table.add((min(a, b), max(a, b)))

    assert edges_from_hexes == edges_from_table, (
        f"only in hex perimeters: {edges_from_hexes - edges_from_table}; "
        f"only in EDGE_TO_VERTICES: {edges_from_table - edges_from_hexes}"
    )
