# Catan Board Geometry Tables (canonical reference)

These are the canonical hex/vertex/edge numbering tables for the standard 19-hex Catan board, used in `catan_engine/src/board.rs`. **Do not regenerate or modify these.** They were derived from first principles, verified against 10 topology invariants, and the verification script is at `scratch_catan.py` in the repo root.

## Numbering convention

**Hexes (0..18)** — row-major, top-to-bottom, left-to-right:
- Row 0 (3 hexes): 0, 1, 2
- Row 1 (4 hexes): 3, 4, 5, 6
- Row 2 (5 hexes): 7, 8, 9, 10, 11
- Row 3 (4 hexes): 12, 13, 14, 15
- Row 4 (3 hexes): 16, 17, 18

**Vertices (0..53)** — pointy-top hex orientation; sorted lexicographically by (y, x) screen coordinates (top-to-bottom, then left-to-right within each horizontal band).

**Hex corner ordering** — clockwise starting from the TOP vertex: TOP, UPPER-RIGHT, LOWER-RIGHT, BOTTOM, LOWER-LEFT, UPPER-LEFT.

**Edges (0..71)** — derived from hex perimeters and deduplicated; sorted by (min_vertex, max_vertex). Each row `[a, b]` satisfies `a < b`.

**Vertex degree distribution:**
- 18 corner-of-perimeter vertices: degree 2 (touch 1 hex)
- 12 edge-of-perimeter vertices: degree 3 (touch 2 hexes)
- 24 interior vertices: degree 3 (touch 3 hexes)
- Total edges = (18·2 + 36·3) / 2 = 72 ✓

## Verified invariants

1. 19 rows × 6 unique vertex IDs each
2. Union of `hex_to_vertices` covers exactly {0..53}
3. 72 edges, each with 2 distinct endpoints
4. Union of edge endpoints covers exactly {0..53}
5. Handshake: `Σ |adj_vertices over hexes|` = `Σ |adj_hexes over vertices|` = 114
6. Every vertex appears in 1, 2, or 3 hex rows
7. Planar by construction
8. Adjacency consistency (V ∈ hex H ⇔ H ∈ vertex V's hex list)
9. Every vertex has 2 or 3 incident edges
10. Graph (54 vertices, 72 edges) is connected

Verification script: `scratch_catan.py` (run with `python scratch_catan.py`)

## Rust constants — paste into `board.rs`

```rust
/// Standard 19-hex Catan board: each hex's 6 vertex IDs in clockwise order
/// starting from the TOP vertex (pointy-top hex orientation).
///
/// Hex IDs are assigned row-major, top-to-bottom, left-to-right:
///   Row 0 (3 hexes): 0, 1, 2
///   Row 1 (4 hexes): 3, 4, 5, 6
///   Row 2 (5 hexes): 7, 8, 9, 10, 11
///   Row 3 (4 hexes): 12, 13, 14, 15
///   Row 4 (3 hexes): 16, 17, 18
///
/// Vertex IDs (0..=53) are assigned by sorting all unique hex corners
/// lexicographically by (y, x) screen coordinates — i.e., top-to-bottom,
/// then left-to-right within each horizontal band.
fn standard_hex_to_vertices() -> [[u8; 6]; 19] {
    [
        [ 0,  4,  8, 12,  7,  3], // hex  0
        [ 1,  5,  9, 13,  8,  4], // hex  1
        [ 2,  6, 10, 14,  9,  5], // hex  2
        [ 7, 12, 17, 22, 16, 11], // hex  3
        [ 8, 13, 18, 23, 17, 12], // hex  4
        [ 9, 14, 19, 24, 18, 13], // hex  5
        [10, 15, 20, 25, 19, 14], // hex  6
        [16, 22, 28, 33, 27, 21], // hex  7
        [17, 23, 29, 34, 28, 22], // hex  8
        [18, 24, 30, 35, 29, 23], // hex  9
        [19, 25, 31, 36, 30, 24], // hex 10
        [20, 26, 32, 37, 31, 25], // hex 11
        [28, 34, 39, 43, 38, 33], // hex 12
        [29, 35, 40, 44, 39, 34], // hex 13
        [30, 36, 41, 45, 40, 35], // hex 14
        [31, 37, 42, 46, 41, 36], // hex 15
        [39, 44, 48, 51, 47, 43], // hex 16
        [40, 45, 49, 52, 48, 44], // hex 17
        [41, 46, 50, 53, 49, 45], // hex 18
    ]
}

/// Standard 19-hex Catan board: each edge's 2 endpoint vertex IDs.
///
/// Edges are derived from the hex perimeters (each pair of adjacent corners)
/// and deduplicated, then sorted lexicographically by (min_vertex, max_vertex).
/// Each row [a, b] satisfies a < b.
fn standard_edge_to_vertices() -> [[u8; 2]; 72] {
    [
        [ 0,  3], // edge  0
        [ 0,  4], // edge  1
        [ 1,  4], // edge  2
        [ 1,  5], // edge  3
        [ 2,  5], // edge  4
        [ 2,  6], // edge  5
        [ 3,  7], // edge  6
        [ 4,  8], // edge  7
        [ 5,  9], // edge  8
        [ 6, 10], // edge  9
        [ 7, 11], // edge 10
        [ 7, 12], // edge 11
        [ 8, 12], // edge 12
        [ 8, 13], // edge 13
        [ 9, 13], // edge 14
        [ 9, 14], // edge 15
        [10, 14], // edge 16
        [10, 15], // edge 17
        [11, 16], // edge 18
        [12, 17], // edge 19
        [13, 18], // edge 20
        [14, 19], // edge 21
        [15, 20], // edge 22
        [16, 21], // edge 23
        [16, 22], // edge 24
        [17, 22], // edge 25
        [17, 23], // edge 26
        [18, 23], // edge 27
        [18, 24], // edge 28
        [19, 24], // edge 29
        [19, 25], // edge 30
        [20, 25], // edge 31
        [20, 26], // edge 32
        [21, 27], // edge 33
        [22, 28], // edge 34
        [23, 29], // edge 35
        [24, 30], // edge 36
        [25, 31], // edge 37
        [26, 32], // edge 38
        [27, 33], // edge 39
        [28, 33], // edge 40
        [28, 34], // edge 41
        [29, 34], // edge 42
        [29, 35], // edge 43
        [30, 35], // edge 44
        [30, 36], // edge 45
        [31, 36], // edge 46
        [31, 37], // edge 47
        [32, 37], // edge 48
        [33, 38], // edge 49
        [34, 39], // edge 50
        [35, 40], // edge 51
        [36, 41], // edge 52
        [37, 42], // edge 53
        [38, 43], // edge 54
        [39, 43], // edge 55
        [39, 44], // edge 56
        [40, 44], // edge 57
        [40, 45], // edge 58
        [41, 45], // edge 59
        [41, 46], // edge 60
        [42, 46], // edge 61
        [43, 47], // edge 62
        [44, 48], // edge 63
        [45, 49], // edge 64
        [46, 50], // edge 65
        [47, 51], // edge 66
        [48, 51], // edge 67
        [48, 52], // edge 68
        [49, 52], // edge 69
        [49, 53], // edge 70
        [50, 53], // edge 71
    ]
}
```
