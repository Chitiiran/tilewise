//! Immutable board topology for the standard 19-hex Catan board.
//! Built once at engine startup; never mutated.

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resource {
    Wood = 0,
    Brick = 1,
    Sheep = 2,
    Wheat = 3,
    Ore = 4,
}

#[derive(Debug, Clone, Copy)]
pub struct Hex {
    pub resource: Option<Resource>, // None = desert
    pub dice_number: Option<u8>,    // None = desert
}

#[derive(Debug)]
pub struct Board {
    pub hexes: [Hex; 19],
    pub vertex_to_hexes: [Vec<u8>; 54],   // each vertex touches 1-3 hexes
    pub vertex_to_vertices: [Vec<u8>; 54], // adjacency for distance rule
    pub vertex_to_edges: [Vec<u8>; 54],
    pub edge_to_vertices: [[u8; 2]; 72],
    pub hex_to_vertices: [[u8; 6]; 19],
    pub dice_to_hexes: [Vec<u8>; 13],     // index 0,1 unused; 2..=12 used
}

impl Board {
    pub fn standard() -> Self {
        let hexes = standard_hexes();
        let hex_to_vertices = standard_hex_to_vertices();
        let edge_to_vertices = standard_edge_to_vertices();

        let vertex_to_hexes = invert_hex_to_vertices(&hex_to_vertices);
        let vertex_to_edges = invert_edge_to_vertices(&edge_to_vertices);
        let vertex_to_vertices = compute_vertex_adjacency(&edge_to_vertices);
        let dice_to_hexes = compute_dice_to_hexes(&hexes);

        Self {
            hexes,
            vertex_to_hexes,
            vertex_to_vertices,
            vertex_to_edges,
            edge_to_vertices,
            hex_to_vertices,
            dice_to_hexes,
        }
    }
}

fn standard_hexes() -> [Hex; 19] {
    use Resource::*;
    // Standard Catan layout, spiral order from top-left.
    // Resource list and dice-number sequence are the published canonical layout.
    let resources = [
        Some(Ore),   Some(Sheep), Some(Wood),
        Some(Wheat), Some(Brick), Some(Sheep), Some(Brick),
        Some(Wheat), Some(Wood),  None,         Some(Wood),  Some(Ore),
        Some(Wood),  Some(Ore),   Some(Wheat),
        Some(Sheep), Some(Brick), Some(Wheat), Some(Sheep),
    ];
    let numbers = [
        Some(10), Some(2),  Some(9),
        Some(12), Some(6),  Some(4),  Some(10),
        Some(9),  Some(11), None,     Some(3),  Some(8),
        Some(8),  Some(3),  Some(4),
        Some(5),  Some(5),  Some(6),  Some(11),
    ];
    let mut hexes = [Hex { resource: None, dice_number: None }; 19];
    for i in 0..19 {
        hexes[i] = Hex { resource: resources[i], dice_number: numbers[i] };
    }
    hexes
}

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

fn invert_hex_to_vertices(h2v: &[[u8; 6]; 19]) -> [Vec<u8>; 54] {
    let mut v2h: [Vec<u8>; 54] = std::array::from_fn(|_| Vec::new());
    for (hex_id, verts) in h2v.iter().enumerate() {
        for &v in verts {
            v2h[v as usize].push(hex_id as u8);
        }
    }
    v2h
}

fn invert_edge_to_vertices(e2v: &[[u8; 2]; 72]) -> [Vec<u8>; 54] {
    let mut v2e: [Vec<u8>; 54] = std::array::from_fn(|_| Vec::new());
    for (edge_id, verts) in e2v.iter().enumerate() {
        for &v in verts {
            v2e[v as usize].push(edge_id as u8);
        }
    }
    v2e
}

fn compute_vertex_adjacency(e2v: &[[u8; 2]; 72]) -> [Vec<u8>; 54] {
    let mut v2v: [Vec<u8>; 54] = std::array::from_fn(|_| Vec::new());
    for [a, b] in e2v {
        v2v[*a as usize].push(*b);
        v2v[*b as usize].push(*a);
    }
    v2v
}

fn compute_dice_to_hexes(hexes: &[Hex; 19]) -> [Vec<u8>; 13] {
    let mut d2h: [Vec<u8>; 13] = std::array::from_fn(|_| Vec::new());
    for (hex_id, hex) in hexes.iter().enumerate() {
        if let Some(n) = hex.dice_number {
            d2h[n as usize].push(hex_id as u8);
        }
    }
    d2h
}
