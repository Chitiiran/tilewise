//! Immutable board topology for the standard 19-hex Catan board.
//! Built once at engine startup; never mutated.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resource {
    Wood,
    Brick,
    Sheep,
    Wheat,
    Ore,
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

fn standard_hex_to_vertices() -> [[u8; 6]; 19] {
    // Canonical vertex IDs per hex, in clockwise order starting from top.
    // PLACEHOLDER VALUES — Task 6 replaces these with the verified canonical tables.
    [
        [0, 1, 2, 8, 7, 6],         // hex 0
        [2, 3, 4, 10, 9, 8],        // hex 1
        [4, 5, 13, 12, 11, 10],     // hex 2
        [6, 7, 16, 15, 14, 22],     // hex 3
        [7, 8, 9, 17, 16, 18],      // (placeholder values — Task 6 replaces)
        [9, 10, 11, 19, 18, 17],
        [11, 12, 13, 20, 19, 21],
        [14, 15, 23, 22, 28, 27],
        [15, 16, 17, 24, 23, 25],
        [17, 18, 19, 26, 25, 24],   // desert in standard, but topology same
        [19, 20, 21, 31, 30, 26],
        [21, 13, 32, 31, 33, 34],
        [27, 28, 35, 36, 41, 40],
        [28, 29, 37, 36, 35, 38],
        [29, 30, 39, 38, 37, 42],
        [30, 31, 32, 43, 42, 39],
        [32, 33, 44, 43, 47, 46],
        [40, 41, 49, 48, 53, 52],
        [42, 39, 50, 49, 48, 51],
    ]
}

fn standard_edge_to_vertices() -> [[u8; 2]; 72] {
    // PLACEHOLDER — Task 6 replaces with the verified canonical table.
    // Each edge connects exactly 2 vertices.
    [[0, 0]; 72]
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
