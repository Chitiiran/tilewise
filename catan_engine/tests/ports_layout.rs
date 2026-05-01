//! Validate the canonical port layout (P2).
//!
//! Catan rule: each port sits on a perimeter edge of the 19-hex board, the
//! 9 port edges are mutually disjoint, and the kinds are exactly 4 Generic
//! + 5 Specific (one per resource).

use std::collections::HashSet;

use catan_engine::board::{Board, PortKind, Resource};

fn perimeter_edges() -> HashSet<(u8, u8)> {
    // Perimeter = edge belongs to exactly one hex.
    let mut count: std::collections::HashMap<(u8, u8), u32> = Default::default();
    let board = Board::standard();
    for hex_verts in board.hex_to_vertices.iter() {
        for k in 0..6 {
            let a = hex_verts[k];
            let b = hex_verts[(k + 1) % 6];
            let key = if a < b { (a, b) } else { (b, a) };
            *count.entry(key).or_default() += 1;
        }
    }
    count.into_iter().filter(|(_, c)| *c == 1).map(|(k, _)| k).collect()
}

#[test]
fn ports_count_is_9() {
    let board = Board::standard();
    assert_eq!(board.ports.len(), 9);
}

#[test]
fn each_port_has_two_perimeter_vertices() {
    // For each port, both endpoints should appear on a perimeter edge.
    let perim = perimeter_edges();
    let board = Board::standard();
    for (i, port) in board.ports.iter().enumerate() {
        let a = port.vertices[0];
        let b = port.vertices[1];
        let key = if a < b { (a, b) } else { (b, a) };
        assert!(
            perim.contains(&key),
            "port {} edge ({}, {}) is not a perimeter edge", i, a, b,
        );
    }
}

#[test]
fn port_edges_are_disjoint() {
    let board = Board::standard();
    let mut seen_edges: HashSet<(u8, u8)> = HashSet::new();
    let mut seen_vertices: HashSet<u8> = HashSet::new();
    for port in board.ports.iter() {
        let a = port.vertices[0];
        let b = port.vertices[1];
        let key = if a < b { (a, b) } else { (b, a) };
        assert!(seen_edges.insert(key),
            "duplicate port edge ({}, {})", a, b);
        // Vertices may be shared across distinct ports IF the layout
        // places them adjacent (3-of-3 spacing). The official Catan
        // layout doesn't share vertices, so assert disjoint here as a
        // canonical-layout invariant.
        assert!(seen_vertices.insert(a),
            "port vertex {} appears in two ports", a);
        assert!(seen_vertices.insert(b),
            "port vertex {} appears in two ports", b);
    }
}

#[test]
fn port_kinds_match_canonical_layout() {
    let board = Board::standard();
    let mut generic_count = 0;
    let mut specific_count: std::collections::HashMap<Resource, u32> = Default::default();
    for port in board.ports.iter() {
        match port.kind {
            PortKind::Generic => generic_count += 1,
            PortKind::Specific(r) => *specific_count.entry(r).or_default() += 1,
        }
    }
    assert_eq!(generic_count, 4, "expected 4 generic 3:1 ports");
    // Each resource should appear exactly once as a 2:1 specific port.
    for r in [Resource::Wood, Resource::Brick, Resource::Sheep, Resource::Wheat, Resource::Ore] {
        assert_eq!(
            specific_count.get(&r).copied().unwrap_or(0), 1,
            "expected exactly 1 specific port for resource {:?}", r,
        );
    }
}

#[test]
fn port_layout_is_deterministic() {
    // Same seed → same port layout (ports are fixed in standard Catan).
    let b1 = Board::generate_abc(7);
    let b2 = Board::generate_abc(99);
    // Different seeds shuffle resources/numbers but ports are fixed.
    assert_eq!(b1.ports, b2.ports,
        "ports must be the same regardless of board seed");
}
