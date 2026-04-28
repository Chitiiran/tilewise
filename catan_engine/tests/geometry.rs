use catan_engine::board::Board;

#[test]
fn board_has_19_hexes() {
    let b = Board::standard();
    assert_eq!(b.hexes.len(), 19);
}

#[test]
fn exactly_one_hex_is_desert() {
    let b = Board::standard();
    let deserts = b.hexes.iter().filter(|h| h.resource.is_none()).count();
    assert_eq!(deserts, 1);
}

#[test]
fn desert_has_no_dice_number() {
    let b = Board::standard();
    for h in &b.hexes {
        if h.resource.is_none() {
            assert!(h.dice_number.is_none());
        } else {
            assert!(h.dice_number.is_some());
        }
    }
}

#[test]
fn every_vertex_touches_one_to_three_hexes() {
    let b = Board::standard();
    for (v, hexes) in b.vertex_to_hexes.iter().enumerate() {
        let n = hexes.len();
        assert!(
            (1..=3).contains(&n),
            "vertex {v} has {n} adjacent hexes (expected 1-3)"
        );
    }
}

#[test]
fn every_edge_has_exactly_two_endpoints() {
    let b = Board::standard();
    for (e, verts) in b.edge_to_vertices.iter().enumerate() {
        assert_ne!(verts[0], verts[1], "edge {e} has degenerate endpoints");
    }
}

#[test]
fn handshake_lemma_hex_vertex() {
    let b = Board::standard();
    let h_to_v_total: usize = b.hex_to_vertices.iter().map(|h| h.len()).sum();
    let v_to_h_total: usize = b.vertex_to_hexes.iter().map(|v| v.len()).sum();
    assert_eq!(h_to_v_total, v_to_h_total);
}

#[test]
fn vertex_adjacency_is_symmetric() {
    let b = Board::standard();
    for (a, neighbors) in b.vertex_to_vertices.iter().enumerate() {
        for &n in neighbors {
            assert!(
                b.vertex_to_vertices[n as usize].contains(&(a as u8)),
                "vertex {a}->{n} but not {n}->{a}"
            );
        }
    }
}

#[test]
fn dice_to_hexes_covers_all_non_desert() {
    let b = Board::standard();
    let mapped: usize = b.dice_to_hexes.iter().map(|v| v.len()).sum();
    let non_desert = b.hexes.iter().filter(|h| h.resource.is_some()).count();
    assert_eq!(mapped, non_desert);
}

#[test]
fn no_two_red_numbers_adjacent_on_standard_board() {
    // Red numbers (6 and 8) must not be on adjacent hexes per Catan setup rules.
    // This validates our ported standard layout.
    let b = Board::standard();
    let red_hexes: Vec<u8> = b.hexes.iter().enumerate()
        .filter(|(_, h)| matches!(h.dice_number, Some(6) | Some(8)))
        .map(|(i, _)| i as u8)
        .collect();
    // Two hexes are adjacent iff they share at least 2 vertices.
    for &a in &red_hexes {
        for &c in &red_hexes {
            if a == c { continue; }
            let av: std::collections::HashSet<u8> =
                b.hex_to_vertices[a as usize].iter().copied().collect();
            let cv: std::collections::HashSet<u8> =
                b.hex_to_vertices[c as usize].iter().copied().collect();
            let shared = av.intersection(&cv).count();
            assert!(shared < 2, "hexes {a} and {c} are adjacent and both have red numbers");
        }
    }
}
