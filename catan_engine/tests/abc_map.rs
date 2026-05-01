//! Phase 1.6: ABC balanced map generation tests.

use catan_engine::board::{Board, Resource};

#[test]
fn abc_resource_counts_match_official_distribution() {
    for seed in [0u64, 1, 7, 42, 100, 999, 1_000_000] {
        let b = Board::generate_abc(seed);
        let mut counts = [0u32; 6]; // wood, brick, sheep, wheat, ore, desert
        for h in &b.hexes {
            match h.resource {
                Some(Resource::Wood) => counts[0] += 1,
                Some(Resource::Brick) => counts[1] += 1,
                Some(Resource::Sheep) => counts[2] += 1,
                Some(Resource::Wheat) => counts[3] += 1,
                Some(Resource::Ore) => counts[4] += 1,
                None => counts[5] += 1,
            }
        }
        assert_eq!(counts, [4, 3, 4, 4, 3, 1], "seed={}: resource distribution must be 4w/3b/4s/4wh/3o/1d", seed);
    }
}

#[test]
fn abc_desert_has_no_dice_token() {
    for seed in [0u64, 7, 42] {
        let b = Board::generate_abc(seed);
        for h in &b.hexes {
            if h.resource.is_none() {
                assert!(h.dice_number.is_none(), "desert must have no token");
            } else {
                assert!(h.dice_number.is_some(), "non-desert must have token");
            }
        }
    }
}

#[test]
fn abc_token_sequence_matches_official() {
    // Walk the spiral, gather tokens from non-desert hexes, compare to ABC.
    const SPIRAL: [usize; 19] = [
        0, 1, 2, 6, 11, 15, 18, 17, 16, 12, 7, 3,  // outer
        4, 5, 10, 14, 13, 8,                       // middle
        9,                                         // center
    ];
    let abc = [5u8, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11];
    let b = Board::generate_abc(0);
    let mut seen_tokens = Vec::new();
    for &spiral_idx in &SPIRAL {
        if let Some(t) = b.hexes[spiral_idx].dice_number {
            seen_tokens.push(t);
        }
    }
    assert_eq!(seen_tokens.len(), 18);
    assert_eq!(&seen_tokens[..], &abc[..], "ABC token sequence must match official order along spiral");
}

#[test]
fn abc_distribution_varies_with_seed() {
    let b0 = Board::generate_abc(0);
    let b1 = Board::generate_abc(1);
    // At least one hex should differ in resource between two distinct seeds.
    let same_count = b0.hexes.iter().zip(b1.hexes.iter())
        .filter(|(a, b)| a.resource == b.resource)
        .count();
    assert!(same_count < 19, "different seeds should produce different resource arrangements");
}

#[test]
fn abc_seed_is_deterministic() {
    let b0 = Board::generate_abc(42);
    let b1 = Board::generate_abc(42);
    for i in 0..19 {
        assert_eq!(b0.hexes[i].resource, b1.hexes[i].resource);
        assert_eq!(b0.hexes[i].dice_number, b1.hexes[i].dice_number);
    }
}

/// Sanity check: the ABC method's defining property is that no two red numbers
/// (6 or 8) are adjacent. The test below relies on hex adjacency, which we don't
/// have a direct vertex-free index for — so we approximate via vertex sharing:
/// two hexes are adjacent iff they share at least 2 vertices (an edge).
#[test]
fn abc_no_adjacent_red_numbers() {
    use std::collections::HashSet;
    let b = Board::generate_abc(0);

    // Build hex-adjacency from shared-vertex count.
    let mut adj = vec![Vec::<u8>::new(); 19];
    for i in 0..19u8 {
        let vi: HashSet<u8> = b.hex_to_vertices[i as usize].iter().copied().collect();
        for j in (i + 1)..19u8 {
            let vj: HashSet<u8> = b.hex_to_vertices[j as usize].iter().copied().collect();
            if vi.intersection(&vj).count() >= 2 {
                adj[i as usize].push(j);
                adj[j as usize].push(i);
            }
        }
    }

    let mut violations = 0;
    for i in 0..19usize {
        let ti = b.hexes[i].dice_number;
        if !matches!(ti, Some(6) | Some(8)) { continue; }
        for &j in &adj[i] {
            let tj = b.hexes[j as usize].dice_number;
            if matches!(tj, Some(6) | Some(8)) {
                violations += 1;
                eprintln!("Adjacent reds: hex{} ({:?}) – hex{} ({:?})", i, ti, j, tj);
            }
        }
    }
    // Each adjacency is counted twice (i,j and j,i).
    assert_eq!(violations, 0, "ABC must have no adjacent red numbers");
}
