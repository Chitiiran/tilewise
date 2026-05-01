//! Build numpy-shaped observation tensors from GameState.
//!
//! Spec §5 (v2 expanded scalars).
//!
//! ## Scalar layout (N_SCALARS = 59)
//!
//! All "per-player" blocks are perspective-rotated: index 0 = viewer, then
//! viewer+1, viewer+2, viewer+3 (mod 4). All counts are normalized into [0, 1]
//! by their natural cap so the GNN sees comparable scales.
//!
//! ```text
//!  [ 0.. 5)  viewer hand counts (5)                      raw 0..N (no normalization — small)
//!  [ 5.. 8)  opponent hand sizes (3)                     raw count (sum across resources)
//!  [ 8..12)  VP for all 4 players (4)                    raw 0..10
//!  [12]      turn / 100, clamped                         normalized
//!  [13..21)  phase one-hot (8)                           Setup1, Setup2, Roll, Main,
//!                                                         Discard, MoveRobber, Steal, Done
//!  --- v2 additions ---
//!  [21..26)  viewer dev cards held (5)                   raw count: knight, RB, mono, YOP, VP
//!  [26..30)  longest_road_length (4) per player          / MAX_ROADS (15)
//!  [30..34)  knights_played (4) per player               / 14 (deck size)
//!  [34..38)  settlements_built (4) per player            / MAX_SETTLEMENTS (5)
//!  [38..42)  cities_built (4) per player                 / MAX_CITIES (4)
//!  [42..46)  roads_built (4) per player                  / MAX_ROADS (15)
//!  [46..52)  viewer port flags (6)                       0/1: generic, wood, brick, sheep, wheat, ore
//!  [52]      viewer holds longest road                   0/1
//!  [53]      viewer holds largest army                   0/1
//!  [54..59)  bank counts (5) / RESOURCE_SUPPLY (19)      normalized
//! ```

use crate::state::{
    GameState, GamePhase, MAX_CITIES, MAX_ROADS, MAX_SETTLEMENTS, N_DEV_CARDS, N_PLAYERS,
    N_RESOURCES, PORT_BIT_BRICK, PORT_BIT_GENERIC, PORT_BIT_ORE, PORT_BIT_SHEEP, PORT_BIT_WHEAT,
    PORT_BIT_WOOD, RESOURCE_SUPPLY,
};

pub const F_HEX: usize = 8;     // [Wood,Brick,Sheep,Wheat,Ore one-hot, dice-norm, robber, desert]
pub const F_VERT: usize = 7;    // [empty, settle, city, owner-0..3 perspective]
pub const F_EDGE: usize = 6;    // [empty, road, owner-0..3 perspective]
pub const N_SCALARS: usize = 59;

// Section offsets — exposed for downstream consumers + tests.
pub const SCALAR_HAND: usize = 0;          // [0..5)
pub const SCALAR_OPP_HAND_SIZES: usize = 5;// [5..8)
pub const SCALAR_VP: usize = 8;            // [8..12)
pub const SCALAR_TURN: usize = 12;
pub const SCALAR_PHASE: usize = 13;        // [13..21)
pub const SCALAR_DEV_HELD: usize = 21;     // [21..26)
pub const SCALAR_LR_LEN: usize = 26;       // [26..30)
pub const SCALAR_KNIGHTS: usize = 30;      // [30..34)
pub const SCALAR_SETTL_BUILT: usize = 34;  // [34..38)
pub const SCALAR_CITY_BUILT: usize = 38;   // [38..42)
pub const SCALAR_ROAD_BUILT: usize = 42;   // [42..46)
pub const SCALAR_PORTS: usize = 46;        // [46..52)
pub const SCALAR_LR_HOLDER: usize = 52;
pub const SCALAR_LA_HOLDER: usize = 53;
pub const SCALAR_BANK: usize = 54;         // [54..59)

const KNIGHT_DECK_TOTAL: f32 = 14.0;

pub struct Observation {
    pub hex_features: Vec<f32>,    // shape [19, F_HEX]
    pub vertex_features: Vec<f32>, // shape [54, F_VERT]
    pub edge_features: Vec<f32>,   // shape [72, F_EDGE]
    pub scalars: Vec<f32>,         // shape [N_SCALARS]
    pub legal_mask: Vec<bool>,     // shape [ACTION_SPACE_SIZE]
}

pub fn build_observation(state: &GameState, viewer: u8) -> Observation {
    let mut hex_features = vec![0.0f32; 19 * F_HEX];
    for (i, h) in state.board.hexes.iter().enumerate() {
        if let Some(r) = h.resource {
            hex_features[i * F_HEX + r as usize] = 1.0;
        } else {
            hex_features[i * F_HEX + 7] = 1.0; // desert flag
        }
        if let Some(n) = h.dice_number {
            hex_features[i * F_HEX + 5] = (n as f32 - 7.0) / 5.0;
        }
        if state.robber_hex as usize == i {
            hex_features[i * F_HEX + 6] = 1.0;
        }
    }
    let mut vertex_features = vec![0.0f32; 54 * F_VERT];
    for v in 0..54usize {
        if let Some(p) = state.cities.get(v) {
            vertex_features[v * F_VERT + 2] = 1.0;
            vertex_features[v * F_VERT + 3 + perspective_idx(p, viewer)] = 1.0;
        } else if let Some(p) = state.settlements.get(v) {
            vertex_features[v * F_VERT + 1] = 1.0;
            vertex_features[v * F_VERT + 3 + perspective_idx(p, viewer)] = 1.0;
        } else {
            vertex_features[v * F_VERT] = 1.0;
        }
    }
    let mut edge_features = vec![0.0f32; 72 * F_EDGE];
    for e in 0..72usize {
        if let Some(p) = state.roads.get(e) {
            edge_features[e * F_EDGE + 1] = 1.0;
            edge_features[e * F_EDGE + 2 + perspective_idx(p, viewer)] = 1.0;
        } else {
            edge_features[e * F_EDGE] = 1.0;
        }
    }
    let mut scalars = vec![0.0f32; N_SCALARS];

    // [0..5) viewer hand counts
    for r in 0..N_RESOURCES {
        scalars[SCALAR_HAND + r] = state.hands[viewer as usize][r] as f32;
    }
    // [5..8) opponent hand sizes — perspective-rotated (offsets 1..3)
    for offset in 1..N_PLAYERS as u8 {
        let opp = (viewer + offset) % N_PLAYERS as u8;
        scalars[SCALAR_OPP_HAND_SIZES + (offset as usize - 1)] =
            state.hands[opp as usize].iter().map(|&x| x as f32).sum();
    }
    // [8..12) VP — perspective-rotated (offsets 0..3)
    for offset in 0..N_PLAYERS as u8 {
        let p = (viewer + offset) % N_PLAYERS as u8;
        scalars[SCALAR_VP + offset as usize] = state.vp[p as usize] as f32;
    }
    // [12] turn / 100, clamped
    scalars[SCALAR_TURN] = (state.turn as f32 / 100.0).min(1.0);
    // [13..21) phase one-hot
    let phase_idx = match state.phase {
        GamePhase::Setup1Place => 0,
        GamePhase::Setup2Place => 1,
        GamePhase::Roll => 2,
        GamePhase::Main => 3,
        GamePhase::Discard { .. } => 4,
        GamePhase::MoveRobber => 5,
        GamePhase::Steal { .. } => 6,
        GamePhase::Done { .. } => 7,
    };
    scalars[SCALAR_PHASE + phase_idx] = 1.0;

    // ------------------------------------------------------------ v2 additions
    // [21..26) viewer dev cards held
    for k in 0..N_DEV_CARDS {
        scalars[SCALAR_DEV_HELD + k] = state.dev_cards_held[viewer as usize][k] as f32;
    }
    // [26..30) longest_road_length per player (perspective)
    // [30..34) knights_played per player (perspective)
    // [34..38) settlements_built per player (perspective)
    // [38..42) cities_built per player (perspective)
    // [42..46) roads_built per player (perspective)
    for offset in 0..N_PLAYERS as u8 {
        let p = ((viewer + offset) % N_PLAYERS as u8) as usize;
        let off = offset as usize;
        scalars[SCALAR_LR_LEN + off] = state.longest_road_length[p] as f32 / MAX_ROADS as f32;
        scalars[SCALAR_KNIGHTS + off] = state.knights_played[p] as f32 / KNIGHT_DECK_TOTAL;
        scalars[SCALAR_SETTL_BUILT + off] =
            state.settlements_built[p] as f32 / MAX_SETTLEMENTS as f32;
        scalars[SCALAR_CITY_BUILT + off] = state.cities_built[p] as f32 / MAX_CITIES as f32;
        scalars[SCALAR_ROAD_BUILT + off] = state.roads_built[p] as f32 / MAX_ROADS as f32;
    }
    // [46..52) viewer port ownership (6 bits)
    let pb = state.ports_owned[viewer as usize];
    scalars[SCALAR_PORTS + 0] = if pb & PORT_BIT_GENERIC != 0 { 1.0 } else { 0.0 };
    scalars[SCALAR_PORTS + 1] = if pb & PORT_BIT_WOOD != 0 { 1.0 } else { 0.0 };
    scalars[SCALAR_PORTS + 2] = if pb & PORT_BIT_BRICK != 0 { 1.0 } else { 0.0 };
    scalars[SCALAR_PORTS + 3] = if pb & PORT_BIT_SHEEP != 0 { 1.0 } else { 0.0 };
    scalars[SCALAR_PORTS + 4] = if pb & PORT_BIT_WHEAT != 0 { 1.0 } else { 0.0 };
    scalars[SCALAR_PORTS + 5] = if pb & PORT_BIT_ORE != 0 { 1.0 } else { 0.0 };
    // [52] viewer holds longest road
    if state.longest_road_holder == Some(viewer) {
        scalars[SCALAR_LR_HOLDER] = 1.0;
    }
    // [53] viewer holds largest army
    if state.largest_army_holder == Some(viewer) {
        scalars[SCALAR_LA_HOLDER] = 1.0;
    }
    // [54..59) bank / RESOURCE_SUPPLY
    for r in 0..N_RESOURCES {
        scalars[SCALAR_BANK + r] = state.bank[r] as f32 / RESOURCE_SUPPLY as f32;
    }

    // Legal mask: prefer the cached value if it's been computed; fall back to recompute.
    // The cache is invalidated by step()/apply_chance_outcome(), so reading it here when
    // legal_mask_dirty is false is exact. If dirty, we must recompute to stay consistent.
    let legal_mask = if !state.legal_mask_dirty {
        let mut m = vec![false; crate::actions::ACTION_SPACE_SIZE];
        for id in state.legal_mask_cached.iter_ids() {
            m[id as usize] = true;
        }
        m
    } else {
        let mut m = vec![false; crate::actions::ACTION_SPACE_SIZE];
        for a in crate::rules::legal_actions(state) {
            m[crate::actions::encode(a) as usize] = true;
        }
        m
    };

    Observation { hex_features, vertex_features, edge_features, scalars, legal_mask }
}

fn perspective_idx(player: u8, viewer: u8) -> usize {
    ((player + N_PLAYERS as u8 - viewer) % N_PLAYERS as u8) as usize
}
