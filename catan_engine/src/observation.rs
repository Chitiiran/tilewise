//! Build numpy-shaped observation tensors from GameState.
//! Spec §5.

use crate::state::{GameState, N_PLAYERS, N_RESOURCES};

pub const F_HEX: usize = 8;     // [Wood,Brick,Sheep,Wheat,Ore one-hot, dice-norm, robber, desert]
pub const F_VERT: usize = 7;    // [empty, settle, city, owner-0..3 perspective]
pub const F_EDGE: usize = 6;    // [empty, road, owner-0..3 perspective]
pub const N_SCALARS: usize = 22;

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
        if let Some(p) = state.cities[v] {
            vertex_features[v * F_VERT + 2] = 1.0;
            vertex_features[v * F_VERT + 3 + perspective_idx(p, viewer)] = 1.0;
        } else if let Some(p) = state.settlements[v] {
            vertex_features[v * F_VERT + 1] = 1.0;
            vertex_features[v * F_VERT + 3 + perspective_idx(p, viewer)] = 1.0;
        } else {
            vertex_features[v * F_VERT] = 1.0;
        }
    }
    let mut edge_features = vec![0.0f32; 72 * F_EDGE];
    for e in 0..72usize {
        if let Some(p) = state.roads[e] {
            edge_features[e * F_EDGE + 1] = 1.0;
            edge_features[e * F_EDGE + 2 + perspective_idx(p, viewer)] = 1.0;
        } else {
            edge_features[e * F_EDGE] = 1.0;
        }
    }
    let mut scalars = vec![0.0f32; N_SCALARS];
    // Viewer's hand counts (5)
    for r in 0..N_RESOURCES { scalars[r] = state.hands[viewer as usize][r] as f32; }
    // Opponent hand sizes (3) — perspective-rotated
    let mut idx = 5;
    for offset in 1..N_PLAYERS as u8 {
        let opp = (viewer + offset) % N_PLAYERS as u8;
        scalars[idx] = state.hands[opp as usize].iter().map(|&x| x as f32).sum();
        idx += 1;
    }
    // VP for all 4 players (perspective order)
    for offset in 0..N_PLAYERS as u8 {
        let p = (viewer + offset) % N_PLAYERS as u8;
        scalars[idx] = state.vp[p as usize] as f32;
        idx += 1;
    }
    // Turn (normalized) + phase one-hot (8)
    scalars[idx] = (state.turn as f32 / 100.0).min(1.0);
    idx += 1;
    let phase_idx = match state.phase {
        crate::state::GamePhase::Setup1Place => 0,
        crate::state::GamePhase::Setup2Place => 1,
        crate::state::GamePhase::Roll => 2,
        crate::state::GamePhase::Main => 3,
        crate::state::GamePhase::Discard { .. } => 4,
        crate::state::GamePhase::MoveRobber => 5,
        crate::state::GamePhase::Steal { .. } => 6,
        crate::state::GamePhase::Done { .. } => 7,
    };
    scalars[idx + phase_idx] = 1.0;

    let legal = crate::rules::legal_actions(state);
    let mut legal_mask = vec![false; crate::actions::ACTION_SPACE_SIZE];
    for a in legal {
        legal_mask[crate::actions::encode(a) as usize] = true;
    }

    Observation { hex_features, vertex_features, edge_features, scalars, legal_mask }
}

fn perspective_idx(player: u8, viewer: u8) -> usize {
    ((player + N_PLAYERS as u8 - viewer) % N_PLAYERS as u8) as usize
}
