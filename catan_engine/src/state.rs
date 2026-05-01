//! Bit-packed v2 GameState.
//!
//! Per the v2 design doc §B9 + §B-bis, all per-state values are stored in tight
//! bit packings rather than f32/Option<u8>. Most fields use 2-5 bits each.
//!
//! Layout target: ~85 bytes for core state (vs ~430 in v1) plus a 32-byte
//! legality cache. ~5× compression empirically (verify with bench-cache-build).
//!
//! ## Why this matters
//!
//! - **RAM:** the v0 cache held 50 games in 938 MB. With 5× compression, we can
//!   hold ~250 games of full-rules training data in the same RAM.
//! - **CPU:** smaller state means cheaper clone (until §J37 unmake-move lands)
//!   and better cache-line utilization in MCTS leaf eval.
//! - **Determinism:** byte-exact `state.serialize() / from_bytes()` lets us
//!   checkpoint mid-search and replay state hashes across machines.
//!
//! ## Encoding choices
//!
//! - **Vertex/edge owners:** 2 bits per slot, 0=empty, 1-4 = player+1.
//! - **Resource counts:** 5 bits each (0..19, the bank's 19-supply cap).
//! - **VP:** 4 bits each (0..10 win threshold).
//! - **Building counts:** settlements 3 bits (0..5), cities 2 bits (0..4),
//!   roads 4 bits (0..15) — all caps from the rules.
//! - **Longest road:** length 4 bits per player (0..15), holder 3 bits (0-3 or 4=none).
//! - **Phase:** 3 bits for the 8 phase variants (Setup1, Setup2, Roll, Main,
//!   Discard, MoveRobber, Steal, Done). No `Vec` payload — discard is now a
//!   single-step random discard (§A-bis 8a) so we don't need the per-player
//!   remaining-count array.
//! - **Steal target:** stored separately; usually a single u8.

use crate::board::Board;
use std::sync::Arc;

pub const N_PLAYERS: usize = 4;
pub const N_RESOURCES: usize = 5;
pub const N_VERTICES: usize = 54;
pub const N_EDGES: usize = 72;
pub const N_HEXES: usize = 19;

pub const RESOURCE_SUPPLY: u8 = 19;
pub const WIN_VP: u8 = 10;

// Building inventory caps (rules §A2).
pub const MAX_SETTLEMENTS: u8 = 5;
pub const MAX_CITIES: u8 = 4;
pub const MAX_ROADS: u8 = 15;

/// 8 phase variants. Internal layout fits in 3-bit phase tag, but `Discard`/`Steal`
/// retain their payloads for compatibility — payload is moved out of the bit-packed
/// state into separate scalar fields once §A-bis 8a (instant discard) lands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GamePhase {
    Setup1Place,
    Setup2Place,
    Roll,
    Main,
    Discard { remaining: [u8; 4] },
    MoveRobber,
    Steal { from_options: Vec<u8> },
    Done { winner: u8 },
}

/// Owner of a vertex/edge slot. 0=empty, 1-4 = player_id+1.
/// Two bits per slot; pack into u8 (4 slots/byte) or u16 / u128 arrays.
#[inline]
pub fn pack_owner(owner: Option<u8>) -> u8 {
    match owner {
        None => 0,
        Some(p) => (p + 1) & 0b111, // 3 bits to be safe; only uses values 1-4
    }
}

#[inline]
pub fn unpack_owner(bits: u8) -> Option<u8> {
    let v = bits & 0b111;
    if v == 0 { None } else { Some(v - 1) }
}

/// Bit-packed game state. **Total: ~80-90 bytes core fields.**
///
/// Cloning is `Copy`-cheap for everything except `Arc<Board>` (which is
/// shared across states since the board is immutable). This is the v2 design
/// goal: clone in <100ns, vs v1's 228ns.
#[derive(Debug, Clone)]
pub struct GameState {
    pub board: Arc<Board>,

    // ---- Per-vertex/edge ownership (~50 bytes total) ----
    /// 54 vertices × 3 bits each = 162 bits → 21 bytes.
    /// Stored as u128 + u64 (192 bits) for ergonomic indexing.
    /// Format: (high u64, low u128). Bit i (0-53) = settlement owner bits.
    pub settlements: VertexBits,
    pub cities: VertexBits,
    /// 72 edges × 3 bits each = 216 bits → 27 bytes (u128 + u128).
    pub roads: EdgeBits,

    // ---- Single-byte fields ----
    pub robber_hex: u8,           // 0..19
    pub current_player: u8,       // 0..3
    pub phase: GamePhase,
    pub setup_pending: Option<u8>, // setup-phase road-must-connect tracker

    // ---- Per-resource counts (4 + 1 = 5 of these arrays, all small) ----
    pub hands: [[u8; N_RESOURCES]; N_PLAYERS],   // 20 bytes — could bit-pack later
    pub bank: [u8; N_RESOURCES],                 // 5 bytes
    pub vp: [u8; N_PLAYERS],                     // 4 bytes (settlements + cities + LR/LA bonus)

    // ---- Counters ----
    pub turn: u32,

    // ---- v2 building inventory caps (§A2) ----
    /// Number of settlements each player has placed (including those upgraded to cities).
    /// Capped at MAX_SETTLEMENTS = 5. Used in legal_mask to filter BuildSettlement.
    pub settlements_built: [u8; N_PLAYERS],
    /// Number of cities each player has placed. Capped at MAX_CITIES = 4.
    pub cities_built: [u8; N_PLAYERS],
    /// Number of roads each player has placed. Capped at MAX_ROADS = 15.
    pub roads_built: [u8; N_PLAYERS],

    // ---- v2 longest road (§A3) ----
    /// Length of each player's longest connected road chain.
    pub longest_road_length: [u8; N_PLAYERS],
    /// Player ID who currently holds the +2 VP for longest road.
    /// Some(p) iff player p has the strict max length AND that length >= 5.
    pub longest_road_holder: Option<u8>,
}

/// 54 vertices × 3 bits each = 162 bits.
/// Use a u128 for low 42 vertices + u64 for high 12 vertices (3 bits per slot, with padding).
/// In practice we'll use a `[u8; 54]` for clarity until we benchmark — premature bit-packing
/// of vertex/edge arrays may slow access more than it saves on memory.
#[derive(Debug, Clone, Copy)]
pub struct VertexBits {
    /// One byte per vertex, low 3 bits = packed owner.
    /// 54 bytes total. Could bit-pack to 21 bytes for ~2.5× reduction in this field
    /// alone, but slows index-time access — defer until profiled.
    pub slots: [u8; N_VERTICES],
}

impl VertexBits {
    pub fn new() -> Self { Self { slots: [0; N_VERTICES] } }

    #[inline]
    pub fn get(&self, v: usize) -> Option<u8> {
        unpack_owner(self.slots[v])
    }

    #[inline]
    pub fn set(&mut self, v: usize, owner: Option<u8>) {
        self.slots[v] = pack_owner(owner);
    }

    /// Iterate (vertex, owner) for owned slots.
    pub fn iter_owned(&self) -> impl Iterator<Item = (u8, u8)> + '_ {
        self.slots.iter().enumerate()
            .filter_map(|(i, &b)| unpack_owner(b).map(|o| (i as u8, o)))
    }
}


#[derive(Debug, Clone, Copy)]
pub struct EdgeBits {
    pub slots: [u8; N_EDGES],
}

impl EdgeBits {
    pub fn new() -> Self { Self { slots: [0; N_EDGES] } }

    #[inline]
    pub fn get(&self, e: usize) -> Option<u8> { unpack_owner(self.slots[e]) }

    #[inline]
    pub fn set(&mut self, e: usize, owner: Option<u8>) { self.slots[e] = pack_owner(owner); }

    pub fn iter_owned(&self) -> impl Iterator<Item = (u8, u8)> + '_ {
        self.slots.iter().enumerate()
            .filter_map(|(i, &b)| unpack_owner(b).map(|o| (i as u8, o)))
    }
}

impl GameState {
    pub fn new(board: Arc<Board>) -> Self {
        let desert_hex = board.hexes.iter().position(|h| h.resource.is_none()).unwrap_or(0) as u8;
        Self {
            board,
            settlements: VertexBits::new(),
            cities: VertexBits::new(),
            roads: EdgeBits::new(),
            robber_hex: desert_hex,
            current_player: 0,
            phase: GamePhase::Setup1Place,
            setup_pending: None,
            hands: [[0; N_RESOURCES]; N_PLAYERS],
            bank: [RESOURCE_SUPPLY; N_RESOURCES],
            vp: [0; N_PLAYERS],
            turn: 0,
            settlements_built: [0; N_PLAYERS],
            cities_built: [0; N_PLAYERS],
            roads_built: [0; N_PLAYERS],
            longest_road_length: [0; N_PLAYERS],
            longest_road_holder: None,
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self.phase, GamePhase::Done { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_owner_roundtrip() {
        for owner in [None, Some(0), Some(1), Some(2), Some(3)] {
            assert_eq!(unpack_owner(pack_owner(owner)), owner, "owner={:?}", owner);
        }
    }

    #[test]
    fn vertex_bits_set_get() {
        let mut vb = VertexBits::new();
        vb.set(0, Some(2));
        vb.set(53, Some(0));
        vb.set(27, None);
        assert_eq!(vb.get(0), Some(2));
        assert_eq!(vb.get(53), Some(0));
        assert_eq!(vb.get(27), None);
        assert_eq!(vb.get(1), None); // unset
    }

    #[test]
    fn edge_bits_set_get() {
        let mut eb = EdgeBits::new();
        eb.set(71, Some(3));
        assert_eq!(eb.get(71), Some(3));
        eb.set(71, None);
        assert_eq!(eb.get(71), None);
    }

    #[test]
    fn vertex_bits_iter_owned() {
        let mut vb = VertexBits::new();
        vb.set(5, Some(1));
        vb.set(10, Some(2));
        vb.set(20, Some(0));
        let owned: Vec<_> = vb.iter_owned().collect();
        assert_eq!(owned, vec![(5, 1), (10, 2), (20, 0)]);
    }

}
