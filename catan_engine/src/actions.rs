//! Action enum + ID encoding. IDs are stable across versions per spec §4.

use crate::board::Resource;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    BuildSettlement(u8), // vertex 0..54
    BuildCity(u8),       // vertex 0..54
    BuildRoad(u8),       // edge 0..72
    MoveRobber(u8),      // hex 0..19
    Discard(Resource),   // dead in v2 (instant random discard) — kept for back-compat
    EndTurn,
    RollDice,
    /// v2 maritime trade: give N of `give` to bank, receive 1 of `get`.
    /// N is 4:1 if no port; 3:1 with generic port; 2:1 with the specific resource port.
    /// Same-for-same is excluded.
    TradeBank { give: Resource, get: Resource },
}

/// v1: 206 (settlements + cities + roads + robber + discard + end + roll).
/// v2: 226 (adds 20 TradeBank actions at 206-225). Same-for-same trades are illegal.
pub const ACTION_SPACE_SIZE: usize = 226;
pub const TRADE_BANK_BASE: u32 = 206;
pub const N_TRADE_BANK_ACTIONS: u32 = 20; // 5 give × 4 valid get

/// Number of u64 words needed to hold the legal-action bitmap.
/// Sized to support v2's expanded action space (dev cards + trades) without
/// changing the API. 4 × 64 = 256 bits, room for ~50 more actions beyond v1's 206.
pub const LEGAL_MASK_WORDS: usize = 4;
pub const LEGAL_MASK_BITS: usize = LEGAL_MASK_WORDS * 64;

/// Legal-action bitmap. Bit i set ⇔ action i is legal in the current state.
///
/// **Today (Phase 1.2 deliverable):** computed lazily from `legal_actions()` —
/// no perf win yet, just an API that downstream tools (GNN policy mask, MCTS
/// children enumeration) can use directly without rebuilding bool[N] arrays.
///
/// **Future (Phase 2+):** incremental updates on every state mutation, so this
/// becomes O(1) lookup. See design doc §B-bis 11a-g.
#[derive(Debug, Clone, Copy, Default)]
pub struct LegalMask {
    pub words: [u64; LEGAL_MASK_WORDS],
}

impl LegalMask {
    pub fn new() -> Self { Self::default() }

    /// Build from a list of legal action IDs. Used by current `engine.legal_mask()`
    /// PyO3 method and the slow-path fallback for incremental updates.
    pub fn from_action_ids(ids: &[u32]) -> Self {
        let mut m = Self::new();
        for &id in ids {
            m.set(id as usize);
        }
        m
    }

    #[inline]
    pub fn set(&mut self, action_id: usize) {
        debug_assert!(action_id < LEGAL_MASK_BITS, "action_id {} out of range", action_id);
        let word = action_id / 64;
        let bit = action_id % 64;
        self.words[word] |= 1u64 << bit;
    }

    #[inline]
    pub fn get(&self, action_id: usize) -> bool {
        if action_id >= LEGAL_MASK_BITS { return false; }
        let word = action_id / 64;
        let bit = action_id % 64;
        (self.words[word] >> bit) & 1 == 1
    }

    /// Count of legal actions (popcount).
    pub fn count(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    /// Iterate set bits as action IDs.
    pub fn iter_ids(&self) -> impl Iterator<Item = u32> + '_ {
        (0..LEGAL_MASK_BITS).filter(move |&id| self.get(id)).map(|id| id as u32)
    }
}

pub fn encode(action: Action) -> u32 {
    match action {
        Action::BuildSettlement(v) => v as u32,
        Action::BuildCity(v) => 54 + v as u32,
        Action::BuildRoad(e) => 108 + e as u32,
        Action::MoveRobber(h) => 180 + h as u32,
        Action::Discard(r) => 199 + resource_index(r) as u32,
        Action::EndTurn => 204,
        Action::RollDice => 205,
        Action::TradeBank { give, get } => {
            let g = resource_index(give) as u32;
            let r = resource_index(get) as u32;
            assert!(g != r, "TradeBank give and get must differ");
            // Compact "get index" within 0..4: skip the value equal to give.
            let get_compact = if r < g { r } else { r - 1 };
            TRADE_BANK_BASE + g * 4 + get_compact
        }
    }
}

pub fn decode(id: u32) -> Option<Action> {
    match id {
        0..=53 => Some(Action::BuildSettlement(id as u8)),
        54..=107 => Some(Action::BuildCity((id - 54) as u8)),
        108..=179 => Some(Action::BuildRoad((id - 108) as u8)),
        180..=198 => Some(Action::MoveRobber((id - 180) as u8)),
        199..=203 => Some(Action::Discard(resource_from_index((id - 199) as u8))),
        204 => Some(Action::EndTurn),
        205 => Some(Action::RollDice),
        206..=225 => {
            let off = id - TRADE_BANK_BASE;
            let g = (off / 4) as u8;
            let get_compact = (off % 4) as u8;
            // Re-expand: get is get_compact if get_compact < g, else get_compact + 1.
            let r = if get_compact < g { get_compact } else { get_compact + 1 };
            Some(Action::TradeBank {
                give: resource_from_index(g),
                get: resource_from_index(r),
            })
        }
        _ => None,
    }
}

fn resource_index(r: Resource) -> u8 {
    match r {
        Resource::Wood => 0,
        Resource::Brick => 1,
        Resource::Sheep => 2,
        Resource::Wheat => 3,
        Resource::Ore => 4,
    }
}

fn resource_from_index(i: u8) -> Resource {
    match i {
        0 => Resource::Wood,
        1 => Resource::Brick,
        2 => Resource::Sheep,
        3 => Resource::Wheat,
        4 => Resource::Ore,
        _ => panic!("invalid resource index {i}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legal_mask_set_get() {
        let mut m = LegalMask::new();
        assert_eq!(m.count(), 0);
        m.set(0);
        m.set(63);
        m.set(64);
        m.set(205);
        assert!(m.get(0));
        assert!(m.get(63));
        assert!(m.get(64));
        assert!(m.get(205));
        assert!(!m.get(1));
        assert!(!m.get(206));
        assert_eq!(m.count(), 4);
    }

    #[test]
    fn legal_mask_iter_ids_returns_set_bits() {
        let m = LegalMask::from_action_ids(&[0, 5, 100, 205]);
        let ids: Vec<u32> = m.iter_ids().collect();
        assert_eq!(ids, vec![0, 5, 100, 205]);
    }

    #[test]
    fn legal_mask_round_trip_via_action_ids() {
        // Pretend a few specific actions are legal.
        let ids = vec![0, 53, 54, 107, 108, 179, 180, 199, 204, 205];
        let m = LegalMask::from_action_ids(&ids);
        let recovered: Vec<u32> = m.iter_ids().collect();
        assert_eq!(recovered, ids);
        assert_eq!(m.count(), ids.len() as u32);
    }
}
