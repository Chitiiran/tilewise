//! Mutable game state. Internal representation; hidden info filtered at observation time.

use crate::board::Board;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GamePhase {
    /// First settlement+road placement, in player order 0..N_PLAYERS.
    Setup1Place,
    /// Second settlement+road placement, in REVERSE player order.
    /// Triggers starting-resource production for the second settlement.
    Setup2Place,
    /// Current player must roll the dice.
    Roll,
    /// Current player builds / ends turn.
    Main,
    /// After a 7 roll: each player with hand>7 must discard half.
    /// `remaining[i]` = number of cards player i still needs to discard.
    Discard { remaining: [u8; 4] },
    /// Current player must move the robber.
    MoveRobber,
    /// Current player picks a steal target from `from_options`.
    /// If `from_options` is empty, no steal happens (auto-skipped in step()).
    Steal { from_options: Vec<u8> },
    /// Game over.
    Done { winner: u8 },
}

pub const N_PLAYERS: usize = 4;
pub const N_RESOURCES: usize = 5;
pub const RESOURCE_SUPPLY: u8 = 19;
pub const WIN_VP: u8 = 10;

#[derive(Debug, Clone)]
pub struct GameState {
    pub board: Arc<Board>,
    pub settlements: [Option<u8>; 54], // owner per vertex
    pub cities: [Option<u8>; 54],
    pub roads: [Option<u8>; 72],
    pub robber_hex: u8,
    pub hands: [[u8; N_RESOURCES]; N_PLAYERS],
    pub bank: [u8; N_RESOURCES],
    pub vp: [u8; N_PLAYERS],
    pub turn: u32,
    pub current_player: u8,
    pub phase: GamePhase,
    /// Setup-phase tracker: (vertex, edge) of the most recent placement,
    /// used to enforce "road must connect to the just-placed settlement".
    pub setup_pending: Option<u8>,
}

impl GameState {
    pub fn new(board: Arc<Board>) -> Self {
        let desert_hex = board.hexes.iter().position(|h| h.resource.is_none()).unwrap_or(0) as u8;
        Self {
            board,
            settlements: [None; 54],
            cities: [None; 54],
            roads: [None; 72],
            robber_hex: desert_hex,
            hands: [[0; N_RESOURCES]; N_PLAYERS],
            bank: [RESOURCE_SUPPLY; N_RESOURCES],
            vp: [0; N_PLAYERS],
            turn: 0,
            current_player: 0,
            phase: GamePhase::Setup1Place,
            setup_pending: None,
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self.phase, GamePhase::Done { .. })
    }
}
