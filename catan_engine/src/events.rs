//! Typed event stream — source of truth for stats and replay.

use crate::board::Resource;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameEvent {
    DiceRolled        { roll: u8 },
    ResourcesProduced { player: u8, hex: u8, resource: Resource, amount: u8 },
    BuildSettlement   { player: u8, vertex: u8 },
    BuildCity         { player: u8, vertex: u8 },
    BuildRoad         { player: u8, edge: u8 },
    RobberMoved       { player: u8, from_hex: u8, to_hex: u8 },
    Robbed            { from: u8, to: u8, resource: Option<Resource> },
    Discarded         { player: u8, resource: Resource },
    TurnEnded         { player: u8 },
    GameOver          { winner: u8 },
}

#[derive(Clone)]
pub struct EventLog {
    events: Vec<GameEvent>,
}

impl EventLog {
    pub fn new() -> Self {
        Self { events: Vec::with_capacity(1024) }
    }

    pub fn push(&mut self, e: GameEvent) {
        self.events.push(e);
    }

    pub fn as_slice(&self) -> &[GameEvent] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }
}

impl Default for EventLog {
    fn default() -> Self { Self::new() }
}
