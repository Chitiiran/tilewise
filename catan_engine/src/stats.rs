//! Game statistics, folded incrementally from the event log.
//! Spec §7. schema_version is bumped when the struct layout changes.

use crate::events::GameEvent;

pub const STATS_SCHEMA_VERSION: u32 = 1;
pub const N_PLAYERS: usize = 4;
pub const N_RESOURCES: usize = 5;
pub const N_HEXES: usize = 19;

#[derive(Debug, Clone, Default)]
pub struct PerPlayerStats {
    pub vp_final: u8,
    pub won: bool,
    pub settlements_built: u32,
    pub cities_built: u32,
    pub roads_built: u32,
    pub resources_gained: [u32; N_RESOURCES],
    pub resources_gained_from_robber: [u32; N_RESOURCES],
    pub resources_lost_to_robber: [u32; N_RESOURCES],
    pub resources_lost_to_discard: [u32; N_RESOURCES],
    pub cards_in_hand_max: u32,
    pub times_robbed: u32,
    pub robber_moves: u32,
    pub discards_triggered: u32,
}

#[derive(Debug, Clone)]
pub struct GameStats {
    pub schema_version: u32,
    pub turns_played: u32,
    pub dice_histogram: [u32; 11], // index 0 = roll 2, ..., 10 = roll 12
    pub seven_count: u32,
    pub production_per_hex: [u32; N_HEXES],
    pub total_resources_produced: u32,
    pub winner_player_id: i8,
    pub players: [PerPlayerStats; N_PLAYERS],
}

impl GameStats {
    pub fn new() -> Self {
        Self {
            schema_version: STATS_SCHEMA_VERSION,
            turns_played: 0,
            dice_histogram: [0; 11],
            seven_count: 0,
            production_per_hex: [0; N_HEXES],
            total_resources_produced: 0,
            winner_player_id: -1,
            players: Default::default(),
        }
    }

    pub fn fold_event(&mut self, event: &GameEvent) {
        match *event {
            GameEvent::DiceRolled { roll } => {
                if roll == 7 {
                    self.seven_count += 1;
                } else if (2..=12).contains(&roll) {
                    self.dice_histogram[(roll - 2) as usize] += 1;
                }
            }
            GameEvent::ResourcesProduced { player, hex, resource: _, amount } => {
                self.production_per_hex[hex as usize] += amount as u32;
                self.total_resources_produced += amount as u32;
                self.players[player as usize]
                    .resources_gained[_resource_idx(event)] += amount as u32;
            }
            GameEvent::BuildSettlement { player, .. } => {
                self.players[player as usize].settlements_built += 1;
            }
            GameEvent::BuildCity { player, .. } => {
                self.players[player as usize].cities_built += 1;
            }
            GameEvent::BuildRoad { player, .. } => {
                self.players[player as usize].roads_built += 1;
            }
            GameEvent::RobberMoved { player, .. } => {
                self.players[player as usize].robber_moves += 1;
            }
            GameEvent::Robbed { from, to, resource } => {
                self.players[from as usize].times_robbed += 1;
                if let Some(r) = resource {
                    self.players[from as usize].resources_lost_to_robber[r as usize] += 1;
                    self.players[to as usize].resources_gained_from_robber[r as usize] += 1;
                }
            }
            GameEvent::Discarded { player, resource } => {
                self.players[player as usize].resources_lost_to_discard[resource as usize] += 1;
                self.players[player as usize].discards_triggered += 1;
            }
            GameEvent::TurnEnded { .. } => {
                self.turns_played += 1;
            }
            GameEvent::GameOver { winner } => {
                self.winner_player_id = winner as i8;
                for p in 0..N_PLAYERS {
                    self.players[p].won = p as u8 == winner;
                }
            }
        }
    }
}

fn _resource_idx(e: &GameEvent) -> usize {
    if let GameEvent::ResourcesProduced { resource, .. } = e {
        *resource as usize
    } else { 0 }
}
