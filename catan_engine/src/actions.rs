//! Action enum + ID encoding. IDs are stable across versions per spec §4.

use crate::board::Resource;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    BuildSettlement(u8), // vertex 0..54
    BuildCity(u8),       // vertex 0..54
    BuildRoad(u8),       // edge 0..72
    MoveRobber(u8),      // hex 0..19
    Discard(Resource),
    EndTurn,
    RollDice,
}

pub const ACTION_SPACE_SIZE: usize = 206;

pub fn encode(action: Action) -> u32 {
    match action {
        Action::BuildSettlement(v) => v as u32,
        Action::BuildCity(v) => 54 + v as u32,
        Action::BuildRoad(e) => 108 + e as u32,
        Action::MoveRobber(h) => 180 + h as u32,
        Action::Discard(r) => 199 + resource_index(r) as u32,
        Action::EndTurn => 204,
        Action::RollDice => 205,
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
