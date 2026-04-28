//! Pure functions over GameState. No I/O, no globals.
//! Every rule is unit-testable by constructing a state and calling the function.

use crate::actions::Action;
use crate::board::Resource;
use crate::events::GameEvent;
use crate::rng::Rng;
use crate::state::GameState;
use crate::state::GamePhase;

pub fn legal_actions(state: &GameState) -> Vec<Action> {
    // Implemented incrementally across Phases 4–5, dispatching on state.phase.
    // Each phase variant gets its own helper below.
    match &state.phase {
        crate::state::GamePhase::Setup1Place => legal_actions_setup_place(state),
        crate::state::GamePhase::Setup2Place => legal_actions_setup_place(state),
        crate::state::GamePhase::Roll => vec![Action::EndTurn],
        crate::state::GamePhase::Main => legal_actions_main(state),
        crate::state::GamePhase::Discard { remaining } => {
            // The "current discarder" is the lowest-indexed player still owing cards.
            let p = remaining.iter().position(|&n| n > 0).unwrap_or(0);
            let mut out = Vec::new();
            for r in 0..5u8 {
                if state.hands[p][r as usize] > 0 {
                    out.push(Action::Discard(match r {
                        0 => Resource::Wood, 1 => Resource::Brick, 2 => Resource::Sheep,
                        3 => Resource::Wheat, 4 => Resource::Ore, _ => unreachable!()
                    }));
                }
            }
            out
        }
        crate::state::GamePhase::MoveRobber => vec![],     // filled in Task 19
        crate::state::GamePhase::Steal { .. } => vec![],   // filled in Task 19
        crate::state::GamePhase::Done { .. } => vec![],
    }
}

pub(crate) fn legal_actions_setup_place(state: &GameState) -> Vec<Action> {
    // Two sub-states:
    //   setup_pending == None  → place a settlement
    //   setup_pending == Some(v) → place a road adjacent to v
    match state.setup_pending {
        None => legal_setup_settlements(state),
        Some(v) => legal_setup_roads(state, v),
    }
}

fn legal_setup_settlements(state: &GameState) -> Vec<Action> {
    let mut out = Vec::with_capacity(54);
    for v in 0u8..54 {
        if is_legal_settlement_location(state, v) {
            out.push(Action::BuildSettlement(v));
        }
    }
    out
}

fn legal_setup_roads(state: &GameState, just_placed_vertex: u8) -> Vec<Action> {
    let mut out = Vec::new();
    for &e in &state.board.vertex_to_edges[just_placed_vertex as usize] {
        if state.roads[e as usize].is_none() {
            out.push(Action::BuildRoad(e));
        }
    }
    out
}

fn is_legal_settlement_location(state: &GameState, v: u8) -> bool {
    if state.settlements[v as usize].is_some() || state.cities[v as usize].is_some() {
        return false;
    }
    // Distance rule: no neighbor vertex may have a settlement or city.
    for &n in &state.board.vertex_to_vertices[v as usize] {
        if state.settlements[n as usize].is_some() || state.cities[n as usize].is_some() {
            return false;
        }
    }
    true
}

pub fn apply(state: &mut GameState, action: Action, rng: &mut Rng) -> Vec<GameEvent> {
    let mut events = Vec::new();
    match (&state.phase, action) {
        (GamePhase::Setup1Place, Action::BuildSettlement(v)) => {
            state.settlements[v as usize] = Some(state.current_player);
            state.vp[state.current_player as usize] += 1;
            state.setup_pending = Some(v);
            events.push(GameEvent::BuildSettlement { player: state.current_player, vertex: v });
        }
        (GamePhase::Setup1Place, Action::BuildRoad(e)) => {
            state.roads[e as usize] = Some(state.current_player);
            state.setup_pending = None;
            events.push(GameEvent::BuildRoad { player: state.current_player, edge: e });
            // Advance player; if all 4 placed, enter Setup-2 (reverse order, player 3 first).
            if state.current_player < 3 {
                state.current_player += 1;
            } else {
                state.phase = GamePhase::Setup2Place;
                // current_player stays at 3 — Setup-2 starts with player 3.
            }
        }
        (GamePhase::Setup2Place, Action::BuildSettlement(v)) => {
            state.settlements[v as usize] = Some(state.current_player);
            state.vp[state.current_player as usize] += 1;
            state.setup_pending = Some(v);
            events.push(GameEvent::BuildSettlement { player: state.current_player, vertex: v });
            // Yield starting resources: 1 card per non-desert hex adjacent.
            let board = state.board.clone();
            for &h in &board.vertex_to_hexes[v as usize] {
                if let Some(res) = board.hexes[h as usize].resource {
                    let ri = res as usize;
                    if state.bank[ri] > 0 {
                        state.bank[ri] -= 1;
                        state.hands[state.current_player as usize][ri] += 1;
                        events.push(GameEvent::ResourcesProduced {
                            player: state.current_player,
                            hex: h,
                            resource: res,
                            amount: 1,
                        });
                    }
                }
            }
        }
        (GamePhase::Setup2Place, Action::BuildRoad(e)) => {
            state.roads[e as usize] = Some(state.current_player);
            state.setup_pending = None;
            events.push(GameEvent::BuildRoad { player: state.current_player, edge: e });
            // Reverse-order advance; when player 0 finishes, transition to Roll.
            if state.current_player > 0 {
                state.current_player -= 1;
            } else {
                state.phase = GamePhase::Roll;
                // current_player stays at 0 — main game starts with player 0.
            }
        }
        (GamePhase::Roll, Action::EndTurn) => {
            // "EndTurn" in Roll phase = "roll the dice" (Task 15 design choice:
            // single action means no separate RollDice ID needed).
            use rand::Rng as _;
            let d1 = rng.inner().gen_range(1u8..=6);
            let d2 = rng.inner().gen_range(1u8..=6);
            let roll = d1 + d2;
            events.push(GameEvent::DiceRolled { roll });
            if roll == 7 {
                // Discard + robber sub-flow handled in Tasks 18-19.
                let mut remaining = [0u8; 4];
                for p in 0..4usize {
                    let total: u8 = state.hands[p].iter().sum();
                    if total > 7 {
                        remaining[p] = total / 2;
                    }
                }
                state.phase = if remaining.iter().any(|&n| n > 0) {
                    GamePhase::Discard { remaining }
                } else {
                    GamePhase::MoveRobber
                };
            } else {
                produce_resources(state, roll, &mut events);
                state.phase = GamePhase::Main;
            }
        }
        (GamePhase::Main, Action::BuildSettlement(v)) => {
            let p = state.current_player;
            let mut bank = state.bank;
            let mut hand = state.hands[p as usize];
            pay(&mut hand, &mut bank, &SETTLEMENT_COST);
            state.hands[p as usize] = hand;
            state.bank = bank;
            state.settlements[v as usize] = Some(p);
            state.vp[p as usize] += 1;
            events.push(GameEvent::BuildSettlement { player: p, vertex: v });
            check_win(state, &mut events);
        }
        (GamePhase::Main, Action::BuildCity(v)) => {
            let p = state.current_player;
            let mut bank = state.bank;
            let mut hand = state.hands[p as usize];
            pay(&mut hand, &mut bank, &CITY_COST);
            state.hands[p as usize] = hand;
            state.bank = bank;
            state.settlements[v as usize] = None;
            state.cities[v as usize] = Some(p);
            state.vp[p as usize] += 1; // settlement was 1VP, city is 2VP, net +1
            events.push(GameEvent::BuildCity { player: p, vertex: v });
            check_win(state, &mut events);
        }
        (GamePhase::Main, Action::BuildRoad(e)) => {
            let p = state.current_player;
            let mut bank = state.bank;
            let mut hand = state.hands[p as usize];
            pay(&mut hand, &mut bank, &ROAD_COST);
            state.hands[p as usize] = hand;
            state.bank = bank;
            state.roads[e as usize] = Some(p);
            events.push(GameEvent::BuildRoad { player: p, edge: e });
        }
        (GamePhase::Main, Action::EndTurn) => {
            events.push(GameEvent::TurnEnded { player: state.current_player });
            state.current_player = (state.current_player + 1) % 4;
            state.turn += 1;
            state.phase = GamePhase::Roll;
        }
        (GamePhase::Discard { remaining }, Action::Discard(r)) => {
            let mut rem = *remaining;
            let p = rem.iter().position(|&n| n > 0).unwrap();
            let ri = r as usize;
            assert!(state.hands[p][ri] > 0, "discarding resource player doesn't have");
            state.hands[p][ri] -= 1;
            state.bank[ri] += 1;
            rem[p] -= 1;
            events.push(GameEvent::Discarded { player: p as u8, resource: r });
            state.phase = if rem.iter().all(|&n| n == 0) {
                GamePhase::MoveRobber
            } else {
                GamePhase::Discard { remaining: rem }
            };
        }
        _ => {
            // Other transitions implemented in Tasks 13–20.
            let _ = rng;
        }
    }
    events
}

pub fn is_terminal(state: &GameState) -> bool {
    state.is_terminal()
}

fn produce_resources(state: &mut GameState, roll: u8, events: &mut Vec<GameEvent>) {
    let board = state.board.clone();
    let hexes_for_roll = &board.dice_to_hexes[roll as usize];
    for &h in hexes_for_roll {
        if h == state.robber_hex { continue; }
        let res = match board.hexes[h as usize].resource {
            Some(r) => r,
            None => continue,
        };
        let ri = res as usize;
        for &v in &board.hex_to_vertices[h as usize] {
            // Settlement = 1 card; city = 2.
            let owner_qty = match (state.settlements[v as usize], state.cities[v as usize]) {
                (_, Some(p)) => Some((p, 2u8)),
                (Some(p), None) => Some((p, 1u8)),
                _ => None,
            };
            if let Some((p, qty)) = owner_qty {
                let take = qty.min(state.bank[ri]);
                if take > 0 {
                    state.bank[ri] -= take;
                    state.hands[p as usize][ri] += take;
                    events.push(GameEvent::ResourcesProduced {
                        player: p, hex: h, resource: res, amount: take,
                    });
                }
            }
        }
    }
}

const SETTLEMENT_COST: [u8; 5] = [1, 1, 1, 1, 0]; // wood,brick,sheep,wheat,ore
const CITY_COST: [u8; 5] =       [0, 0, 0, 2, 3];
const ROAD_COST: [u8; 5] =       [1, 1, 0, 0, 0];

fn can_afford(hand: &[u8; 5], cost: &[u8; 5]) -> bool {
    hand.iter().zip(cost).all(|(h, c)| h >= c)
}

fn pay(hand: &mut [u8; 5], bank: &mut [u8; 5], cost: &[u8; 5]) {
    for i in 0..5 {
        hand[i] -= cost[i];
        bank[i] += cost[i];
    }
}

fn legal_actions_main(state: &GameState) -> Vec<Action> {
    let mut out = vec![Action::EndTurn];
    let p = state.current_player;
    let hand = &state.hands[p as usize];
    if can_afford(hand, &SETTLEMENT_COST) {
        for v in 0u8..54 {
            if is_legal_settlement_for_player(state, v, p) {
                out.push(Action::BuildSettlement(v));
            }
        }
    }
    if can_afford(hand, &CITY_COST) {
        for v in 0u8..54 {
            if state.settlements[v as usize] == Some(p) {
                out.push(Action::BuildCity(v));
            }
        }
    }
    if can_afford(hand, &ROAD_COST) {
        for e in 0u8..72 {
            if is_legal_road_for_player(state, e, p) {
                out.push(Action::BuildRoad(e));
            }
        }
    }
    out
}

fn is_legal_settlement_for_player(state: &GameState, v: u8, p: u8) -> bool {
    if !is_legal_settlement_location(state, v) { return false; }
    // Main-phase rule: must be adjacent to one of the player's roads.
    state.board.vertex_to_edges[v as usize].iter()
        .any(|&e| state.roads[e as usize] == Some(p))
}

fn is_legal_road_for_player(state: &GameState, e: u8, p: u8) -> bool {
    if state.roads[e as usize].is_some() { return false; }
    // Connects to one of the player's roads or settlements/cities.
    let [a, b] = state.board.edge_to_vertices[e as usize];
    for v in [a, b] {
        if state.settlements[v as usize] == Some(p) || state.cities[v as usize] == Some(p) {
            return true;
        }
        // Or one of the OTHER edges adjacent to this vertex is the player's road,
        // unless that vertex is occupied by another player (blocking).
        let blocked = matches!(state.settlements[v as usize], Some(o) if o != p)
                   || matches!(state.cities[v as usize], Some(o) if o != p);
        if !blocked {
            for &e2 in &state.board.vertex_to_edges[v as usize] {
                if e2 != e && state.roads[e2 as usize] == Some(p) {
                    return true;
                }
            }
        }
    }
    false
}

fn check_win(state: &mut GameState, events: &mut Vec<GameEvent>) {
    for p in 0..4u8 {
        if state.vp[p as usize] >= crate::state::WIN_VP {
            state.phase = GamePhase::Done { winner: p };
            events.push(GameEvent::GameOver { winner: p });
            return;
        }
    }
}
