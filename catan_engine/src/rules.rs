//! Pure functions over GameState. No I/O, no globals.
//! Every rule is unit-testable by constructing a state and calling the function.

use crate::actions::Action;
use crate::board::Resource;
use crate::events::GameEvent;
use crate::rng::Rng;
use crate::state::{GameState, GamePhase, N_RESOURCES};

pub fn legal_actions(state: &GameState) -> Vec<Action> {
    // Implemented incrementally across Phases 4–5, dispatching on state.phase.
    // Each phase variant gets its own helper below.
    match &state.phase {
        crate::state::GamePhase::Setup1Place => legal_actions_setup_place(state),
        crate::state::GamePhase::Setup2Place => legal_actions_setup_place(state),
        crate::state::GamePhase::Roll => vec![Action::RollDice],
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
        crate::state::GamePhase::MoveRobber => {
            (0u8..19)
                .filter(|&h| h != state.robber_hex)
                .map(Action::MoveRobber)
                .collect()
        }
        crate::state::GamePhase::Steal { .. } => {
            // Chance node — environment supplies the outcome via apply_chance_outcome().
            // Players have no legal actions here.
            vec![]
        }
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
        if state.roads.get(e as usize).is_none() {
            out.push(Action::BuildRoad(e));
        }
    }
    out
}

fn is_legal_settlement_location(state: &GameState, v: u8) -> bool {
    if state.settlements.get(v as usize).is_some() || state.cities.get(v as usize).is_some() {
        return false;
    }
    // Distance rule: no neighbor vertex may have a settlement or city.
    for &n in &state.board.vertex_to_vertices[v as usize] {
        if state.settlements.get(n as usize).is_some() || state.cities.get(n as usize).is_some() {
            return false;
        }
    }
    true
}

pub fn apply(state: &mut GameState, action: Action, rng: &mut Rng) -> Vec<GameEvent> {
    let mut events = Vec::new();
    match (&state.phase, action) {
        (GamePhase::Setup1Place, Action::BuildSettlement(v)) => {
            let p = state.current_player;
            state.settlements.set(v as usize, Some(p));
            state.settlements_built[p as usize] += 1;
            state.vp[p as usize] += 1;
            state.setup_pending = Some(v);
            events.push(GameEvent::BuildSettlement { player: p, vertex: v });
        }
        (GamePhase::Setup1Place, Action::BuildRoad(e)) => {
            let p = state.current_player;
            state.roads.set(e as usize, Some(p));
            state.roads_built[p as usize] += 1;
            state.setup_pending = None;
            events.push(GameEvent::BuildRoad { player: p, edge: e });
            update_longest_road(state);
            // Advance player; if all 4 placed, enter Setup-2 (reverse order, player 3 first).
            if state.current_player < 3 {
                state.current_player += 1;
            } else {
                state.phase = GamePhase::Setup2Place;
                // current_player stays at 3 — Setup-2 starts with player 3.
            }
        }
        (GamePhase::Setup2Place, Action::BuildSettlement(v)) => {
            let p = state.current_player;
            state.settlements.set(v as usize, Some(p));
            state.settlements_built[p as usize] += 1;
            state.vp[p as usize] += 1;
            state.setup_pending = Some(v);
            events.push(GameEvent::BuildSettlement { player: p, vertex: v });
            // Longest-road can change if a new settlement breaks an opponent's chain.
            update_longest_road(state);
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
            let p = state.current_player;
            state.roads.set(e as usize, Some(p));
            state.roads_built[p as usize] += 1;
            state.setup_pending = None;
            events.push(GameEvent::BuildRoad { player: p, edge: e });
            update_longest_road(state);
            // Reverse-order advance; when player 0 finishes, transition to Roll.
            if state.current_player > 0 {
                state.current_player -= 1;
            } else {
                state.phase = GamePhase::Roll;
                // current_player stays at 0 — main game starts with player 0.
            }
        }
        (GamePhase::Roll, Action::RollDice) => {
            // Dedicated RollDice action — splits dice-roll out of EndTurn so
            // chance handling (Phase 1+) can target this transition explicitly.
            // Both this rng-driven arm and the env-driven apply_chance_outcome path
            // delegate to apply_dice_roll so post-roll behavior is single-sourced.
            use rand::Rng as _;
            let d1 = rng.inner().gen_range(1u8..=6);
            let d2 = rng.inner().gen_range(1u8..=6);
            let roll = d1 + d2;
            let mut sub = apply_dice_roll(state, roll, rng);
            events.append(&mut sub);
        }
        (GamePhase::Main, Action::BuildSettlement(v)) => {
            let p = state.current_player;
            let mut bank = state.bank;
            let mut hand = state.hands[p as usize];
            pay(&mut hand, &mut bank, &SETTLEMENT_COST);
            state.hands[p as usize] = hand;
            state.bank = bank;
            state.settlements.set(v as usize, Some(p));
            state.settlements_built[p as usize] += 1;
            state.vp[p as usize] += 1;
            events.push(GameEvent::BuildSettlement { player: p, vertex: v });
            // A settlement on an opponent's road can break their longest-road chain.
            update_longest_road(state);
            check_win(state, &mut events);
        }
        (GamePhase::Main, Action::BuildCity(v)) => {
            let p = state.current_player;
            let mut bank = state.bank;
            let mut hand = state.hands[p as usize];
            pay(&mut hand, &mut bank, &CITY_COST);
            state.hands[p as usize] = hand;
            state.bank = bank;
            state.settlements.set(v as usize, None);
            state.cities.set(v as usize, Some(p));
            // Note: settlements_built is the LIFETIME settlements placed.
            // Upgrading to a city doesn't decrement settlements_built; the
            // settlement no longer exists on the board but it counted toward
            // the lifetime cap. (Real Catan: 5 settlement pieces total — once
            // upgraded, the piece comes back to your supply, so the cap is
            // really "settlements simultaneously on board". v1 used "lifetime
            // built". Using "currently on board" matches Catan more accurately.)
            //
            // We use "currently on board" semantics: when a settlement
            // upgrades to a city, it's no longer counted toward the
            // settlements_built cap.
            state.settlements_built[p as usize] = state.settlements_built[p as usize].saturating_sub(1);
            state.cities_built[p as usize] += 1;
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
            state.roads.set(e as usize, Some(p));
            state.roads_built[p as usize] += 1;
            events.push(GameEvent::BuildRoad { player: p, edge: e });
            update_longest_road(state);
            check_win(state, &mut events);
        }
        (GamePhase::Main, Action::TradeBank { give, get }) => {
            let p = state.current_player as usize;
            let give_idx = give as usize;
            let get_idx = get as usize;
            // 4:1 bank trade. Ports (3:1, 2:1) deferred; will be added when we
            // track per-player port ownership.
            const TRADE_RATIO: u8 = 4;
            assert!(state.hands[p][give_idx] >= TRADE_RATIO,
                "TradeBank legality bug: {} doesn't have {} {:?}", p, TRADE_RATIO, give);
            assert!(state.bank[get_idx] >= 1,
                "TradeBank legality bug: bank doesn't have {:?}", get);
            state.hands[p][give_idx] -= TRADE_RATIO;
            state.bank[give_idx] += TRADE_RATIO;
            state.hands[p][get_idx] += 1;
            state.bank[get_idx] -= 1;
            // No event type for trades yet — could add GameEvent::TradeBank later
            // for stats tracking. For now silent.
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
        (GamePhase::MoveRobber, Action::MoveRobber(h)) => {
            let from = state.robber_hex;
            state.robber_hex = h;
            events.push(GameEvent::RobberMoved {
                player: state.current_player, from_hex: from, to_hex: h,
            });
            // Identify steal targets: opponents with buildings on this hex AND non-empty hand.
            let board = state.board.clone();
            let me = state.current_player;
            let mut targets: Vec<u8> = Vec::new();
            for &v in &board.hex_to_vertices[h as usize] {
                let owner = state.settlements.get(v as usize).or(state.cities.get(v as usize));
                if let Some(p) = owner {
                    if p != me && state.hands[p as usize].iter().sum::<u8>() > 0
                        && !targets.contains(&p)
                    {
                        targets.push(p);
                    }
                }
            }
            if targets.is_empty() {
                state.phase = GamePhase::Main;
            } else {
                // Hand control to chance: which victim is picked + which card is taken.
                state.phase = GamePhase::Steal { from_options: targets };
            }
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
            let owner_qty = match (state.settlements.get(v as usize), state.cities.get(v as usize)) {
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

fn idx_to_resource(i: u8) -> Resource {
    match i {
        0 => Resource::Wood,
        1 => Resource::Brick,
        2 => Resource::Sheep,
        3 => Resource::Wheat,
        4 => Resource::Ore,
        _ => panic!("invalid resource index {i}"),
    }
}

fn pay(hand: &mut [u8; 5], bank: &mut [u8; 5], cost: &[u8; 5]) {
    for i in 0..5 {
        hand[i] -= cost[i];
        bank[i] += cost[i];
    }
}

fn legal_actions_main(state: &GameState) -> Vec<Action> {
    use crate::state::{MAX_SETTLEMENTS, MAX_CITIES, MAX_ROADS};
    let mut out = Vec::new();
    let p = state.current_player;
    let hand = &state.hands[p as usize];
    let pi = p as usize;
    // Inventory caps (§A2): BuildSettlement only legal if currently <5 on board.
    if can_afford(hand, &SETTLEMENT_COST) && state.settlements_built[pi] < MAX_SETTLEMENTS {
        for v in 0u8..54 {
            if is_legal_settlement_for_player(state, v, p) {
                out.push(Action::BuildSettlement(v));
            }
        }
    }
    if can_afford(hand, &CITY_COST) && state.cities_built[pi] < MAX_CITIES {
        for v in 0u8..54 {
            if state.settlements.get(v as usize) == Some(p) {
                out.push(Action::BuildCity(v));
            }
        }
    }
    // Maritime trade (4:1 bank, §A5): give 4 of one resource, get 1 of another
    // (different) resource. Ports (3:1 generic, 2:1 specific) deferred.
    for give_idx in 0..5u8 {
        if hand[give_idx as usize] >= 4 {
            let give = idx_to_resource(give_idx);
            for get_idx in 0..5u8 {
                if get_idx == give_idx { continue; }
                if state.bank[get_idx as usize] == 0 { continue; } // bank must have it
                let get = idx_to_resource(get_idx);
                out.push(Action::TradeBank { give, get });
            }
        }
    }
    if can_afford(hand, &ROAD_COST) && state.roads_built[pi] < MAX_ROADS {
        for e in 0u8..72 {
            if is_legal_road_for_player(state, e, p) {
                out.push(Action::BuildRoad(e));
            }
        }
    }
    out.push(Action::EndTurn);
    out
}

fn is_legal_settlement_for_player(state: &GameState, v: u8, p: u8) -> bool {
    if !is_legal_settlement_location(state, v) { return false; }
    // Main-phase rule: must be adjacent to one of the player's roads.
    state.board.vertex_to_edges[v as usize].iter()
        .any(|&e| state.roads.get(e as usize) == Some(p))
}

fn is_legal_road_for_player(state: &GameState, e: u8, p: u8) -> bool {
    if state.roads.get(e as usize).is_some() { return false; }
    // Connects to one of the player's roads or settlements/cities.
    let [a, b] = state.board.edge_to_vertices[e as usize];
    for v in [a, b] {
        if state.settlements.get(v as usize) == Some(p) || state.cities.get(v as usize) == Some(p) {
            return true;
        }
        // Or one of the OTHER edges adjacent to this vertex is the player's road,
        // unless that vertex is occupied by another player (blocking).
        let blocked = matches!(state.settlements.get(v as usize), Some(o) if o != p)
                   || matches!(state.cities.get(v as usize), Some(o) if o != p);
        if !blocked {
            for &e2 in &state.board.vertex_to_edges[v as usize] {
                if e2 != e && state.roads.get(e2 as usize) == Some(p) {
                    return true;
                }
            }
        }
    }
    false
}

/// Compute the longest connected road chain owned by `player`, treating
/// vertices with opponent settlements/cities as chain-breakers.
///
/// Algorithm: for each owned edge, DFS from each of its endpoints, tracking
/// visited edges (not vertices — a single vertex can be revisited if multiple
/// of the player's edges meet there, but each edge is used at most once in a
/// simple path).
///
/// Catan's longest road = longest simple path through the player's roads,
/// not including edges blocked by opponent buildings at the connecting vertex.
fn longest_road_for_player(state: &GameState, player: u8) -> u8 {
    let owned_edges: Vec<u8> = (0..72u8)
        .filter(|&e| state.roads.get(e as usize) == Some(player))
        .collect();
    if owned_edges.is_empty() {
        return 0;
    }
    // For each owned edge, try starting a DFS from each of its two vertices.
    let mut best = 0u8;
    let mut visited_edges = [false; 72];
    for &start_edge in &owned_edges {
        let [v0, v1] = state.board.edge_to_vertices[start_edge as usize];
        for &start_v in &[v0, v1] {
            visited_edges.fill(false);
            visited_edges[start_edge as usize] = true;
            let other_v = if start_v == v0 { v1 } else { v0 };
            // DFS from other_v, knowing start_edge is used and contributes 1 to length.
            let len = dfs_road_length(state, player, other_v, &mut visited_edges, 1);
            if len > best { best = len; }
        }
    }
    best
}

/// DFS helper: from vertex `at`, walk along owned edges (avoiding visited ones,
/// avoiding vertices blocked by opponent buildings) and return max chain length.
fn dfs_road_length(
    state: &GameState,
    player: u8,
    at: u8,
    visited_edges: &mut [bool; 72],
    so_far: u8,
) -> u8 {
    // If `at` has an opponent settlement/city, the chain is blocked here.
    let owner = state.settlements.get(at as usize).or(state.cities.get(at as usize));
    if let Some(o) = owner {
        if o != player {
            return so_far;
        }
    }
    let mut best = so_far;
    for &e in &state.board.vertex_to_edges[at as usize] {
        if visited_edges[e as usize] { continue; }
        if state.roads.get(e as usize) != Some(player) { continue; }
        let [v0, v1] = state.board.edge_to_vertices[e as usize];
        let next = if v0 == at { v1 } else { v0 };
        visited_edges[e as usize] = true;
        let len = dfs_road_length(state, player, next, visited_edges, so_far + 1);
        if len > best { best = len; }
        visited_edges[e as usize] = false;
    }
    best
}

/// Recompute longest_road_length for all players and update the longest_road_holder.
/// Called after any BuildRoad or BuildSettlement (settlements can break opponent chains).
/// Holder is the strict max owner with length >= 5; ties = no holder change unless
/// the previous holder no longer has the max.
pub fn update_longest_road(state: &mut GameState) {
    for p in 0..4u8 {
        state.longest_road_length[p as usize] = longest_road_for_player(state, p);
    }
    // Find the player with the strict max length >= 5.
    let max_len = state.longest_road_length.iter().copied().max().unwrap_or(0);
    if max_len < 5 {
        // No one qualifies; remove holder if any.
        if let Some(prev) = state.longest_road_holder.take() {
            // Subtract the +2 VP we previously granted.
            state.vp[prev as usize] = state.vp[prev as usize].saturating_sub(2);
        }
        return;
    }
    let candidates: Vec<u8> = (0..4u8)
        .filter(|&p| state.longest_road_length[p as usize] == max_len)
        .collect();

    // Determine new holder per Catan rules:
    // - If exactly one player has the strict max → they become holder.
    // - If multiple are tied at max → previous holder keeps if they're tied; otherwise no holder.
    let new_holder = if candidates.len() == 1 {
        Some(candidates[0])
    } else {
        match state.longest_road_holder {
            Some(prev) if candidates.contains(&prev) => Some(prev),
            _ => None,
        }
    };

    if new_holder != state.longest_road_holder {
        // Transfer +2 VP from old holder (if any) to new holder (if any).
        if let Some(prev) = state.longest_road_holder {
            state.vp[prev as usize] = state.vp[prev as usize].saturating_sub(2);
        }
        if let Some(new) = new_holder {
            state.vp[new as usize] += 2;
        }
        state.longest_road_holder = new_holder;
    }
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

/// Apply a specific dice roll value (used by both the rng-driven Action::RollDice path
/// and the env-driven apply_chance_outcome path, so behavior is single-sourced).
///
/// **v2 simplification (§A-bis 8a):** when a 7 is rolled, each player owing a
/// discard auto-discards floor(hand/2) random cards in this same step. No
/// multi-step Discard phase. Uses the engine's RNG so the discards are
/// deterministic given the dice seed.
pub fn apply_dice_roll(
    state: &mut GameState,
    roll: u8,
    rng: &mut Rng,
) -> Vec<GameEvent> {
    let mut events = Vec::new();
    events.push(GameEvent::DiceRolled { roll });
    if roll == 7 {
        // Collect (player, n_to_discard) for everyone owing.
        let mut owe = [0u8; 4];
        for p in 0..4usize {
            let total: u8 = state.hands[p].iter().sum();
            if total > 7 {
                owe[p] = total / 2;
            }
        }
        // Apply random discards inline.
        for p in 0..4usize {
            for _ in 0..owe[p] {
                let total: u8 = state.hands[p].iter().sum();
                if total == 0 { break; }
                // Pick a random flat card index (0..total) and find which resource it lands on.
                use rand::Rng as _;
                let pick = rng.inner().gen_range(0u8..total);
                let mut acc = 0u8;
                for ri in 0..N_RESOURCES {
                    let count = state.hands[p][ri];
                    if pick < acc + count {
                        state.hands[p][ri] -= 1;
                        state.bank[ri] += 1;
                        let res = match ri {
                            0 => crate::board::Resource::Wood,
                            1 => crate::board::Resource::Brick,
                            2 => crate::board::Resource::Sheep,
                            3 => crate::board::Resource::Wheat,
                            4 => crate::board::Resource::Ore,
                            _ => unreachable!(),
                        };
                        events.push(GameEvent::Discarded { player: p as u8, resource: res });
                        break;
                    }
                    acc += count;
                }
            }
        }
        // Skip the multi-step Discard phase entirely.
        state.phase = GamePhase::MoveRobber;
    } else {
        produce_resources(state, roll, &mut events);
        state.phase = GamePhase::Main;
    }
    events
}

/// Apply a specific steal: take `card_index`-th card from `victim`'s hand and give to current player.
pub(crate) fn apply_steal(state: &mut GameState, victim: u8, card_index: u8) -> Vec<GameEvent> {
    debug_assert!(
        state.hands[victim as usize].iter().sum::<u8>() > 0,
        "Steal phase entered with empty victim hand — invariant violated upstream",
    );
    let mut events = Vec::new();
    let me = state.current_player;
    let mut acc = 0u8;
    for ri in 0..5usize {
        let count = state.hands[victim as usize][ri];
        if card_index < acc + count {
            state.hands[victim as usize][ri] -= 1;
            state.hands[me as usize][ri] += 1;
            let res = match ri {
                0 => Resource::Wood,
                1 => Resource::Brick,
                2 => Resource::Sheep,
                3 => Resource::Wheat,
                4 => Resource::Ore,
                _ => unreachable!(),
            };
            events.push(GameEvent::Robbed { from: victim, to: me, resource: Some(res) });
            state.phase = GamePhase::Main;
            return events;
        }
        acc += count;
    }
    panic!("card_index {card_index} out of range; victim hand sums to {acc}");
}
