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
            update_ports_owned_for_player(state, p);
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
            update_ports_owned_for_player(state, p);
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
            update_ports_owned_for_player(state, p);
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
            update_ports_owned_for_player(state, p);
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
            let p = state.current_player;
            let pi = p as usize;
            let give_idx = give as usize;
            let get_idx = get as usize;
            let ratio = trade_ratio_for(state, p, give_idx as u8);
            assert!(state.hands[pi][give_idx] >= ratio,
                "TradeBank legality bug: P{} doesn't have {} {:?}", pi, ratio, give);
            assert!(state.bank[get_idx] >= 1,
                "TradeBank legality bug: bank doesn't have {:?}", get);
            state.hands[pi][give_idx] -= ratio;
            state.bank[give_idx] += ratio;
            state.hands[pi][get_idx] += 1;
            state.bank[get_idx] -= 1;
            // No event type for trades yet — could add GameEvent::TradeBank later.
        }
        (GamePhase::Main, Action::BuyDevCard) => {
            use crate::state::{DEV_CARD_VP, N_DEV_CARDS};
            let p = state.current_player;
            let pi = p as usize;
            let mut bank = state.bank;
            let mut hand = state.hands[pi];
            pay(&mut hand, &mut bank, &DEV_CARD_COST);
            state.hands[pi] = hand;
            state.bank = bank;
            // Random draw from the deck weighted by remaining counts.
            use rand::Rng as _;
            let total_remaining: u32 = state.dev_card_deck_remaining.iter().map(|&x| x as u32).sum();
            assert!(total_remaining > 0, "BuyDevCard with empty deck");
            let pick = rng.inner().gen_range(0u32..total_remaining);
            let mut acc = 0u32;
            for ct in 0..N_DEV_CARDS {
                let count = state.dev_card_deck_remaining[ct] as u32;
                if pick < acc + count {
                    state.dev_card_deck_remaining[ct] -= 1;
                    if ct == DEV_CARD_VP {
                        // VP card: apply immediately (no hidden info in our engine).
                        state.dev_cards_played[pi][DEV_CARD_VP] += 1;
                        state.vp[pi] += 1;
                        check_win(state, &mut events);
                    } else {
                        state.dev_cards_held[pi][ct] += 1;
                        state.dev_cards_just_bought[pi] = true;
                    }
                    break;
                }
                acc += count;
            }
        }
        (GamePhase::Main, Action::PlayKnight) => {
            use crate::state::DEV_CARD_KNIGHT;
            let p = state.current_player;
            let pi = p as usize;
            assert!(state.dev_cards_held[pi][DEV_CARD_KNIGHT] > 0, "PlayKnight with no knight");
            state.dev_cards_held[pi][DEV_CARD_KNIGHT] -= 1;
            state.dev_cards_played[pi][DEV_CARD_KNIGHT] += 1;
            state.knights_played[pi] += 1;
            state.dev_card_played_this_turn[pi] = true;
            update_largest_army(state);
            // Transition to MoveRobber, like a 7 was rolled (without the discard).
            state.phase = GamePhase::MoveRobber;
        }
        (GamePhase::Main, Action::PlayRoadBuilding) => {
            use crate::state::DEV_CARD_ROAD_BUILDING;
            let p = state.current_player;
            let pi = p as usize;
            assert!(state.dev_cards_held[pi][DEV_CARD_ROAD_BUILDING] > 0);
            state.dev_cards_held[pi][DEV_CARD_ROAD_BUILDING] -= 1;
            state.dev_cards_played[pi][DEV_CARD_ROAD_BUILDING] += 1;
            state.dev_card_played_this_turn[pi] = true;
            // Simplification: grant 2 wood + 2 brick from bank (player builds with these).
            // Strict rule would let them place 2 free roads; this is rule-equivalent.
            let take_wood = 2u8.min(state.bank[0]);
            let take_brick = 2u8.min(state.bank[1]);
            state.hands[pi][0] += take_wood;
            state.hands[pi][1] += take_brick;
            state.bank[0] -= take_wood;
            state.bank[1] -= take_brick;
        }
        (GamePhase::Main, Action::PlayMonopoly(r)) => {
            use crate::state::DEV_CARD_MONOPOLY;
            let p = state.current_player;
            let pi = p as usize;
            assert!(state.dev_cards_held[pi][DEV_CARD_MONOPOLY] > 0);
            state.dev_cards_held[pi][DEV_CARD_MONOPOLY] -= 1;
            state.dev_cards_played[pi][DEV_CARD_MONOPOLY] += 1;
            state.dev_card_played_this_turn[pi] = true;
            let ri = r as usize;
            // Take all of `r` from every other player.
            let mut taken = 0u8;
            for opp in 0..4usize {
                if opp != pi {
                    taken += state.hands[opp][ri];
                    state.hands[opp][ri] = 0;
                }
            }
            state.hands[pi][ri] += taken;
        }
        (GamePhase::Main, Action::PlayYearOfPlenty(r1, r2)) => {
            use crate::state::DEV_CARD_YOP;
            let p = state.current_player;
            let pi = p as usize;
            assert!(state.dev_cards_held[pi][DEV_CARD_YOP] > 0);
            state.dev_cards_held[pi][DEV_CARD_YOP] -= 1;
            state.dev_cards_played[pi][DEV_CARD_YOP] += 1;
            state.dev_card_played_this_turn[pi] = true;
            // Take 1 each from bank (skip if bank empty).
            for r in [r1, r2] {
                let ri = r as usize;
                if state.bank[ri] > 0 {
                    state.bank[ri] -= 1;
                    state.hands[pi][ri] += 1;
                }
            }
        }
        (GamePhase::Main, Action::PlayVpCard) => {
            // VP cards are applied immediately at buy. PlayVpCard is a no-op
            // for compatibility with action-list iteration.
        }
        (GamePhase::Main, Action::ProposeTrade { give, get }) => {
            // 1-for-1 player trade. First opponent in seat order with ≥1 of `get`
            // accepts. If none do, no resources move.
            let p = state.current_player;
            let pi = p as usize;
            let give_idx = give as usize;
            let get_idx = get as usize;
            assert!(state.hands[pi][give_idx] >= 1);
            // Find first opponent (in seat order) that can accept.
            let mut accepted = false;
            for offset in 1..4u8 {
                let opp = ((p + offset) % 4) as usize;
                if state.hands[opp][get_idx] >= 1 {
                    state.hands[pi][give_idx] -= 1;
                    state.hands[opp][give_idx] += 1;
                    state.hands[opp][get_idx] -= 1;
                    state.hands[pi][get_idx] += 1;
                    accepted = true;
                    break;
                }
            }
            // Silent if no one accepts. Could emit GameEvent::TradeRejected later.
            let _ = accepted;
            // Per-turn cap: prevents the trade-loop pathology where MCTS
            // alternates A→B then B→A forever (total resources conserved,
            // no terminal reward, no progress). Reset on EndTurn.
            state.trades_this_turn = state.trades_this_turn.saturating_add(1);
        }
        (GamePhase::Main, Action::EndTurn) => {
            // Reset per-turn flags.
            for p in 0..4 {
                state.dev_cards_just_bought[p] = false;
                state.dev_card_played_this_turn[p] = false;
            }
            state.trades_this_turn = 0;
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
    // End-of-apply terminal check: any action that mutates VP (build, dev card,
    // OR a longest-road / largest-army bonus flip via update_*) must trigger
    // termination if VP >= WIN_VP. The per-action check_win calls are now
    // redundant but kept for early termination in long branches.
    // Found via seed=1100021: P0 reached 11 VP at step 642 because the LA flip
    // happened in a knight play whose handler didn't call check_win.
    if !matches!(state.phase, GamePhase::Done { .. }) {
        check_win(state, &mut events);
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
const DEV_CARD_COST: [u8; 5] =   [0, 0, 1, 1, 1]; // 1 sheep + 1 wheat + 1 ore

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
    // Maritime trade (§A5): give N of one resource, get 1 of another
    // (different) resource. Ratio: 2 if specific port owned, 3 if any generic
    // port owned, else 4.
    for give_idx in 0..5u8 {
        let ratio = trade_ratio_for(state, p, give_idx);
        if hand[give_idx as usize] >= ratio {
            let give = idx_to_resource(give_idx);
            for get_idx in 0..5u8 {
                if get_idx == give_idx { continue; }
                if state.bank[get_idx as usize] == 0 { continue; } // bank must have it
                let get = idx_to_resource(get_idx);
                out.push(Action::TradeBank { give, get });
            }
        }
    }
    // Buy dev card: need 1 sheep + 1 wheat + 1 ore AND deck non-empty.
    if can_afford(hand, &DEV_CARD_COST)
        && state.dev_card_deck_remaining.iter().any(|&n| n > 0)
    {
        out.push(Action::BuyDevCard);
    }

    // Play dev card: only one per turn (except VP, which we apply immediately on buy).
    // Can't play a card the turn you bought it.
    if !state.dev_card_played_this_turn[pi] && !state.dev_cards_just_bought[pi] {
        use crate::state::*;
        if state.dev_cards_held[pi][DEV_CARD_KNIGHT] > 0 {
            out.push(Action::PlayKnight);
        }
        if state.dev_cards_held[pi][DEV_CARD_ROAD_BUILDING] > 0 {
            out.push(Action::PlayRoadBuilding);
        }
        if state.dev_cards_held[pi][DEV_CARD_MONOPOLY] > 0 {
            for r in 0..5u8 {
                out.push(Action::PlayMonopoly(idx_to_resource(r)));
            }
        }
        if state.dev_cards_held[pi][DEV_CARD_YOP] > 0 {
            for r1 in 0..5u8 {
                for r2 in 0..5u8 {
                    out.push(Action::PlayYearOfPlenty(idx_to_resource(r1), idx_to_resource(r2)));
                }
            }
        }
        // VP card: applied at buy time; PlayVpCard exists for explicit reveal but is no-op.
        if state.dev_cards_held[pi][DEV_CARD_VP] > 0 {
            out.push(Action::PlayVpCard);
        }
    }

    // Player-to-player 1-for-1 trade (§A6 simplified).
    // Proposer must have ≥1 of give, AND at least one opponent must have ≥1 of get,
    // AND the per-turn trade cap must not be reached (MAX_TRADES_PER_TURN).
    if state.trades_this_turn < crate::state::MAX_TRADES_PER_TURN {
        for give_idx in 0..5u8 {
            if hand[give_idx as usize] >= 1 {
                for get_idx in 0..5u8 {
                    if get_idx == give_idx { continue; }
                    let any_opp_has = (0..4u8)
                        .filter(|&opp| opp != p)
                        .any(|opp| state.hands[opp as usize][get_idx as usize] >= 1);
                    if any_opp_has {
                        out.push(Action::ProposeTrade {
                            give: idx_to_resource(give_idx),
                            get: idx_to_resource(get_idx),
                        });
                    }
                }
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

/// Recompute ports_owned for a single player from current settlement/city placements.
/// Called whenever the player places a settlement or upgrades to a city
/// (since a city replaces a settlement at the same vertex, port ownership
/// is preserved but we recompute for safety).
pub fn update_ports_owned_for_player(state: &mut GameState, player: u8) {
    use crate::board::PortKind;
    use crate::state::{
        PORT_BIT_GENERIC, PORT_BIT_WOOD, PORT_BIT_BRICK, PORT_BIT_SHEEP, PORT_BIT_WHEAT, PORT_BIT_ORE,
    };
    let mut bits = 0u8;
    for port in &state.board.ports {
        let owned = port.vertices.iter().any(|&v| {
            state.settlements.get(v as usize) == Some(player)
                || state.cities.get(v as usize) == Some(player)
        });
        if owned {
            match port.kind {
                PortKind::Generic => bits |= PORT_BIT_GENERIC,
                PortKind::Specific(r) => match r {
                    crate::board::Resource::Wood => bits |= PORT_BIT_WOOD,
                    crate::board::Resource::Brick => bits |= PORT_BIT_BRICK,
                    crate::board::Resource::Sheep => bits |= PORT_BIT_SHEEP,
                    crate::board::Resource::Wheat => bits |= PORT_BIT_WHEAT,
                    crate::board::Resource::Ore => bits |= PORT_BIT_ORE,
                },
            }
        }
    }
    state.ports_owned[player as usize] = bits;
}

/// Effective trade ratio for player giving `give_idx`: 2 if they have the
/// specific port, else 3 if they have any generic port, else 4.
pub fn trade_ratio_for(state: &GameState, player: u8, give_idx: u8) -> u8 {
    let bits = state.ports_owned[player as usize];
    let specific_bit = crate::state::resource_port_bit(give_idx);
    if (bits & specific_bit) != 0 { 2 }
    else if (bits & crate::state::PORT_BIT_GENERIC) != 0 { 3 }
    else { 4 }
}

/// Recompute longest_road_length for all players and update the longest_road_holder.
/// Called after any BuildRoad or BuildSettlement (settlements can break opponent chains).
/// Holder is the strict max owner with length >= 5; ties = no holder change unless
/// the previous holder no longer has the max.
///
/// When `state.bonuses_enabled = false`, length and holder still update (so
/// observation features stay populated for the GNN), but the +2 VP transfer
/// is suppressed.
pub fn update_longest_road(state: &mut GameState) {
    for p in 0..4u8 {
        state.longest_road_length[p as usize] = longest_road_for_player(state, p);
    }
    // Find the player with the strict max length >= 5.
    let max_len = state.longest_road_length.iter().copied().max().unwrap_or(0);
    if max_len < 5 {
        // No one qualifies; remove holder if any.
        if let Some(prev) = state.longest_road_holder.take() {
            if state.bonuses_enabled {
                state.vp[prev as usize] = state.vp[prev as usize].saturating_sub(2);
            }
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
        if state.bonuses_enabled {
            // Transfer +2 VP from old holder (if any) to new holder (if any).
            if let Some(prev) = state.longest_road_holder {
                state.vp[prev as usize] = state.vp[prev as usize].saturating_sub(2);
            }
            if let Some(new) = new_holder {
                state.vp[new as usize] += 2;
            }
        }
        state.longest_road_holder = new_holder;
    }
}

/// Test-only: run only the holder-transfer logic of `update_longest_road`,
/// bypassing the recompute-length-from-edges step. Lets tests force a length
/// array directly without owning matching road edges.
#[doc(hidden)]
pub fn transfer_longest_road_holder_for_test(state: &mut GameState) {
    let max_len = state.longest_road_length.iter().copied().max().unwrap_or(0);
    if max_len < 5 {
        if let Some(prev) = state.longest_road_holder.take() {
            if state.bonuses_enabled {
                state.vp[prev as usize] = state.vp[prev as usize].saturating_sub(2);
            }
        }
        return;
    }
    let candidates: Vec<u8> = (0..4u8)
        .filter(|&p| state.longest_road_length[p as usize] == max_len)
        .collect();
    let new_holder = if candidates.len() == 1 {
        Some(candidates[0])
    } else {
        match state.longest_road_holder {
            Some(prev) if candidates.contains(&prev) => Some(prev),
            _ => None,
        }
    };
    if new_holder != state.longest_road_holder {
        if state.bonuses_enabled {
            if let Some(prev) = state.longest_road_holder {
                state.vp[prev as usize] = state.vp[prev as usize].saturating_sub(2);
            }
            if let Some(new) = new_holder {
                state.vp[new as usize] += 2;
            }
        }
        state.longest_road_holder = new_holder;
    }
}

/// Recompute largest_army_holder + transferable +2 VP, just like longest road.
/// Holder is the strict-max-knights-played owner with knights >= 3.
///
/// When `state.bonuses_enabled = false`, holder still updates (so the GNN
/// observation reflects who has the most knights) but no +2 VP is awarded.
pub fn update_largest_army(state: &mut GameState) {
    let max_k = state.knights_played.iter().copied().max().unwrap_or(0);
    if max_k < 3 {
        if let Some(prev) = state.largest_army_holder.take() {
            if state.bonuses_enabled {
                state.vp[prev as usize] = state.vp[prev as usize].saturating_sub(2);
            }
        }
        return;
    }
    let candidates: Vec<u8> = (0..4u8)
        .filter(|&p| state.knights_played[p as usize] == max_k)
        .collect();
    let new_holder = if candidates.len() == 1 {
        Some(candidates[0])
    } else {
        match state.largest_army_holder {
            Some(prev) if candidates.contains(&prev) => Some(prev),
            _ => None,
        }
    };
    if new_holder != state.largest_army_holder {
        if state.bonuses_enabled {
            if let Some(prev) = state.largest_army_holder {
                state.vp[prev as usize] = state.vp[prev as usize].saturating_sub(2);
            }
            if let Some(new) = new_holder {
                state.vp[new as usize] += 2;
            }
        }
        state.largest_army_holder = new_holder;
    }
}

fn check_win(state: &mut GameState, events: &mut Vec<GameEvent>) {
    for p in 0..4u8 {
        if state.vp[p as usize] >= state.vp_target {
            state.phase = GamePhase::Done { winner: p };
            events.push(GameEvent::GameOver { winner: p });
            return;
        }
    }
}

/// Test-only entry point: invoke the win check on a state without dispatching
/// through the full action-apply path. Used by `tests/v3_rules.rs` to verify
/// `vp_target` controls the threshold.
#[doc(hidden)]
pub fn check_win_for_test(state: &mut GameState) {
    let mut evs = Vec::new();
    check_win(state, &mut evs);
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
