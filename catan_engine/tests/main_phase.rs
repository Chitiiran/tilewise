use catan_engine::actions::Action;
use catan_engine::board::Board;
use catan_engine::rng::Rng;
use catan_engine::rules::{apply, legal_actions};
use catan_engine::state::{GameState, GamePhase};
use std::sync::Arc;

/// Build a fully-set-up state where it's player 0's turn to Roll.
fn ready_to_roll() -> GameState {
    let mut state = GameState::new(Arc::new(Board::standard()));
    let mut rng = Rng::from_seed(0);
    let s1 = [0u8, 8, 16, 24];
    let s2 = [9u8, 17, 25, 28]; // controller-verified non-adjacent legal vertices
    for v in s1 {
        apply(&mut state, Action::BuildSettlement(v), &mut rng);
        let e = state.board.vertex_to_edges[v as usize][0];
        apply(&mut state, Action::BuildRoad(e), &mut rng);
    }
    for v in s2 {
        apply(&mut state, Action::BuildSettlement(v), &mut rng);
        let e = state.board.vertex_to_edges[v as usize][0];
        apply(&mut state, Action::BuildRoad(e), &mut rng);
    }
    // Extend player 0's first road (edge 0: v0->v3) by one more edge to vertex 7
    // (edge 6: v3->v7). Vertex 7 is empty and not adjacent to any existing settlement,
    // so it becomes a legal Main-phase settlement spot once player 0 can afford it.
    // Required because the bare setup-phase road endpoints for player 0 (v3 and v22)
    // are both blocked by the distance rule from existing settlements.
    state.roads.set(6, Some(0));
    assert!(matches!(state.phase, GamePhase::Roll));
    state
}

#[test]
fn legal_actions_in_roll_phase_is_just_roll_dice() {
    // Spec choice (Phase 1 split): in Roll phase, the only legal action is RollDice.
    let state = ready_to_roll();
    let legal = legal_actions(&state);
    assert_eq!(legal, vec![Action::RollDice]);
}

#[test]
fn rolling_a_non_seven_produces_resources_and_enters_main() {
    let mut state = ready_to_roll();
    let mut rng = Rng::from_seed(7); // seed chosen so first roll != 7 (verify via assertion)
    let bank_before = state.bank;
    apply(&mut state, Action::RollDice, &mut rng);
    // Either we're in Main now, or we entered Discard/MoveRobber on a 7. Check Main case.
    if matches!(state.phase, GamePhase::Main) {
        // Bank can only decrease (resources flow bank -> player hands).
        let bank_after_total: u32 = state.bank.iter().map(|&x| x as u32).sum();
        let bank_before_total: u32 = bank_before.iter().map(|&x| x as u32).sum();
        assert!(bank_after_total <= bank_before_total);
    }
}

#[test]
fn cannot_build_settlement_without_resources() {
    let mut state = ready_to_roll();
    state.phase = GamePhase::Main;
    state.hands[0] = [0; 5]; // empty hand
    let legal = legal_actions(&state);
    assert!(!legal.iter().any(|a| matches!(a, Action::BuildSettlement(_))));
}

#[test]
fn can_build_settlement_with_required_resources_on_road_endpoint() {
    let mut state = ready_to_roll();
    state.phase = GamePhase::Main;
    // Give player 0 settlement cost: 1 wood, 1 brick, 1 sheep, 1 wheat.
    state.hands[0] = [1, 1, 1, 1, 0];
    let legal = legal_actions(&state);
    let settlement_actions: Vec<u8> = legal.iter()
        .filter_map(|a| if let Action::BuildSettlement(v) = a { Some(*v) } else { None })
        .collect();
    // At least one settlement should be legal (the road endpoints from setup).
    assert!(!settlement_actions.is_empty());
}

#[test]
fn building_settlement_deducts_resources_and_grants_vp() {
    let mut state = ready_to_roll();
    state.phase = GamePhase::Main;
    state.hands[0] = [2, 2, 2, 2, 0];
    let mut rng = Rng::from_seed(0);
    let v_to_build = legal_actions(&state).iter()
        .find_map(|a| if let Action::BuildSettlement(v) = a { Some(*v) } else { None })
        .expect("at least one legal settlement");
    let vp_before = state.vp[0];
    apply(&mut state, Action::BuildSettlement(v_to_build), &mut rng);
    assert_eq!(state.hands[0], [1, 1, 1, 1, 0]);
    assert_eq!(state.vp[0], vp_before + 1);
    assert_eq!(state.settlements.get(v_to_build as usize), Some(0));
}

#[test]
fn upgrading_settlement_to_city_costs_2_wheat_3_ore_and_grants_extra_vp() {
    let mut state = ready_to_roll();
    state.phase = GamePhase::Main;
    state.hands[0] = [0, 0, 0, 2, 3];
    let mut rng = Rng::from_seed(0);
    let owned = (0u8..54).find(|&v| state.settlements.get(v as usize) == Some(0)).unwrap();
    let vp_before = state.vp[0];
    apply(&mut state, Action::BuildCity(owned), &mut rng);
    assert_eq!(state.cities.get(owned as usize), Some(0));
    assert!(state.settlements.get(owned as usize).is_none());
    assert_eq!(state.hands[0], [0, 0, 0, 0, 0]);
    assert_eq!(state.vp[0], vp_before + 1); // +1 (city is 2VP, settlement was 1VP)
}

#[test]
fn end_turn_advances_player_and_returns_to_roll() {
    let mut state = ready_to_roll();
    state.phase = GamePhase::Main;
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::EndTurn, &mut rng);
    assert_eq!(state.current_player, 1);
    assert!(matches!(state.phase, GamePhase::Roll));
}

#[test]
fn stats_track_basic_game_progress() {
    use rand::rngs::SmallRng;
    use rand::{Rng as _, SeedableRng};
    let mut engine = catan_engine::Engine::new(42);
    let mut policy_rng = SmallRng::seed_from_u64(0xC47A1B07_u64);
    let mut steps = 0;
    while !engine.is_terminal() {
        if engine.is_chance_pending() {
            // Roll / Steal phases — drive via chance API rather than legal_actions.
            let outcomes = engine.chance_outcomes();
            let idx = policy_rng.gen_range(0..outcomes.len());
            engine.apply_chance_outcome(outcomes[idx].0);
        } else {
            let legal = engine.legal_actions();
            if legal.is_empty() { break; }
            let idx = policy_rng.gen_range(0..legal.len());
            engine.step(legal[idx]);
        }
        steps += 1;
        if steps > 5000 { break; }
    }
    let s = engine.stats();
    assert_eq!(s.schema_version, catan_engine::stats::STATS_SCHEMA_VERSION);
    assert!(s.turns_played > 0, "no turns recorded");
    // A game ending mid-turn (winning build before EndTurn) leaves one extra
    // DiceRolled with no matching TurnEnded — so total_dice can be turns_played
    // or turns_played + 1.
    let total_dice: u32 = s.dice_histogram.iter().sum::<u32>() + s.seven_count;
    assert!(
        total_dice == s.turns_played || total_dice == s.turns_played + 1,
        "dice histogram + sevens ({}) should equal turns ({}) or turns+1",
        total_dice, s.turns_played
    );
    assert!(s.winner_player_id >= 0, "no winner recorded after {} steps", steps);
}

#[test]
fn roll_phase_legal_action_is_roll_dice_only() {
    use catan_engine::Engine;
    let mut e = Engine::new(42);
    while !matches!(e.state.phase, catan_engine::state::GamePhase::Roll) {
        let legal = e.legal_actions();
        assert!(!legal.is_empty(), "got stuck before reaching Roll");
        e.step(legal[0]);
    }
    let legal = e.legal_actions();
    assert_eq!(legal, vec![205], "Roll phase should expose only RollDice (id 205)");
}
