use catan_engine::actions::{decode, encode, ACTION_SPACE_SIZE};
use catan_engine::Engine;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn action_id_round_trip(id in 0u32..(ACTION_SPACE_SIZE as u32)) {
        let action = decode(id).unwrap();
        prop_assert_eq!(encode(action), id);
    }

    #[test]
    fn random_legal_play_never_panics(seed in any::<u64>(), choices in prop::collection::vec(any::<u32>(), 0..1000)) {
        let mut engine = Engine::new(seed);
        for c in choices {
            if engine.is_terminal() { break; }
            let legal = engine.legal_actions();
            if legal.is_empty() { break; }
            let a = legal[c as usize % legal.len()];
            engine.step(a);
        }
        // No panic = test passes.
    }

    #[test]
    fn invariant_total_resources_within_supply(seed in any::<u64>(), choices in prop::collection::vec(any::<u32>(), 0..500)) {
        let mut engine = Engine::new(seed);
        for c in choices {
            if engine.is_terminal() { break; }
            let legal = engine.legal_actions();
            if legal.is_empty() { break; }
            engine.step(legal[c as usize % legal.len()]);
            // For each resource: bank + sum(player hands) == 19.
            for r in 0..5 {
                let in_hands: u32 = (0..4).map(|p| engine.state.hands[p][r] as u32).sum();
                let bank = engine.state.bank[r] as u32;
                prop_assert_eq!(in_hands + bank, 19u32, "resource {} bookkeeping broken", r);
            }
        }
    }

    #[test]
    fn invariant_vp_non_decreasing(seed in any::<u64>(), choices in prop::collection::vec(any::<u32>(), 0..500)) {
        let mut engine = Engine::new(seed);
        let mut last_vp = [0u8; 4];
        for c in choices {
            if engine.is_terminal() { break; }
            let legal = engine.legal_actions();
            if legal.is_empty() { break; }
            engine.step(legal[c as usize % legal.len()]);
            for p in 0..4 {
                prop_assert!(engine.state.vp[p] >= last_vp[p]);
                last_vp[p] = engine.state.vp[p];
            }
        }
    }
}
