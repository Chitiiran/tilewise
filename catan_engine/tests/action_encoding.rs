use catan_engine::actions::{Action, ACTION_SPACE_SIZE, decode, encode};
use catan_engine::board::Resource;

#[test]
fn action_space_size_is_226_for_v2() {
    // v2 added 20 TradeBank actions (5 give × 4 valid get) at 206-225.
    assert_eq!(ACTION_SPACE_SIZE, 226);
}

#[test]
fn encode_decode_roundtrip_for_all_ids() {
    for id in 0..ACTION_SPACE_SIZE {
        let action = decode(id as u32).expect("every ID must decode");
        let re_encoded = encode(action);
        assert_eq!(re_encoded, id as u32, "round-trip failed for ID {id}");
    }
}

#[test]
fn id_layout_matches_spec() {
    // Spec §4: BuildSettlement(0..53)=0..53,
    //          BuildCity(0..53)=54..107,
    //          BuildRoad(0..71)=108..179,
    //          MoveRobber(0..18)=180..198,
    //          Discard(Wood..Ore)=199..203,
    //          EndTurn=204,
    //          RollDice=205
    assert_eq!(encode(Action::BuildSettlement(0)), 0);
    assert_eq!(encode(Action::BuildSettlement(53)), 53);
    assert_eq!(encode(Action::BuildCity(0)), 54);
    assert_eq!(encode(Action::BuildCity(53)), 107);
    assert_eq!(encode(Action::BuildRoad(0)), 108);
    assert_eq!(encode(Action::BuildRoad(71)), 179);
    assert_eq!(encode(Action::MoveRobber(0)), 180);
    assert_eq!(encode(Action::MoveRobber(18)), 198);
    assert_eq!(encode(Action::Discard(Resource::Wood)), 199);
    assert_eq!(encode(Action::Discard(Resource::Ore)), 203);
    assert_eq!(encode(Action::EndTurn), 204);
    assert_eq!(encode(Action::RollDice), 205);
}

#[test]
fn out_of_range_ids_return_none() {
    // v2: TradeBank actions occupy 206..226, so the first invalid ID is 226.
    assert!(decode(226).is_none());
    assert!(decode(u32::MAX).is_none());
}

#[test]
fn roll_dice_action_round_trips() {
    use catan_engine::actions::{Action, encode, decode};
    let id = encode(Action::RollDice);
    assert_eq!(id, 205);
    assert_eq!(decode(205), Some(Action::RollDice));
}

#[test]
fn action_space_size_is_226_after_v2() {
    assert_eq!(catan_engine::actions::ACTION_SPACE_SIZE, 226);
}

#[test]
fn trade_bank_round_trip_for_all_pairs() {
    for give_idx in 0..5u8 {
        for get_idx in 0..5u8 {
            if give_idx == get_idx { continue; }
            let give = match give_idx { 0=>Resource::Wood, 1=>Resource::Brick, 2=>Resource::Sheep, 3=>Resource::Wheat, 4=>Resource::Ore, _=>unreachable!() };
            let get = match get_idx { 0=>Resource::Wood, 1=>Resource::Brick, 2=>Resource::Sheep, 3=>Resource::Wheat, 4=>Resource::Ore, _=>unreachable!() };
            let id = encode(Action::TradeBank { give, get });
            assert!((206..226).contains(&id), "TradeBank ID out of range: {}", id);
            let decoded = decode(id).expect("must decode");
            assert_eq!(decoded, Action::TradeBank { give, get });
        }
    }
}

#[test]
fn trade_bank_action_count_is_20() {
    let mut count = 0;
    for id in 206..226 {
        if decode(id).is_some() { count += 1; }
    }
    assert_eq!(count, 20, "must have exactly 20 valid TradeBank actions (5 give × 4 valid get)");
}
