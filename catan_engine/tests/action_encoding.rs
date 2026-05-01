use catan_engine::actions::{Action, ACTION_SPACE_SIZE, decode, encode};
use catan_engine::board::Resource;

#[test]
fn action_space_size_is_280_for_v2_4() {
    // v2.1 added 20 TradeBank actions (206-225).
    // v2.3 added 34 dev card actions (226-259).
    // v2.4 added 20 ProposeTrade actions (260-279).
    assert_eq!(ACTION_SPACE_SIZE, 280);
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
    // v2.4: ProposeTrade actions occupy 260..280, so the first invalid ID is 280.
    assert!(decode(280).is_none());
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
fn action_space_size_is_280_after_v2_4() {
    assert_eq!(catan_engine::actions::ACTION_SPACE_SIZE, 280);
}

#[test]
fn propose_trade_action_round_trip_for_all_pairs() {
    let resources = [Resource::Wood, Resource::Brick, Resource::Sheep, Resource::Wheat, Resource::Ore];
    for give in resources {
        for get in resources {
            if give == get { continue; }
            let id = encode(Action::ProposeTrade { give, get });
            assert!((260..280).contains(&id), "ID {} out of range", id);
            let back = decode(id).unwrap();
            assert_eq!(back, Action::ProposeTrade { give, get });
        }
    }
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

#[test]
fn dev_card_actions_round_trip() {
    let mut count = 0;
    for id in 226..260 {
        let action = decode(id).expect("dev card ID should decode");
        let re = encode(action);
        assert_eq!(re, id, "round-trip failed for dev card ID {}", id);
        count += 1;
    }
    assert_eq!(count, 34, "34 dev card actions: 1 buy + 1 knight + 1 rb + 5 mono + 25 yop + 1 vp");
}

#[test]
fn dev_card_specific_ids_match_layout() {
    assert_eq!(encode(Action::BuyDevCard), 226);
    assert_eq!(encode(Action::PlayKnight), 227);
    assert_eq!(encode(Action::PlayRoadBuilding), 228);
    assert_eq!(encode(Action::PlayMonopoly(Resource::Wood)), 229);
    assert_eq!(encode(Action::PlayMonopoly(Resource::Ore)), 233);
    assert_eq!(encode(Action::PlayYearOfPlenty(Resource::Wood, Resource::Wood)), 234);
    assert_eq!(encode(Action::PlayYearOfPlenty(Resource::Ore, Resource::Ore)), 258);
    assert_eq!(encode(Action::PlayVpCard), 259);
}
