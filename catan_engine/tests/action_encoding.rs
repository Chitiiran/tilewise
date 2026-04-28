use catan_engine::actions::{Action, ACTION_SPACE_SIZE, decode, encode};
use catan_engine::board::Resource;

#[test]
fn action_space_size_is_206() {
    assert_eq!(ACTION_SPACE_SIZE, 206);
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
    assert!(decode(206).is_none());
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
fn action_space_size_is_206_after_roll_dice() {
    assert_eq!(catan_engine::actions::ACTION_SPACE_SIZE, 206);
}
