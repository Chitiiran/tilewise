//! Golden-parity tests: the Rust RNG replica must match numpy bit-for-bit.
//! Golden values from mcts_study/scripts/dump_rng_golden.py (seed 777).

use catan_mcts_rs::rng::NpRng;
use catan_mcts_rs::seedseq::SeedSequence;

#[test]
fn seedseq_state4_matches_numpy() {
    // np.random.SeedSequence(777).generate_state(4, uint64)
    let golden: [u64; 4] = [
        8247212343247536711,
        16696933637038441781,
        15338353104878458929,
        10095946210764358792,
    ];
    let got = SeedSequence::new(777).generate_state_u64(4);
    assert_eq!(got, golden.to_vec(), "SeedSequence words mismatch");
}

#[test]
fn seedseq_state4_matches_numpy_seed0() {
    let golden: [u64; 4] = [
        15793235383387715774,
        12390638538380655177,
        2361836109651742017,
        3188717715514472916,
    ];
    let got = SeedSequence::new(0).generate_state_u64(4);
    assert_eq!(got, golden.to_vec());
}

#[test]
fn pcg64_init_state_matches_numpy() {
    // default_rng(777).bit_generator.state["state"] = {state, inc}
    let rng = NpRng::from_seed(777);
    let (state, inc) = rng.raw_state();
    assert_eq!(state, 194185671395561036263475885081288615407u128, "PCG64 state");
    assert_eq!(inc, 225602981554823962289316896657083664657u128, "PCG64 inc");
}

#[test]
fn random_f64_matches_numpy() {
    // default_rng(777).random() x8
    let golden: [f64; 8] = [
        0.6110939299469712,
        0.38281659045082816,
        0.6000705254490022,
        0.9635578674390097,
        0.19616256242654817,
        0.33704142186654373,
        0.5684749820146611,
        0.5395921034316342,
    ];
    let mut rng = NpRng::from_seed(777);
    for (i, &g) in golden.iter().enumerate() {
        let x = rng.random_f64();
        assert_eq!(x.to_bits(), g.to_bits(), "draw {i}: {x} != {g}");
    }
}

#[test]
fn random_f64_matches_numpy_seed20m() {
    // default_rng(20_000_000).random() x5  (production seed_base scale)
    let golden: [f64; 5] = [
        0.255709522513785,
        0.6300281825415268,
        0.4523203556117994,
        0.9754447934184727,
        0.3004036134394109,
    ];
    let mut rng = NpRng::from_seed(20_000_000);
    for (i, &g) in golden.iter().enumerate() {
        let x = rng.random_f64();
        assert_eq!(x.to_bits(), g.to_bits(), "seed20m draw {i}: {x} != {g}");
    }
}
