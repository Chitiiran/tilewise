//! MT19937 (CPython random.Random) golden parity. Golden from
//! mcts_study/scripts/dump_rng_golden.py (mt19937 section).

use catan_mcts_rs::mt19937::MtRng;

#[test]
fn mt_random_matches_cpython() {
    // random.Random(seed).random() x8
    let cases: [(u64, [f64; 8]); 3] = [
        (777, [
            0.22933408950153078, 0.44559617334521107, 0.36859824937216046,
            0.269835098321503, 0.3361436466700177, 0.7523163560031157,
            0.9226950812763804, 0.9122532879410743,
        ]),
        (0, [
            0.8444218515250481, 0.7579544029403025, 0.420571580830845,
            0.25891675029296335, 0.5112747213686085, 0.4049341374504143,
            0.7837985890347726, 0.30331272607892745,
        ]),
        (20_000_000, [
            0.006559064519161217, 0.8343648462909564, 0.6377326717606256,
            0.9770409987921411, 0.17509735816867567, 0.887977358261472,
            0.6419977586851477, 0.4464813144662817,
        ]),
    ];
    for (seed, golden) in cases {
        let mut rng = MtRng::from_seed(seed);
        for (i, &g) in golden.iter().enumerate() {
            let x = rng.random_f64();
            assert_eq!(x.to_bits(), g.to_bits(), "mt seed={seed} i={i}: {x} != {g}");
        }
    }
}
