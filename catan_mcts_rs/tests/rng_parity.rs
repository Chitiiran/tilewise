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

#[test]
fn standard_exponential_matches_numpy() {
    // default_rng(seed).standard_exponential(5)
    let cases: [(u64, [f64; 5]); 3] = [
        (0, [0.6799319039689096, 1.0195971014658647, 0.019806662589055352,
             0.0022693266812281823, 0.5503428726390482]),
        (777, [2.3377578064925664, 0.09710462981073235, 2.688443016862102,
               0.8533552384069857, 0.04975827454270517]),
        (20_000_000, [0.28118294422095275, 1.201654315042931, 1.6506054133859331,
                      0.5922473751711548, 0.3303293974003371]),
    ];
    for (seed, golden) in cases {
        let mut rng = NpRng::from_seed(seed);
        for (i, &g) in golden.iter().enumerate() {
            let x = rng.standard_exponential();
            assert_eq!(x.to_bits(), g.to_bits(), "exp seed={seed} i={i}: {x} != {g}");
        }
    }
}

#[test]
fn standard_gamma_matches_numpy() {
    // default_rng(seed).standard_gamma(0.8, size=6)
    let cases: [(u64, [f64; 6]); 3] = [
        (0, [0.7947089517143643, 1.474014702027899, 0.718838397882517,
             0.5825554017936307, 1.770968789599147, 1.085623609841791]),
        (777, [0.7295240602999434, 0.7033772322389736, 0.13054798035673032,
               0.6332359370134244, 1.1100059673213316, 0.028173371044258152]),
        (20_000_000, [0.18364803337249633, 0.4237481143443123, 0.22878596860061498,
                      0.0005984018061264664, 0.18244889469844758, 1.528103304220895]),
    ];
    for (seed, golden) in cases {
        let mut rng = NpRng::from_seed(seed);
        for (i, &g) in golden.iter().enumerate() {
            let x = rng.standard_gamma(0.8);
            assert_eq!(x.to_bits(), g.to_bits(), "gamma seed={seed} i={i}: {x} != {g}");
        }
    }
}

#[test]
fn standard_normal_matches_numpy() {
    let cases: [(u64, [f64; 6]); 3] = [
        (0, [0.1257302210933933, -0.1321048632913019, 0.6404226504432821,
             0.10490011715303971, -0.535669373161111, 0.36159505490948474]),
        (777, [-0.8475155145647386, 0.06854253280286053, -1.2509259734323444,
               -1.5836366914181446, 0.6324575844117477, -0.4696753890279187]),
        (20_000_000, [0.06552308676543303, 0.06762224520317445, -0.3815895571459841,
                      -0.455889392225337, 0.589229597044965, 0.3956658278454601]),
    ];
    for (seed, golden) in cases {
        let mut rng = NpRng::from_seed(seed);
        for (i, &g) in golden.iter().enumerate() {
            let x = rng.standard_normal();
            assert_eq!(x.to_bits(), g.to_bits(), "normal seed={seed} i={i}: {x} != {g}");
        }
    }
}

#[test]
fn dirichlet_matches_numpy() {
    let cases: [(u64, [f64; 4]); 3] = [
        (0, [0.22260022592629977, 0.4128756886181771,
             0.20134866912970176, 0.16317541632582147]),
        (777, [0.3321022315847162, 0.32019937543444776,
               0.05942953490460936, 0.28826885807622665]),
        (20_000_000, [0.2194697765960426, 0.5064029397990193,
                      0.27341215963495324, 0.0007151239699848184]),
    ];
    for (seed, golden) in cases {
        let mut rng = NpRng::from_seed(seed);
        let out = rng.dirichlet(&[0.8, 0.8, 0.8, 0.8]);
        for (i, (&a, &b)) in out.iter().zip(golden.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "dirichlet seed={seed} i={i}: {a} != {b}");
        }
    }
}

#[test]
fn choice_matches_numpy() {
    let cases: [(u64, [i64; 8]); 3] = [
        (0, [5, 2, 0, 0, 5, 5, 5, 5]),
        (777, [5, 2, 5, 5, 0, 2, 5, 5]),
        (20_000_000, [2, 5, 2, 5, 2, 5, 0, 5]),
    ];
    let items = [0i64, 2, 5];
    let p = [0.2, 0.3, 0.5];
    for (seed, golden) in cases {
        let mut rng = NpRng::from_seed(seed);
        for (i, &g) in golden.iter().enumerate() {
            assert_eq!(rng.choice_i64(&items, &p), g, "choice seed={seed} i={i}");
        }
    }
}
