//! Bit-exact port of NumPy's `default_rng(seed)` = PCG64 seeded via
//! `SeedSequence`. Verified against numpy for seeds {0,1,777,20_000_000}.
//!
//! Only the consumers the MCTS oracle uses are ported: `random_f64`,
//! `dirichlet`, `choice` (the latter two land in this file in Task 3, each with
//! their own numpy golden test). Source: numpy `random/_pcg64.pyx`,
//! `distributions.c`.

use crate::seedseq::SeedSequence;
use crate::ziggurat_tables::{FE_DOUBLE, FI_DOUBLE, KE_DOUBLE, KI_DOUBLE, WE_DOUBLE, WI_DOUBLE};

const PCG_MULT: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

// numpy distributions.c ziggurat constants.
const ZIGGURAT_NOR_R: f64 = 3.6541528853610087963519472518;
const ZIGGURAT_NOR_INV_R: f64 = 0.27366123732975827203338247596;
const ZIGGURAT_EXP_R: f64 = 7.6971174701310497140434110269;

/// NumPy PCG64 generator (XSL-RR-128 output).
pub struct NpRng {
    state: u128,
    inc: u128,
}

impl NpRng {
    /// Seed exactly as `np.random.default_rng(seed)`:
    /// SeedSequence -> 4 u64 words -> pcg_setseq_128_srandom_r.
    pub fn from_seed(seed: u64) -> Self {
        let w = SeedSequence::new(seed).generate_state_u64(4);
        let initstate = ((w[0] as u128) << 64) | (w[1] as u128);
        let initseq = ((w[2] as u128) << 64) | (w[3] as u128);
        let inc = (initseq << 1) | 1;
        let mut rng = Self { state: 0, inc };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    #[inline]
    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    /// XSL-RR 128 -> 64 output, post-step (matches numpy `pcg64_next64`).
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        let rot = (s >> 122) as u32;
        let xored = ((s >> 64) as u64) ^ (s as u64);
        xored.rotate_right(rot)
    }

    /// numpy `next_double`: top 53 bits scaled into [0, 1).
    #[inline]
    pub fn random_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// Test hook: the post-seed (state, inc) — compared to numpy's reported
    /// `bit_generator.state["state"]` {state, inc}.
    pub fn raw_state(&self) -> (u128, u128) {
        (self.state, self.inc)
    }

    /// numpy `random_standard_exponential` (ziggurat). Bit-exact vs numpy.
    pub fn standard_exponential(&mut self) -> f64 {
        loop {
            let mut ri = self.next_u64();
            ri >>= 3;
            let idx = (ri & 0xFF) as usize;
            ri >>= 8;
            let x = ri as f64 * WE_DOUBLE[idx];
            if ri < KE_DOUBLE[idx] {
                return x;
            }
            // standard_exponential_unlikely
            if idx == 0 {
                return ZIGGURAT_EXP_R - (-self.random_f64()).ln_1p();
            }
            if (FE_DOUBLE[idx - 1] - FE_DOUBLE[idx]) * self.random_f64() + FE_DOUBLE[idx]
                < (-x).exp()
            {
                return x;
            }
        }
    }

    /// numpy `random_standard_normal` (ziggurat). Bit-exact vs numpy.
    pub fn standard_normal(&mut self) -> f64 {
        loop {
            let r = self.next_u64();
            let idx = (r & 0xFF) as usize;
            let r = r >> 8;
            let sign = r & 0x1;
            let rabs = (r >> 1) & 0x000F_FFFF_FFFF_FFFF;
            let mut x = rabs as f64 * WI_DOUBLE[idx];
            if sign & 0x1 != 0 {
                x = -x;
            }
            if rabs < KI_DOUBLE[idx] {
                return x;
            }
            if idx == 0 {
                loop {
                    let xx = -ZIGGURAT_NOR_INV_R * (-self.random_f64()).ln_1p();
                    let yy = -(-self.random_f64()).ln_1p();
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 0x1 != 0 {
                            -(ZIGGURAT_NOR_R + xx)
                        } else {
                            ZIGGURAT_NOR_R + xx
                        };
                    }
                }
            } else if (FI_DOUBLE[idx - 1] - FI_DOUBLE[idx]) * self.random_f64() + FI_DOUBLE[idx]
                < (-0.5 * x * x).exp()
            {
                return x;
            }
        }
    }

    /// numpy `random_standard_gamma`. Bit-exact vs numpy.
    pub fn standard_gamma(&mut self, shape: f64) -> f64 {
        if shape == 1.0 {
            return self.standard_exponential();
        }
        if shape == 0.0 {
            return 0.0;
        }
        if shape < 1.0 {
            loop {
                let u = self.random_f64();
                let v = self.standard_exponential();
                if u <= 1.0 - shape {
                    let x = u.powf(1.0 / shape);
                    if x <= v {
                        return x;
                    }
                } else {
                    let y = -((1.0 - u) / shape).ln();
                    let x = (1.0 - shape + shape * y).powf(1.0 / shape);
                    if x <= v + y {
                        return x;
                    }
                }
            }
        }
        let b = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * b).sqrt();
        loop {
            let mut x;
            let mut v;
            loop {
                x = self.standard_normal();
                v = 1.0 + c * x;
                if v > 0.0 {
                    break;
                }
            }
            v = v * v * v;
            let u = self.random_f64();
            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return b * v;
            }
            if u.ln() < 0.5 * x * x + b * (1.0 - v + v.ln()) {
                return b * v;
            }
        }
    }

    /// numpy `Generator.dirichlet(alpha)` — STANDARD case (alpha.max() >= 0.1):
    /// unit-normalize a vector of standard_gamma draws. acc summed sequentially,
    /// then each *= 1/acc (multiply, matching numpy's `invacc`).
    ///
    /// PANICS if max(alpha) < 0.1 (the stick-breaking small-alpha path, not
    /// ported — the MCTS default dirichlet_alpha=0.8 never hits it; a future
    /// alpha<0.1 would need the Beta-RV path added + golden-tested).
    pub fn dirichlet(&mut self, alpha: &[f64]) -> Vec<f64> {
        let max_a = alpha.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_a >= 0.1,
            "dirichlet small-alpha (max {max_a} < 0.1) stick-breaking path not ported"
        );
        let mut out: Vec<f64> = alpha.iter().map(|&a| self.standard_gamma(a)).collect();
        let mut acc = 0.0;
        for &v in &out {
            acc += v;
        }
        let invacc = 1.0 / acc;
        for v in &mut out {
            *v *= invacc;
        }
        out
    }

    /// numpy `Generator.choice(items, p=...)` (with replacement, single draw):
    /// idx = searchsorted(cumsum(p) with last=1.0, random_f64(), side='right').
    /// Returns the chosen item.
    pub fn choice_i64(&mut self, items: &[i64], p: &[f64]) -> i64 {
        let mut cum = Vec::with_capacity(p.len());
        let mut s = 0.0;
        for &pi in p {
            s += pi;
            cum.push(s);
        }
        if let Some(last) = cum.last_mut() {
            *last = 1.0; // numpy clamps the final cdf entry to 1.0
        }
        let r = self.random_f64();
        // searchsorted(..., side='right'): first index with cum[idx] > r.
        let mut idx = cum.len() - 1;
        for (i, &c) in cum.iter().enumerate() {
            if r < c {
                idx = i;
                break;
            }
        }
        items[idx]
    }
}
