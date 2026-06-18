//! Bit-exact port of NumPy's `default_rng(seed)` = PCG64 seeded via
//! `SeedSequence`. Verified against numpy for seeds {0,1,777,20_000_000}.
//!
//! Only the consumers the MCTS oracle uses are ported: `random_f64`,
//! `dirichlet`, `choice` (the latter two land in this file in Task 3, each with
//! their own numpy golden test). Source: numpy `random/_pcg64.pyx`,
//! `distributions.c`.

use crate::seedseq::SeedSequence;

const PCG_MULT: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

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
}
