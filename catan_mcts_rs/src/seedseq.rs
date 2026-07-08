//! Bit-exact port of NumPy's `SeedSequence` (pool_size=4) for an integer seed.
//!
//! Verified against `np.random.SeedSequence(seed).generate_state(n, uint64)` for
//! seeds {0, 1, 777, 20_000_000} (see mcts_study/scripts/probe_seedseq.py).
//! Source: numpy `random/bit_generator.pyx` / randomgen `_seed_sequence.pyx`.

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16; // itemsize(u32)*8/2

const POOL_SIZE: usize = 4;

/// `hashmix`: folds `value` into the running `hash_const` (mutated in place).
#[inline]
fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^= v >> XSHIFT;
    v
}

#[inline]
fn mix(x: u32, y: u32) -> u32 {
    let mut result = x.wrapping_mul(MIX_MULT_L).wrapping_sub(y.wrapping_mul(MIX_MULT_R));
    result ^= result >> XSHIFT;
    result
}

/// Coerce a non-negative integer seed to little-endian u32 limbs (0 -> [0]).
fn coerce_to_u32_words(seed: u64) -> Vec<u32> {
    if seed == 0 {
        return vec![0];
    }
    let mut words = Vec::new();
    let mut n = seed;
    while n > 0 {
        words.push((n & 0xFFFF_FFFF) as u32);
        n >>= 32;
    }
    words
}

pub struct SeedSequence {
    pool: [u32; POOL_SIZE],
}

impl SeedSequence {
    pub fn new(seed: u64) -> Self {
        let entropy = coerce_to_u32_words(seed);
        let mut pool = [0u32; POOL_SIZE];
        let mut hash_const = INIT_A;

        // Fill the pool from entropy (or 0) up to pool size.
        for i in 0..POOL_SIZE {
            let v = if i < entropy.len() { entropy[i] } else { 0 };
            pool[i] = hashmix(v, &mut hash_const);
        }
        // Mix all pool bits together so late bits affect earlier bits.
        for i_src in 0..POOL_SIZE {
            for i_dst in 0..POOL_SIZE {
                if i_src != i_dst {
                    let h = hashmix(pool[i_src], &mut hash_const);
                    pool[i_dst] = mix(pool[i_dst], h);
                }
            }
        }
        // Mix in any remaining entropy beyond the pool size.
        for i_src in POOL_SIZE..entropy.len() {
            for i_dst in 0..POOL_SIZE {
                let h = hashmix(entropy[i_src], &mut hash_const);
                pool[i_dst] = mix(pool[i_dst], h);
            }
        }
        Self { pool }
    }

    /// Generate `n` u32 words (matches numpy's `generate_state(n, uint32)`).
    pub fn generate_state_u32(&self, n: usize) -> Vec<u32> {
        let mut hash_const = INIT_B;
        let mut out = Vec::with_capacity(n);
        let mut src_cycle = 0usize;
        for _ in 0..n {
            let mut data_val = self.pool[src_cycle];
            data_val ^= hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            data_val = data_val.wrapping_mul(hash_const);
            data_val ^= data_val >> XSHIFT;
            out.push(data_val);
            src_cycle = (src_cycle + 1) % POOL_SIZE;
        }
        out
    }

    /// Generate `n` u64 words (numpy combines u32 pairs little-endian: lo, hi).
    pub fn generate_state_u64(&self, n: usize) -> Vec<u64> {
        let w = self.generate_state_u32(n * 2);
        (0..n).map(|i| (w[2 * i] as u64) | ((w[2 * i + 1] as u64) << 32)).collect()
    }
}
