//! Bit-exact port of CPython's `random.Random` (MT19937) `random()`.
//!
//! The arena GAME-level chance fast-path uses `random.Random(seed)` (stdlib
//! Mersenne), NOT numpy — see catan_az.arena._play_arena_game. So the Rust
//! arena must replicate it too. Golden-tested vs Python for seeds 0/777/20M.
//!
//! Source: CPython Modules/_randommodule.c. Integer seeding: abs(seed) as
//! little-endian u32 words -> init_by_array.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

pub struct MtRng {
    mt: [u32; N],
    index: usize,
}

impl MtRng {
    pub fn from_seed(seed: u64) -> Self {
        // Integer seed -> abs value as little-endian u32 key words (>=1 word).
        let mut key: Vec<u32> = Vec::new();
        let mut n = seed; // seeds are non-negative here
        if n == 0 {
            key.push(0);
        } else {
            while n > 0 {
                key.push((n & 0xFFFF_FFFF) as u32);
                n >>= 32;
            }
        }
        let mut r = MtRng { mt: [0u32; N], index: N + 1 };
        r.init_by_array(&key);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for mti in 1..N {
            let prev = self.mt[mti - 1];
            self.mt[mti] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(mti as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, init_key: &[u32]) {
        self.init_genrand(19650218);
        let key_length = init_key.len();
        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = if N > key_length { N } else { key_length };
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ ((prev ^ (prev >> 30)).wrapping_mul(1664525)))
                .wrapping_add(init_key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key_length {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ ((prev ^ (prev >> 30)).wrapping_mul(1566083941)))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x8000_0000;
    }

    fn genrand_u32(&mut self) -> u32 {
        if self.index >= N {
            let mag01 = [0u32, MATRIX_A];
            for kk in 0..(N - M) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            for kk in (N - M)..(N - 1) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] =
                    self.mt[kk + M - N] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// CPython `random()`: (a*2^26 + b) / 2^53, a=u32>>5, b=u32>>6.
    pub fn random_f64(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }
}
