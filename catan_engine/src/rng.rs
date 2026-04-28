//! Seeded RNG. All randomness flows through here for determinism.
//! Properly defined in Task 22; stub for now so rules.rs compiles.

use rand::rngs::SmallRng;
use rand::SeedableRng;

#[derive(Clone)]
pub struct Rng {
    inner: SmallRng,
}

impl Rng {
    pub fn from_seed(seed: u64) -> Self {
        Self { inner: SmallRng::seed_from_u64(seed) }
    }

    pub fn inner(&mut self) -> &mut SmallRng {
        &mut self.inner
    }
}
