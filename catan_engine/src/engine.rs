//! Top-level orchestrator. Stub for now — fleshed out across Phases 2–5.

pub struct Engine {
    _seed: u64,
}

impl Engine {
    pub fn new(seed: u64) -> Self {
        Self { _seed: seed }
    }

    pub fn is_terminal(&self) -> bool {
        // Stub: always true so smoke test "completes" trivially.
        // Will be replaced by real terminal check in Phase 5.
        true
    }

    pub fn legal_actions(&self) -> Vec<u32> {
        // Stub: empty. Smoke test will skip the inner loop because is_terminal=true.
        vec![]
    }

    pub fn step(&mut self, _action: u32) {
        // Stub.
    }
}
