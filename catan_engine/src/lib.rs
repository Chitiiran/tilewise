use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyArray2, IntoPyArray};

pub mod actions;
pub mod board;
pub mod engine;
pub mod events;
pub mod observation;
pub mod rng;
pub mod rules;
pub mod state;
pub mod stats;

pub use engine::Engine;

#[pyclass(name = "Engine")]
struct PyEngine {
    inner: Engine,
}

#[pymethods]
impl PyEngine {
    #[new]
    fn new(seed: u64) -> Self {
        Self { inner: Engine::new(seed) }
    }

    /// Construct an engine with the fixed canonical board (no ABC randomization).
    /// Useful for v1-style reproducibility / regression tests.
    #[staticmethod]
    fn with_standard_board(seed: u64) -> Self {
        Self { inner: Engine::with_standard_board(seed) }
    }

    /// Construct with v3 rule flags. `vp_target` sets the win threshold
    /// (v2 default = 10). `bonuses_enabled=False` disables the +2 VP awards
    /// for longest road and largest army (v2 default = True). The holders
    /// are still tracked so observation features stay populated.
    #[staticmethod]
    fn with_rules(seed: u64, vp_target: u8, bonuses_enabled: bool) -> Self {
        Self { inner: Engine::with_rules(seed, vp_target, bonuses_enabled) }
    }

    /// Read the engine's VP threshold (v2=10, v3=5). Immutable for the
    /// engine's lifetime.
    fn vp_target(&self) -> u8 { self.inner.state.vp_target }

    /// Whether longest-road / largest-army grant +2 VP. Immutable for the
    /// engine's lifetime.
    fn bonuses_enabled(&self) -> bool { self.inner.state.bonuses_enabled }

    /// Read the current VP of player `p` (0..=3). Used by v3's VP-aware
    /// MCTS sim-budget schedule (smaller search budget when acting player
    /// is close to winning) and by the playback viewer.
    fn vp(&self, p: u8) -> PyResult<u8> {
        if p >= 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("player must be 0..=3, got {}", p)
            ));
        }
        Ok(self.inner.state.vp[p as usize])
    }

    fn legal_actions<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.inner.legal_actions().into_pyarray_bound(py)
    }

    /// Legal-action bitmap as a bool array. Bit i ⇔ action i legal.
    /// Faster than legal_actions() for "is this specific action legal?" queries
    /// and the natural input shape for GNN policy masks.
    /// Phase 2.5: cached, recomputed only when state mutates.
    fn legal_mask<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let m = self.inner.legal_mask();
        let bits: Vec<bool> = (0..crate::actions::LEGAL_MASK_BITS)
            .map(|i| m.get(i))
            .collect();
        bits.into_pyarray_bound(py)
    }

    fn step(&mut self, action_id: u32) {
        self.inner.step(action_id);
    }

    fn is_terminal(&self) -> bool { self.inner.is_terminal() }

    fn current_player(&self) -> u8 { self.inner.state.current_player }

    fn is_chance_pending(&self) -> bool {
        self.inner.is_chance_pending()
    }

    fn chance_outcomes(&self) -> Vec<(u32, f64)> {
        self.inner.chance_outcomes()
    }

    fn apply_chance_outcome(&mut self, value: u32) {
        self.inner.apply_chance_outcome(value);
    }

    /// M1 combined call: returns (is_terminal, is_chance_pending, current_player)
    /// in one PyO3 round-trip. The OpenSpiel adapter's `current_player()`
    /// previously made 3 separate FFI calls per query; this collapses them.
    fn query_status(&self) -> (bool, bool, u8) {
        (
            self.inner.is_terminal(),
            self.inner.is_chance_pending(),
            self.inner.state.current_player,
        )
    }

    /// M1 combined call: dispatches `step()` or `apply_chance_outcome()`
    /// based on the engine's current phase, then returns the post-action
    /// status tuple. Replaces the adapter's
    ///     if engine.is_chance_pending(): engine.apply_chance_outcome(a)
    ///     else: engine.step(a)
    /// pattern + the immediate-next status read MCTS does after every step.
    /// One PyO3 call instead of 4-5.
    fn apply_action_smart(&mut self, action_id: u32) -> (bool, bool, u8) {
        if self.inner.is_chance_pending() {
            self.inner.apply_chance_outcome(action_id);
        } else {
            self.inner.step(action_id);
        }
        (
            self.inner.is_terminal(),
            self.inner.is_chance_pending(),
            self.inner.state.current_player,
        )
    }

    fn random_rollout_to_terminal(&mut self, rollout_seed: u64) -> [f32; 4] {
        self.inner.random_rollout_to_terminal(rollout_seed)
    }

    fn lookahead_vp_value(&mut self, depth: u32, eval_seed: u64) -> [f32; 4] {
        self.inner.lookahead_vp_value(depth, eval_seed)
    }

    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }

    fn action_history(&self) -> Vec<u32> {
        self.inner.action_history().to_vec()
    }

    fn observation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let viewer = self.inner.state.current_player;
        self.observation_inner(py, viewer)
    }

    /// Observation tensors with explicit viewer (Phase 1.4 / wishlist §2c).
    /// Unlocks 4× perspective-rotated training data per game.
    fn observation_for<'py>(&self, py: Python<'py>, viewer: u8) -> PyResult<Bound<'py, PyDict>> {
        if viewer >= 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("viewer must be 0..3, got {}", viewer)
            ));
        }
        self.observation_inner(py, viewer)
    }

    /// Bank's remaining resources, [wood, brick, sheep, wheat, ore].
    /// Wishlist §2a — replaces Python-side reconstruction in scratch_card_tracker.py.
    fn bank<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        self.inner.state.bank.to_vec().into_pyarray_bound(py)
    }

    /// All 4 players' hands in absolute seat order (no perspective rotation).
    /// Returns a 4×5 numpy array. Wishlist §2b.
    fn all_hands<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let v: Vec<Vec<u8>> = self.inner.state.hands.iter().map(|h| h.to_vec()).collect();
        Ok(PyArray2::from_vec2_bound(py, &v)?)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stats();
        let d = PyDict::new_bound(py);
        d.set_item("schema_version", s.schema_version)?;
        d.set_item("turns_played", s.turns_played)?;
        d.set_item("seven_count", s.seven_count)?;
        d.set_item("dice_histogram", s.dice_histogram.to_vec().into_pyarray_bound(py))?;
        d.set_item("production_per_hex", s.production_per_hex.to_vec().into_pyarray_bound(py))?;
        d.set_item("winner_player_id", s.winner_player_id)?;
        let players = PyList::empty_bound(py);
        for p in 0..4 {
            let pd = PyDict::new_bound(py);
            pd.set_item("vp_final", s.players[p].vp_final)?;
            pd.set_item("won", s.players[p].won)?;
            pd.set_item("settlements_built", s.players[p].settlements_built)?;
            pd.set_item("cities_built", s.players[p].cities_built)?;
            pd.set_item("roads_built", s.players[p].roads_built)?;
            pd.set_item("cards_in_hand_max", s.players[p].cards_in_hand_max)?;
            pd.set_item("times_robbed", s.players[p].times_robbed)?;
            pd.set_item("robber_moves", s.players[p].robber_moves)?;
            pd.set_item("discards_triggered", s.players[p].discards_triggered)?;
            // Wishlist §2d: previously-tracked-but-not-surfaced fields.
            pd.set_item(
                "resources_gained",
                s.players[p].resources_gained.to_vec().into_pyarray_bound(py),
            )?;
            pd.set_item(
                "resources_gained_from_robber",
                s.players[p].resources_gained_from_robber.to_vec().into_pyarray_bound(py),
            )?;
            pd.set_item(
                "resources_lost_to_robber",
                s.players[p].resources_lost_to_robber.to_vec().into_pyarray_bound(py),
            )?;
            pd.set_item(
                "resources_lost_to_discard",
                s.players[p].resources_lost_to_discard.to_vec().into_pyarray_bound(py),
            )?;
            players.append(pd)?;
        }
        d.set_item("players", players)?;
        Ok(d)
    }
}

impl PyEngine {
    /// Shared implementation for observation() and observation_for().
    fn observation_inner<'py>(&self, py: Python<'py>, viewer: u8) -> PyResult<Bound<'py, PyDict>> {
        let obs = observation::build_observation(&self.inner.state, viewer);
        let d = PyDict::new_bound(py);
        d.set_item(
            "hex_features",
            PyArray2::from_vec2_bound(py, &chunks(&obs.hex_features, observation::F_HEX))?,
        )?;
        d.set_item(
            "vertex_features",
            PyArray2::from_vec2_bound(py, &chunks(&obs.vertex_features, observation::F_VERT))?,
        )?;
        d.set_item(
            "edge_features",
            PyArray2::from_vec2_bound(py, &chunks(&obs.edge_features, observation::F_EDGE))?,
        )?;
        d.set_item("scalars", obs.scalars.into_pyarray_bound(py))?;
        d.set_item(
            "legal_mask",
            obs.legal_mask.iter().map(|&b| b as u8).collect::<Vec<u8>>().into_pyarray_bound(py),
        )?;
        Ok(d)
    }
}

fn chunks(flat: &[f32], width: usize) -> Vec<Vec<f32>> {
    flat.chunks(width).map(|c| c.to_vec()).collect()
}

#[pyfunction]
fn engine_version() -> &'static str { "3.0.0-v3-flags" }

#[pyfunction]
fn action_space_size() -> usize { actions::ACTION_SPACE_SIZE }

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(engine_version, m)?)?;
    m.add_function(wrap_pyfunction!(action_space_size, m)?)?;
    m.add_class::<PyEngine>()?;
    Ok(())
}
