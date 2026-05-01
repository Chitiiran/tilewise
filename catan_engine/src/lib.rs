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

    fn legal_actions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.inner.legal_actions().into_pyarray_bound(py)
    }

    /// Legal-action bitmap as a length-256 bool array. Bit i ⇔ action i legal.
    /// Faster than legal_actions() for "is this specific action legal?" queries
    /// and the natural input shape for GNN policy masks.
    fn legal_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
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
fn engine_version() -> &'static str { "0.1.0" }

#[pyfunction]
fn action_space_size() -> usize { actions::ACTION_SPACE_SIZE }

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(engine_version, m)?)?;
    m.add_function(wrap_pyfunction!(action_space_size, m)?)?;
    m.add_class::<PyEngine>()?;
    Ok(())
}
