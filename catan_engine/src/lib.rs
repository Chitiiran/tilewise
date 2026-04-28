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

    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }

    fn action_history(&self) -> Vec<u32> {
        self.inner.action_history().to_vec()
    }

    fn observation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let viewer = self.inner.state.current_player;
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
            players.append(pd)?;
        }
        d.set_item("players", players)?;
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
