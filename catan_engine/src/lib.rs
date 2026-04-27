use pyo3::prelude::*;

mod engine;
pub mod board;
pub mod actions;
pub mod events;
pub mod state;
pub use engine::Engine;
pub use board::{Board, Hex, Resource};

#[pyfunction]
fn engine_version() -> &'static str {
    "0.1.0"
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(engine_version, m)?)?;
    Ok(())
}
