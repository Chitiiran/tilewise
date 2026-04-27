use pyo3::prelude::*;

mod engine;
mod board;
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
