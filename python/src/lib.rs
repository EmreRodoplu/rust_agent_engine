use pyo3::prelude::*;

mod tools;
mod engine;

#[pymodule]
fn rust_agent_engine(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<engine::PyLLMConfig>()?;
    m.add_class::<engine::PyAgent>()?;
    Ok(())
}