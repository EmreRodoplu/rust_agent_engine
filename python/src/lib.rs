use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

mod tools;
mod engine;

#[pymodule]
fn rust_agent_engine(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<engine::PyLLMConfig>()?;
    m.add_class::<engine::PyAgent>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);