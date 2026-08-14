use pyo3::prelude::*;


mod tools;
mod engine;
mod memory;
mod rag;
mod utils;
mod schedule;

#[pymodule]
fn _rust_agent_engine(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    
    m.add_class::<engine::PyAgent>()?;
    m.add_class::<engine::PyLLMConfig>()?;
    m.add_class::<memory::PyAgentMemory>()?;
    m.add_class::<rag::PyRagEngine>()?;
    m.add_class::<schedule::PyTaskManager>()?;
    
    Ok(())
}

