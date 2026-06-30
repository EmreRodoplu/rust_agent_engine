use pyo3_stub_gen::Result;
use rust_agent_engine::stub_info; 

fn main() -> Result<()> {
    let stub = stub_info()?;
    stub.generate()?; 
    Ok(())
}