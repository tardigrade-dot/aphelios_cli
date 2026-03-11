use crate::error::Result;

pub fn run(path: String) -> Result<()> {
    println!("init at {path}");
    Ok(())
}
