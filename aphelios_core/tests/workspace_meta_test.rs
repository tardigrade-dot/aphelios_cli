#[test]
fn test_test() -> anyhow::Result<()> {
    let version = env!("CARGO_PKG_VERSION");
    println!("version: {}", version);
    Ok(())
}
