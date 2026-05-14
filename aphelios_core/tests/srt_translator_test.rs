use anyhow::Result;
use aphelios_core::{init_logging, srt_translator::process_translator};
use tracing::info;

#[tokio::test]
async fn process_test() -> Result<()> {
    init_logging();

    let srt_path = "/Users/larry/coderesp/aphelios_cli/output/download/This Book Predicted EVERYTHING… Why 1984 Still Feels So Real Today-en.srt";
    let output_path = "/Users/larry/coderesp/aphelios_cli/output/download/This Book Predicted EVERYTHING… Why 1984 Still Feels So Real Today.srt";

    let r = process_translator("A conversation about Orwell's novel 1984", srt_path, output_path);

    info!("✅ output: {}", r.await?);
    Ok(())
}
