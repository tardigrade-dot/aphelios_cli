use anyhow::Result;
use aphelios_core::{init_logging, srt_translator::process_translator};
use tracing::info;

#[tokio::test]
async fn process_test() -> Result<()> {
    init_logging();

    let srt_path = "/Users/larry/codehub/aphelios_cli/output/download/This video will APPEAR every time someone ESCAPES the collective illusion — Yuval Harari ｜｜ Yuval-en.srt";
    let output_path = "/Users/larry/codehub/aphelios_cli/output/download/This video will APPEAR every time someone ESCAPES the collective illusion — Yuval Harari ｜｜ Yuval.srt";

    let context = "Inspired by the ideas of Yuval Noah Harari, this deep analysis uncovers the hidden psychological, emotional, and spiritual shift happening around the world.";
    let r = process_translator(context, srt_path, output_path);

    info!("✅ output: {}", r.await?);
    Ok(())
}
