use anyhow::Result;
use aphelios_core::{init_logging, srt_translator::process_translator};
use tracing::info;

#[tokio::test]
async fn process_test() -> Result<()> {
    init_logging();

    let srt_path = "/Users/larry/coderesp/aphelios_cli/output/download/A Beginner's Guide to Kobo Abe (Speculative Reader)-original.srt";
    let output_path = "/Users/larry/coderesp/aphelios_cli/output/download/A Beginner's Guide to Kobo Abe (Speculative Reader).srt";

    let r = process_translator("A Beginner's Guide to Kobo Abe (Speculative Reader), 罗马化：Abe Kōbō, 对应的中文是: 安部 公房", srt_path, output_path);

    info!("✅ output: {}", r.await?);
    Ok(())
}
