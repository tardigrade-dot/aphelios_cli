use anyhow::Result;
use aphelios_core::{init_logging, srt_translator::process_translator};
use tracing::info;

#[tokio::test]
async fn process_test() -> Result<()> {
    init_logging();

    let srt_path = "/Users/larry/coderesp/aphelios_cli/output/download/Robert Service on Trotsky 07⧸26⧸2010-en.srt";
    let output_path = "/Users/larry/coderesp/aphelios_cli/output/download/Robert Service on Trotsky 07⧸26⧸2010.srt";

    let r = process_translator("Robert Service of Stanford University's Hoover Institution and the University of Oxford talks with EconTalk host Russ Roberts about the life and death of Leon Trotsky. Based on Service's biography of Trotsky, the conversation covers Trotsky's influence on the Russian Revolution, his influence on policy alongside Lenin, his expulsion from Soviet Union in 1928 and his murder in 1940 by Stalin's order", srt_path, output_path);

    info!("✅ output: {}", r.await?);
    Ok(())
}
