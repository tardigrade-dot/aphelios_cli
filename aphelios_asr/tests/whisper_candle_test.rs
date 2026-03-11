// https://github.com/openai/whisper/blob/main/whisper/model.py/rgs
// TODO:
// - Batch size greater than 1.
// - More token filters (SuppressBlanks, ApplyTimestampRules).

use anyhow::Result;
use aphelios_asr::whisper::run_asr;
use aphelios_core::utils::core_utils;
use tracing::info;

const AUDIO_SHORT: &str = "/Volumes/sw/video/mQlxALUw3h4.wav";
const AUDIO_LONG: &str = "/Volumes/sw/video/RYdrPg6xdYo.wav";

///38s
#[tokio::test]
async fn asr_test_short() -> Result<()> {
    core_utils::init_tracing();

    let audio = AUDIO_SHORT.to_string();
    let lang: Option<&str> = Some("en");

    info!("=== ASR Test | audio: {} | lang: {:?} ===", audio, lang);

    let _ = run_asr(&audio, lang).await;
    Ok(())
}
/// 305s / 256s
#[tokio::test]
async fn asr_test_long() -> Result<()> {
    core_utils::init_tracing();

    let audio = AUDIO_LONG.to_string();
    let lang: Option<&str> = Some("en");

    info!("=== ASR Test | audio: {} | lang: {:?} ===", audio, lang);

    let _ = run_asr(&audio, lang).await;
    Ok(())
}

#[tokio::test]
async fn asr_test() -> Result<()> {
    let audio_path = "/Volumes/sw/video/Why Does Joseph Stalin Matter？.wav";
    core_utils::init_tracing();

    let audio = audio_path.to_string();
    let lang: Option<&str> = Some("en");

    info!("=== ASR Test | audio: {} | lang: {:?} ===", audio, lang);

    let _ = run_asr(&audio, lang).await.unwrap();
    Ok(())
}
