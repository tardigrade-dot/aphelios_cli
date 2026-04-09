use anyhow::Result;
use aphelios_asr::sensevoice::{sensevoice_asr, SenseVoiceConfig};
use aphelios_core::init_logging;
use tracing::info;

#[test]
fn sensevoice_test() -> Result<()> {
    init_logging();
    let input_audio = "/Volumes/sw/video/qinsheng.wav";
    let sv_result = sensevoice_asr(
        "/Volumes/sw/onnx_models/sensevoice",
        input_audio,
        SenseVoiceConfig {
            language: "zh".to_string(),
            ..SenseVoiceConfig::default()
        },
    );

    assert!(sv_result.is_ok());
    let result = sv_result.unwrap();
    info!("{}", result.text);
    for ts in result.timestamp {
        info!("[{}] {},{}", ts.word, ts.start_sec, ts.end_sec);
    }
    Ok(())
}
