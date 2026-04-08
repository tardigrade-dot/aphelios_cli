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

#[test]
fn sensevoice_generate_srt_test() -> Result<()> {
    init_logging();
    let input_audio = "/Users/larry/coderesp/aphelios_cli/test_data/wenhuaquanliyuguojia-12_0.wav";

    let srt_path = aphelios_asr::text_match::audio_text_match(input_audio, None)?;

    info!("SRT file saved to: {}", srt_path);

    Ok(())
}
