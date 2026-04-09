use anyhow::Result;
use aphelios_asr::text_match;
use aphelios_core::init_logging;
use tracing::info;

#[test]
fn sensevoice_generate_srt_test() -> Result<()> {
    init_logging();
    let input_audio = "/Volumes/sw/tts_result/wenhuaquanliyuguojia/wenhuaquanliyuguojia-0_0.wav";

    let srt_path = text_match::audio_text_match(input_audio, None)?;

    info!("SRT file saved to: {}", srt_path);

    Ok(())
}

#[test]
fn sensevoice_batch_process_wav_txt_dir_test() -> Result<()> {
    init_logging();
    let model_path = "/Volumes/sw/onnx_models/sensevoice";
    let dir_path = "/Volumes/sw/tts_result/wenhuaquanliyuguojia";
    let results = text_match::batch_process_wav_txt_dir(model_path, dir_path, None, None)?;

    info!(
        "Batch processing completed. Generated {} SRT files",
        results.len()
    );

    Ok(())
}
