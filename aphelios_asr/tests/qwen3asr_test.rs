use anyhow::Result;
use aphelios_asr::qwenasr::qwen3asr_with_vad;
use aphelios_core::init_logging;

#[tokio::test]
async fn vad_and_qwenasr_test() -> Result<()> {
    init_logging();

    let audio_path = "/Volumes/sw/video/mQlxALUw3h4.wav";
    let language = "English";

    // let audio_path = "/Volumes/sw/video/zh-voice-example.wav";
    // let language = "Chinese";

    const ASR_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/Qwen3-ASR-0.6B";
    const ALIGNER_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/Qwen3-ForcedAligner-0.6B";

    // qwen3asr_with_vad now returns Vec<AlignItem> and saves SRT file automatically
    let _items = qwen3asr_with_vad(
        ASR_MODEL_DIR,
        ALIGNER_MODEL_DIR,
        "/Volumes/sw/onnx_models/silero-vad/onnx",
        audio_path,
        language,
    )?;

    Ok(())
}

#[test]
fn qwen3asr_simple_test() -> Result<()> {
    init_logging();
    let qwen3asr_model = "/Volumes/sw/pretrained_models/Qwen3-ASR-0.6B";
    let aligner_model = "/Volumes/sw/pretrained_models/Qwen3-ForcedAligner-0.6B";

    // let input = PathBuf::from("/Volumes/sw/video/mQlxALUw3h4.wav");
    // let language = "English";

    // no vad, for small audio file
    let input = "/Volumes/sw/video/qinsheng.wav";
    let language = "Chinese";

    aphelios_asr::qwenasr::qwen3asr_simple(&qwen3asr_model, &aligner_model, &input, language)?;
    Ok(())
}
