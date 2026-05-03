use anyhow::Result;
use aphelios_asr::qwenasr::qwen3asr_with_vad;
use aphelios_core::init_logging;
use tracing::info;

#[test]
fn test_features() {
    init_logging();
    info!("end");
}

#[tokio::test]
async fn vad_and_qwenasr_test() -> Result<()> {
    // Initialize logging - will write to file in release mode, console in dev mode
    init_logging();
    tracing::info!("Starting vad_and_qwenasr_test");

    let audio_path = "/Volumes/sw/video/Why Does Joseph Stalin Matter？.mp4";
    let language = "English";

    // let audio_path = "/Volumes/sw/video/中国人为何容易陷入无限内斗？中国社会为何盛行猎巫思维？一集讲透中国社会“驱魔仪式”的底层逻辑｜文革｜中国文化｜哲学｜心理学｜.wav";
    // let language = "Chinese";

    const ASR_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/Qwen3-ASR-0.6B";
    const ALIGNER_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/Qwen3-ForcedAligner-0.6B";

    tracing::info!("Running ASR with audio: {}", audio_path);
    // qwen3asr_with_vad now returns Vec<AlignItem> and saves SRT file automatically
    let _items = qwen3asr_with_vad(
        ASR_MODEL_DIR,
        ALIGNER_MODEL_DIR,
        "/Volumes/sw/onnx_models/silero-vad/onnx",
        audio_path,
        language,
    )
    .await?;

    tracing::info!("vad_and_qwenasr_test completed successfully");
    Ok(())
}

#[test]
fn qwen3asr_simple_test() -> Result<()> {
    // Initialize logging - will write to file in release mode, console in dev mode
    init_logging();
    tracing::info!("Starting qwen3asr_simple_test");

    let qwen3asr_model = "/Volumes/sw/pretrained_models/Qwen3-ASR-0.6B";
    let aligner_model = "/Volumes/sw/pretrained_models/Qwen3-ForcedAligner-0.6B";

    // let input = PathBuf::from("/Volumes/sw/video/mQlxALUw3h4.wav");
    // let language = "English";

    // no vad, for small audio file
    let input = "/Volumes/sw/video/qinsheng.wav";
    let language = "Chinese";

    tracing::info!("Running simple ASR with audio: {}", input);
    aphelios_asr::qwenasr::qwen3asr_simple(&qwen3asr_model, &aligner_model, &input, language)?;
    tracing::info!("qwen3asr_simple_test completed successfully");
    Ok(())
}
