use anyhow::{Context, Result};
use aphelios_asr::qwenasr;
use aphelios_core::init_logging;
use tracing::info;

struct AudioInfo {
    pub path_str: String,
    pub language_str: String,
}

#[tokio::test]
async fn vad_and_qwenasr() -> Result<()> {
    init_logging();
    info!("Starting vad_and_qwenasr_test");

    const ASR_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/Qwen3-ASR-0.6B";
    const ALIGNER_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/Qwen3-ForcedAligner-0.6B";
    let audio_li = vec![
        // AudioInfo{path_str:"/Users/larry/coderesp/aphelios_cli/test_data/b457.wav".to_string(), language_str:"Chinese".to_string()},
        // AudioInfo{path_str:"/Volumes/sw/video/mQlxALUw3h4.wav".to_string(), language_str:"English".to_string()},
        AudioInfo{path_str:"/Volumes/sw/video/j3UDNtjvv7E-This Is Fascism.mp4".to_string(), language_str:"English".to_string()}
    ];

    for au in audio_li{
        info!("Running ASR with audio: {}", au.path_str);
        let items = qwenasr::qwen3asr_with_vad(
            ASR_MODEL_DIR,
            ALIGNER_MODEL_DIR,
            "/Volumes/sw/onnx_models/silero-vad/onnx",
            &au.path_str,
            &au.language_str,
        )
        .await?;

        let final_text: Vec<String> = items.iter().map(|i|{i.text.to_string()}).collect();
        info!("final_text : {}", final_text.join(" "));
    }
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
    qwenasr::qwen3asr_simple(&qwen3asr_model, &aligner_model, &input, language).context("qwen3asr_simple error")?;
    tracing::info!("qwen3asr_simple_test completed successfully");
    Ok(())
}
