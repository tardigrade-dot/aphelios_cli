use std::path::Path;

use anyhow::{Context, Result};
use aphelios_asr::qwen3asr::{self, load_audio_wav, AsrInference, StreamingOptions};
use aphelios_core::init_logging;
use tracing::info;

struct AudioInfo {
    pub path_str: String,
    pub language_str: String,
}

#[tokio::test]
async fn vad_and_qwen3asr() -> Result<()> {
    init_logging();
    info!("Starting vad_and_qwen3asr test");

    const ASR_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/Qwen3-ASR-0.6B";
    const ALIGNER_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/Qwen3-ForcedAligner-0.6B";
    let audio_li = vec![
        // AudioInfo{path_str:"/Users/larry/coderesp/aphelios_cli/test_data/b457.wav".to_string(), language_str:"Chinese".to_string()},
        AudioInfo {
            path_str: "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4.enhanced.wav".to_string(),
            language_str: "English".to_string(),
        }, // 150s
           // AudioInfo{path_str:"/Volumes/sw/download_video/Every Constitutional Crisis (in American History).mp4".to_string(), language_str:"English".to_string()}
    ];
    // let context = "the title of this audio : 1917 Centennial Series: War, Revolution, Socialism, War. Stephen Kotkin";
    let context = "the audio is a News";
    for au in audio_li {
        info!("Running ASR with audio: {}", au.path_str);
        let items = qwen3asr::qwen3asr_with_vad(Some(ASR_MODEL_DIR), Some(ALIGNER_MODEL_DIR), Some("/Volumes/sw/onnx_models/silero-vad/onnx"), &au.path_str, &au.language_str, Some(context)).await?;

        let final_text: Vec<String> = items
            .iter()
            .map(|i| i.text.to_string())
            .collect();
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

    // let input = "/Volumes/sw/video/mQlxALUw3h4.wav";
    // let language = "English";

    // no vad, for small audio file
    let input = "/Volumes/sw/video/qinsheng.wav";
    let language = "Chinese";

    tracing::info!("Running simple ASR with audio: {}", input);
    let text = qwen3asr::qwen3asr_simple(Some(qwen3asr_model), input, language).context("qwen3asr_simple error")?;
    tracing::info!("Result: {}", text);
    tracing::info!("qwen3asr_simple_test completed successfully");
    Ok(())
}

#[tokio::test]
async fn streaming_test() -> anyhow::Result<()> {
    init_logging();
    info!("Starting streaming ASR test");

    let model_dir = "/Volumes/sw/pretrained_models/Qwen3-ASR-0.6B";
    let audio_path = "/Volumes/sw/video/qinsheng.wav";
    let language = "Chinese";

    // ── Load model ──────────────────────────────────────────────────────
    let device = aphelios_core::utils::common::get_device();
    info!("Loading Qwen3-ASR model from {} on {:?}", model_dir, device);
    let asr = AsrInference::load(Some(model_dir), device)?;

    // ── Load audio ──────────────────────────────────────────────────────
    let samples = load_audio_wav(audio_path, 16000)?;
    info!("Loaded {} audio samples ({:.2}s)", samples.len(), samples.len() as f64 / 16000.0);

    // ── Configure streaming ─────────────────────────────────────────────
    let options = StreamingOptions::default()
        .with_language(language)
        .with_chunk_size_sec(1.0) // 1-second chunks
        .with_max_new_tokens_streaming(64);

    let mut state = asr.init_streaming(options);

    // ── Feed audio in small packets (~20ms each) to simulate real-time ──
    let packet_size = 320; // 20ms @ 16kHz
    let mut offset = 0;
    let mut step = 0;

    while offset < samples.len() {
        let end = (offset + packet_size).min(samples.len());
        let chunk = &samples[offset..end];

        if let Some(result) = asr.feed_audio(&mut state, chunk)? {
            step += 1;
            info!("[stream step {}] partial: {}", step, result.text);
        }

        offset = end;
    }

    // ── Final flush ─────────────────────────────────────────────────────
    info!("Flushing remaining audio...");
    let final_result = asr.finish_streaming(&mut state)?;
    info!("[final] language: {}, text: {}", final_result.language, final_result.text);

    assert!(!final_result.text.is_empty(), "final text should not be empty");
    info!("Streaming ASR test completed successfully");
    Ok(())
}
