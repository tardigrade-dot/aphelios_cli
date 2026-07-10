mod config;
mod decoder;
mod encoder;
mod error;
mod inference;
mod linear;
mod mel;
mod streaming;

pub use encoder::EncoderCache;
pub use error::{AsrError, Result};
pub use inference::{AsrInference, TranscribeOptions, TranscribeResult};
pub use mel::load_audio_wav;
pub use streaming::{StreamingOptions, StreamingState};

// ── High-level public API ─────────────────────────────────────────────────────

use std::path::Path;

use anyhow::Context;
use aphelios_core::{measure_time, utils::common::get_device};
use tracing::info;

/// Simple transcription using Qwen3-ASR (new implementation).
///
/// Loads the ASR model, transcribes a WAV file, and returns the transcribed text.
/// Does NOT perform forced alignment — use the qwenasr aligner separately if needed.
pub fn qwen3asr_simple(
    asr_model: Option<&str>,
    input: &str,
    language: &str,
) -> anyhow::Result<String> {
    let device = get_device();
    let asr = measure_time!(
        "load Qwen3-ASR model",
        AsrInference::load(asr_model, device)
            .context("Failed to load Qwen3-ASR model")?
    );

    let options = TranscribeOptions::default()
        .with_language(language);

    let result = measure_time!(
        "Qwen3ASR",
        asr.transcribe(input, options)
            .map_err(|e| anyhow::anyhow!("Qwen3ASR inference error: {}", e))?
    );

    info!("Language: {}", result.language);
    info!("{}", result.text);
    Ok(result.text)
}

/// Simple transcription with optional context prefix.
///
/// Like [`qwen3asr_simple`] but injects a context string into the prompt
/// to guide the model's output style and vocabulary.
pub fn qwen3asr_simple_with_context(
    asr_model: Option<&str>,
    input: &str,
    language: &str,
    context: Option<&str>,
) -> anyhow::Result<String> {
    let device = get_device();

    let asr = measure_time!(
        "load Qwen3-ASR model",
        AsrInference::load(asr_model, device)
            .context("Failed to load Qwen3-ASR model")?
    );

    let mut options = TranscribeOptions::default()
        .with_language(language);
    if let Some(ctx) = context {
        options = options.with_system_prompt(ctx);
    }

    let result = measure_time!(
        "Qwen3ASR",
        asr.transcribe(input, options)
            .map_err(|e| anyhow::anyhow!("Qwen3ASR inference error: {}", e))?
    );

    info!("Language: {}", result.language);
    info!("{}", result.text);
    Ok(result.text)
}

/// VAD + ASR pipeline using Qwen3-ASR.
///
/// 1. Runs Silero VAD to detect speech segments
/// 2. Transcribes each segment batch using Qwen3-ASR
/// 3. Optionally runs forced alignment on each batch
///
/// Returns aligned items with timestamps.
#[allow(clippy::too_many_arguments)]
pub async fn qwen3asr_with_vad(
    asr_model: Option<&str>,
    aligner_model: Option<&str>,
    vad_model_dir: Option<&str>,
    audio_path: &str,
    language: &str,
    ctx: Option<&str>,
) -> anyhow::Result<Vec<crate::qwenasr::aligner::AlignItem>> {
    use crate::qwen3asr::load_audio_wav;
    use crate::qwenasr::aligner::ForcedAligner;
    use crate::silerovad::VadProcessor;

    assert!(Path::new(audio_path).exists(), "file not exists!");

    // ── Phase 1: VAD ──────────────────────────────────────────────────────
    let mut vad = measure_time!("load VAD model", VadProcessor::new_default(vad_model_dir)?);
    let segments = measure_time!("VAD process", vad.process_from_file(audio_path)?);

    #[cfg(feature = "profiling")]
    {
        let output_path = Path::new(audio_path)
            .with_file_name(Path::new(audio_path).file_stem().unwrap().to_str().unwrap())
            .with_extension("vad.srt")
            .to_str()
            .unwrap()
            .to_string();
        let _ = crate::whisper::generate_vad(&segments, &output_path).await;
        info!("[profiling] Generated VAD SRT at {}", output_path);
    }
    info!("[Phase 1] Detected {} speech segments", segments.len());
    assert!(!segments.is_empty(), "Should detect at least one speech segment");

    info!("[Phase 1] Aggregating segments...");
    let batches = vad.aggregate_segments(&segments, 30.0, 0.3);
    info!("[Phase 1] Aggregated {} batches", batches.len());

    // ── Phase 2: Load models ──────────────────────────────────────────────
    let device = get_device();
    info!("[Phase 2] Loading Qwen3-ASR model");
    let asr = measure_time!(
        "load ASR model",
        AsrInference::load(asr_model, device)?
    );

    // Load entire audio as float samples for batch extraction
    let samples = load_audio_wav(audio_path, 16000)?;
    let sample_rate: f64 = 16000.0;
    let padding_duration = 0.01; // 10ms padding

    // ── Phase 3: Transcribe all batches ───────────────────────────────────
    let mut transcription_batches: Vec<TranscribeBatch> = Vec::new();
    let batch_size = batches.len();
    info!("[Phase 3] Transcribing {} batches...", batch_size);

    for (i, batch) in batches.iter().enumerate() {
        let padded_start = f64::max(0.0, batch.start - padding_duration);
        let padded_end = (batch.end + padding_duration).min(samples.len() as f64 / sample_rate);

        let start_sample = (padded_start * sample_rate) as usize;
        let end_sample = (padded_end * sample_rate) as usize;

        if start_sample < samples.len() {
            let end = std::cmp::min(end_sample, samples.len());
            let batch_pcm = samples[start_sample..end].to_vec();

            if !batch_pcm.is_empty() {
                let audio_ms = batch_pcm.len() as f64 / sample_rate * 1000.0;
                let mut options = TranscribeOptions::default()
                    .with_language(language);
                if let Some(c) = ctx {
                    options = options.with_system_prompt(c);
                }

                let result = measure_time!(
                    format!(
                        "transcribe Batch {}/{}: duration {:.2}s",
                        i + 1,
                        batch_size,
                        audio_ms / 1000.0
                    ),
                    asr.transcribe_samples(&batch_pcm, options)
                        .map_err(|e| anyhow::anyhow!("ASR error: {}", e))?
                );

                let preview: String = result.text.chars().take(30).collect();
                info!(
                    "[Phase 3] Batch {}/{}: text len={} text: {}...",
                    i + 1,
                    batch_size,
                    result.text.len(),
                    preview,
                );

                let text_str = result.text.trim();
                if !text_str.is_empty() {
                    transcription_batches.push(TranscribeBatch {
                        pcm: batch_pcm,
                        start_time: padded_start,
                        speech_end_time: batch.end,
                        text: text_str.to_string(),
                    });
                }
            }
        }
    }

    info!(
        "[Phase 3] Transcription complete. Total batches: {}",
        transcription_batches.len()
    );

    // ── Phase 4: Alignment ────────────────────────────────────────────────
    let aligner = ForcedAligner::load_with_device(aligner_model)?;

    let mut total_aligned_items: Vec<crate::qwenasr::aligner::AlignItem> = Vec::new();
    info!("[Phase 4] Running alignment on {} batches...", transcription_batches.len());

    for (i, batch) in transcription_batches.iter().enumerate() {
        #[cfg(not(feature = "profiling"))]
        let _ = i;
        let mut items = measure_time!(
            format!("[Phase 4] Batch {}/{} alignment...", i + 1, batch_size),
            aligner.align_samples(&batch.pcm, &batch.text, language)?
        );

        // Convert to absolute timestamps
        for item in &mut items {
            item.start_time += batch.start_time;
            item.end_time += batch.start_time;
        }
        total_aligned_items.extend(items.iter().cloned());
    }

    info!(
        "[Phase 4] Alignment complete. Total aligned items: {}",
        total_aligned_items.len()
    );

    Ok(total_aligned_items)
}

/// Batch of transcribed audio for downstream alignment.
#[allow(dead_code)]
struct TranscribeBatch {
    pcm: Vec<f32>,
    start_time: f64,
    speech_end_time: f64,
    text: String,
}
