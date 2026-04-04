use std::path::Path;

use anyhow::Result;
use aphelios_core::utils::base::get_device;
use candle_core::Tensor;
use tracing::info;

use crate::{
    qwenasr::{
        aligner::{AlignItem, ForcedAligner},
        transcribe::Pipeline,
    },
    silerovad::VadProcessor,
};
pub mod aligner;
pub mod audio;
pub mod decoder;
pub mod encoder;
pub mod model;
pub mod preset;
pub mod tokenizer;
pub mod transcribe;

pub fn qwen3asr_simple(
    qwen3asr_model: &str,
    aligner_model: &str,
    input: &str,
    language: &str,
) -> Result<()> {
    let device = get_device();
    let mut pipeline =
        transcribe::Pipeline::load_with_device(Path::new(qwen3asr_model), device.clone())
            .unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1)
            });

    let text = pipeline.transcribe(input).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1)
    });

    info!("{text}");

    let aligner =
        ForcedAligner::load_with_device(Path::new(aligner_model), device).unwrap_or_else(|e| {
            eprintln!("error loading aligner: {e}");
            std::process::exit(1)
        });

    let items = aligner
        .align(Path::new(input), &text, language)
        .unwrap_or_else(|e| {
            eprintln!("error during alignment: {e}");
            std::process::exit(1)
        });

    info!("\nAlignment results:");
    let mut items_info = Vec::new();
    for item in &items {
        items_info.push(format!(
            "[{:.3} - {:.3}] {}",
            item.start_time, item.end_time, item.text
        ));
    }
    info!("{}", items_info.join(""));
    Ok(())
}

/// TranscribeBatch 保存每个batch的转录结果，用于后续对齐
struct TranscribeBatch {
    /// 原始音频采样 (带padding)
    pcm: Vec<f32>,
    /// batch起始时间（绝对时间，秒）
    start_time: f64,
    /// batch结束时间（绝对时间，秒）
    end_time: f64,
    /// 转录文本
    text: String,
}

pub fn qwen3asr_with_vad(
    qwen3asr_model: &str,
    aligner_model: &str,
    vad_model_dir: &str,
    audio_path: &str,
    language: &str,
) -> Result<Vec<AlignItem>> {
    // ==================== Phase 1: VAD + ASR Transcription ====================
    info!("[Phase 1] Loading VAD model from {}", vad_model_dir);
    let mut vad = VadProcessor::new_default(vad_model_dir)?;
    info!("[Phase 1] Running VAD on {}", audio_path);
    let segments = vad.process_from_file(audio_path)?;
    info!("[Phase 1] Detected {} speech segments", segments.len());

    assert!(
        !segments.is_empty(),
        "Should detect at least one speech segment"
    );
    info!("[Phase 1] Aggregating segments...");
    let batches = vad.aggregate_segments(&segments, 30.0, 0.3);
    info!("[Phase 1] Aggregated {} batches", batches.len());

    let device = get_device();

    info!(
        "[Phase 1] Loading QwenASR model from {} on {:?}",
        qwen3asr_model, device
    );
    let mut pipeline = Pipeline::load_with_device(Path::new(qwen3asr_model), device.clone())?;

    // Load entire audio as float samples
    let samples = audio::load_wav(Path::new(audio_path), &pipeline.audio_cfg)?;
    let sample_rate = pipeline.audio_cfg.sample_rate as f64;
    let padding_duration = 0.2; // 200ms padding

    // Phase 1: Transcribe all batches (no alignment yet)
    let mut transcription_batches: Vec<TranscribeBatch> = Vec::new();
    let mut final_text = String::new();

    info!("[Phase 1] Transcribing all batches...");
    for (i, batch) in batches.iter().enumerate() {
        // Include padding context
        let padded_start = f64::max(0.0, batch.start - padding_duration);
        let padded_end = batch.end + padding_duration;

        let start_sample = (padded_start * sample_rate) as usize;
        let end_sample = (padded_end * sample_rate) as usize;

        if start_sample < samples.len() {
            let end = std::cmp::min(end_sample, samples.len());
            let batch_pcm = samples[start_sample..end].to_vec();

            if !batch_pcm.is_empty() {
                let audio_ms = batch_pcm.len() as f64 / sample_rate * 1000.0;
                let (flat, n_frames) = audio::mel_spectrogram(&batch_pcm, &pipeline.audio_cfg);
                let mel_bins = pipeline.audio_cfg.mel_bins;
                let mel = Tensor::from_vec(flat, (mel_bins, n_frames), &device)?;

                info!(
                    "[Phase 1] Batch {}: duration {:.2}s",
                    i + 1,
                    audio_ms / 1000.0
                );

                let (text, timings) = pipeline.transcribe_mel(&mel, audio_ms)?;
                info!(
                    "[Phase 1] Batch {} Result: RT={:.2}x, text: {}",
                    i + 1,
                    (timings.encode_ms + timings.decode_ms) / audio_ms,
                    text
                );

                let text_str = text.trim();
                if !text_str.is_empty() {
                    final_text.push_str(text_str);
                    transcription_batches.push(TranscribeBatch {
                        pcm: batch_pcm,
                        start_time: padded_start,
                        end_time: padded_end,
                        text: text_str.to_string(),
                    });
                }
            }
        }
    }

    info!(
        "[Phase 1] Transcription complete. Total batches: {}",
        transcription_batches.len()
    );
    info!("[Phase 1] Final text: {}", final_text);

    // ==================== Phase 2: Alignment ====================
    info!(
        "[Phase 2] Loading ForcedAligner model from {} on {:?}",
        aligner_model, device
    );
    let aligner = ForcedAligner::load_with_device(Path::new(aligner_model), device.clone())?;

    let mut total_aligned_items = Vec::new();
    info!(
        "[Phase 2] Running alignment on {} batches...",
        transcription_batches.len()
    );

    for (i, batch) in transcription_batches.iter().enumerate() {
        info!("[Phase 2] Batch {} alignment...", i + 1);
        let items = aligner.align_samples(&batch.pcm, &batch.text, language)?;

        // Convert to absolute timestamps by adding the segment start time
        for mut item in items {
            item.start_time += batch.start_time;
            item.end_time += batch.start_time;
            total_aligned_items.push(item);
        }
    }

    info!(
        "[Phase 2] Alignment complete. Total aligned items: {}",
        total_aligned_items.len()
    );
    let mut items_info = Vec::new();
    for (_, item) in total_aligned_items.iter().enumerate() {
        items_info.push(format!(
            "[{:.3} - {:.3}] {}",
            item.start_time, item.end_time, item.text
        ));
    }
    info!("{}", items_info.join(""));

    // Generate SRT file
    let srt_content = generate_srt_from_align_items(&total_aligned_items);
    let srt_path = Path::new(audio_path).with_extension("srt");
    std::fs::write(&srt_path, &srt_content)?;
    info!("SRT file saved to: {}", srt_path.display());

    Ok(total_aligned_items)
}

/// Format time for SRT subtitle: HH:MM:SS,mmm
fn format_srt_time(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let ms = total_ms % 1000;
    let s = (total_ms / 1000) % 60;
    let m = (total_ms / 60000) % 60;
    let h = total_ms / 3600000;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
}

/// Sentence-ending punctuation (Chinese and English)
const SENTENCE_ENDINGS: &[char] = &['。', '！', '？', '！', '?', '.', '!', ','];

/// Generate SRT subtitle content from alignment items.
/// Groups consecutive words into sentences based on punctuation, length, and timing.
pub fn generate_srt_from_align_items(items: &[AlignItem]) -> String {
    if items.is_empty() {
        return String::new();
    }

    let mut srt_content = String::new();
    let mut subtitle_index = 1;

    let mut current_sentence = String::new();
    let mut sentence_start: Option<f64> = None;
    let mut sentence_end: Option<f64> = None;

    // Max constraints for a single subtitle entry
    const MAX_CHARS: usize = 60;
    const MAX_DURATION: f64 = 10.0;
    const MAX_GAP: f64 = 1.0;

    for item in items {
        let is_punctuation = item.text.chars().any(|c| SENTENCE_ENDINGS.contains(&c));
        let too_long =
            !current_sentence.is_empty() && current_sentence.chars().count() + item.text.chars().count() > MAX_CHARS;
        let too_much_time = if let Some(start) = sentence_start {
            item.end_time - start > MAX_DURATION
        } else {
            false
        };
        let big_gap = if let Some(end) = sentence_end {
            item.start_time - end > MAX_GAP
        } else {
            false
        };

        // If we need to split BEFORE adding this word
        if (too_long || too_much_time || big_gap) && !current_sentence.is_empty() {
            srt_content.push_str(&format!("{}\n", subtitle_index));
            srt_content.push_str(&format!(
                "{} --> {}\n",
                format_srt_time(sentence_start.unwrap_or(0.0)),
                format_srt_time(sentence_end.unwrap_or(0.0))
            ));
            srt_content.push_str(&format!("{}\n\n", current_sentence.trim()));
            subtitle_index += 1;

            current_sentence.clear();
            sentence_start = None;
            sentence_end = None;
        }

        // Add word to current segment
        if !current_sentence.is_empty() && !current_sentence.ends_with(' ') {
            current_sentence.push(' ');
        }
        current_sentence.push_str(&item.text);

        if sentence_start.is_none() {
            sentence_start = Some(item.start_time);
        }
        sentence_end = Some(item.end_time);

        // If we need to split AFTER adding this word (due to punctuation)
        if is_punctuation {
            srt_content.push_str(&format!("{}\n", subtitle_index));
            srt_content.push_str(&format!(
                "{} --> {}\n",
                format_srt_time(sentence_start.unwrap_or(0.0)),
                format_srt_time(sentence_end.unwrap_or(0.0))
            ));
            srt_content.push_str(&format!("{}\n\n", current_sentence.trim()));
            subtitle_index += 1;

            current_sentence.clear();
            sentence_start = None;
            sentence_end = None;
        }
    }

    // Handle any remaining content
    if !current_sentence.is_empty() {
        srt_content.push_str(&format!("{}\n", subtitle_index));
        srt_content.push_str(&format!(
            "{} --> {}\n",
            format_srt_time(sentence_start.unwrap_or(0.0)),
            format_srt_time(sentence_end.unwrap_or(0.0))
        ));
        srt_content.push_str(&format!("{}\n\n", current_sentence.trim()));
    }

    srt_content
}
