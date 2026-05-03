use std::path::Path;

use anyhow::Result;
use aphelios_core::{
    measure_time,
    utils::common::{get_device, truncate_by_chars},
};
use candle_core::Tensor;
use tracing::info;

use crate::{
    qwenasr::{
        aligner::{tokenize_for_alignment, AlignItem, ForcedAligner},
        transcribe::Pipeline,
    },
    silerovad::VadProcessor,
    whisper::generate_vad,
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
    let mut pipeline =
        transcribe::Pipeline::load_with_device(Path::new(qwen3asr_model))
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
        ForcedAligner::load_with_device(Path::new(aligner_model)).unwrap_or_else(|e| {
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

struct AlignedBatch {
    text: String,
    items: Vec<AlignItem>,
    end_time: f64,
}

pub async fn qwen3asr_with_vad(
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

    #[cfg(feature = "profiling")]
    let vad_start = std::time::Instant::now();

    let segments = vad.process_from_file(audio_path)?;

    #[cfg(feature = "profiling")]
    {
        let vad_duration = vad_start.elapsed();
        info!(
            "[profiling] VAD duration: {:.2}s",
            vad_duration.as_secs_f64()
        );
        let output_path = Path::new(audio_path)
            .with_file_name(Path::new(audio_path).file_stem().unwrap().to_str().unwrap())
            .with_extension("vad.srt")
            .to_str()
            .unwrap()
            .to_string();
        let _ = generate_vad(&segments, &output_path).await;
        info!("[profiling] Generated VAD SRT at {}", output_path);
    }
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
    let mut pipeline = Pipeline::load_with_device(Path::new(qwen3asr_model))?;

    // Load entire audio as float samples
    let samples = audio::load_wav(Path::new(audio_path), &pipeline.audio_cfg)?;
    let sample_rate = pipeline.audio_cfg.sample_rate as f64;
    let padding_duration = 0.2; // 200ms padding

    // Phase 1: Transcribe all batches (no alignment yet)
    let mut transcription_batches: Vec<TranscribeBatch> = Vec::new();
    let mut final_text = String::new();

    let batch_size = batches.len();
    info!("[Phase 1] Transcribing all batches start ...");
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
                    "[Phase 1] Batch {}/{}: duration {:.2}s",
                    i + 1,
                    batch_size,
                    audio_ms / 1000.0
                );

                let desc = format!("{}, {},", i + 1, batch_size,);
                let (text, timings) = measure_time!(desc, pipeline.transcribe_mel(&mel, audio_ms)?);
                let preview = truncate_by_chars(&text, 20);
                info!(
                    "[Phase 1] Batch {}/{} Result: RT={:.2}x, total length {} text: {}...",
                    i + 1,
                    batch_size,
                    (timings.encode_ms + timings.decode_ms) / audio_ms,
                    &text.len(),
                    preview,
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

    // ==================== Phase 2: Alignment ====================
    info!(
        "[Phase 2] Loading ForcedAligner model from {} on {:?}",
        aligner_model, device
    );
    let aligner = ForcedAligner::load_with_device(Path::new(aligner_model))?;

    let mut total_aligned_items = Vec::new();
    let mut aligned_batches = Vec::new();
    info!(
        "[Phase 2] Running alignment on {} batches...",
        transcription_batches.len()
    );

    for (i, batch) in transcription_batches.iter().enumerate() {
        info!("[Phase 2] Batch {}/{} alignment...", i + 1, batch_size);

        let mut items = measure_time!(aligner.align_samples(&batch.pcm, &batch.text, language)?);

        // Convert to absolute timestamps by adding the segment start time
        for item in &mut items {
            item.start_time += batch.start_time;
            item.end_time += batch.start_time;
        }
        total_aligned_items.extend(items.iter().cloned());
        aligned_batches.push(AlignedBatch {
            text: batch.text.clone(),
            items,
            end_time: batch.end_time,
        });
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

    // Generate SRT file
    let srt_content = generate_srt_from_aligned_batches(&aligned_batches);
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
const SOFT_BREAK_PUNCTUATION: &[char] = &[',', '，', '；', ';', '：', ':', '、'];

#[derive(Clone, Debug)]
struct SubtitleToken {
    text: String,
    start_time: f64,
    end_time: f64,
    segment_boundary_after: bool,
}

#[derive(Clone, Debug)]
struct SubtitleEntry {
    text: String,
    start_time: f64,
    end_time: f64,
}

fn is_cjk_text(text: &str) -> bool {
    !text.is_empty()
        && text.chars().all(|ch| {
            let code = ch as u32;
            (0x4E00..=0x9FFF).contains(&code)
                || (0x3400..=0x4DBF).contains(&code)
                || (0x20000..=0x2A6DF).contains(&code)
                || (0x2A700..=0x2B73F).contains(&code)
                || (0x2B740..=0x2B81F).contains(&code)
                || (0x2B820..=0x2CEAF).contains(&code)
                || (0xF900..=0xFAFF).contains(&code)
        })
}

fn needs_space_between(prev: &str, next: &str) -> bool {
    !(is_cjk_text(prev) && is_cjk_text(next))
}

fn is_token_char(ch: char) -> bool {
    is_cjk_text(&ch.to_string()) || ch == '\'' || ch.is_alphanumeric()
}

fn visible_char_count(text: &str) -> usize {
    text.chars().filter(|ch| !ch.is_whitespace()).count()
}

fn is_sentence_break_text(text: &str) -> bool {
    text.chars()
        .rev()
        .any(|ch| !ch.is_whitespace() && SENTENCE_ENDINGS.contains(&ch))
}

fn is_soft_break_text(text: &str) -> bool {
    text.chars()
        .rev()
        .any(|ch| !ch.is_whitespace() && SOFT_BREAK_PUNCTUATION.contains(&ch))
}

fn restore_tokens_from_text(
    text: &str,
    items: &[AlignItem],
    segment_boundary_after_last: bool,
) -> Vec<SubtitleToken> {
    if items.is_empty() {
        return Vec::new();
    }

    let expected_tokens = tokenize_for_alignment(text);
    if expected_tokens.len() != items.len()
        || expected_tokens
            .iter()
            .zip(items.iter())
            .any(|(expected, item)| expected != &item.text)
    {
        return items
            .iter()
            .enumerate()
            .map(|(idx, item)| SubtitleToken {
                text: item.text.clone(),
                start_time: item.start_time,
                end_time: item.end_time,
                segment_boundary_after: segment_boundary_after_last && idx + 1 == items.len(),
            })
            .collect();
    }

    let chars: Vec<char> = text.chars().collect();
    let mut cursor = 0usize;
    let mut restored = Vec::with_capacity(items.len());

    for (idx, item) in items.iter().enumerate() {
        let token_start = cursor;
        let target = &item.text;
        let mut cleaned = String::new();

        while cursor < chars.len() {
            let ch = chars[cursor];
            if is_cjk_text(&ch.to_string()) || ch == '\'' || ch.is_alphanumeric() {
                cleaned.push(ch);
            }
            cursor += 1;

            if cleaned == *target {
                break;
            }
        }

        while cursor < chars.len() && !is_token_char(chars[cursor]) {
            cursor += 1;
        }

        let fragment: String = chars[token_start..cursor].iter().collect();
        restored.push(SubtitleToken {
            text: fragment,
            start_time: item.start_time,
            end_time: item.end_time,
            segment_boundary_after: segment_boundary_after_last && idx + 1 == items.len(),
        });
    }

    if cursor < chars.len() {
        let tail: String = chars[cursor..].iter().collect();
        if let Some(last) = restored.last_mut() {
            last.text.push_str(&tail);
        }
    }

    restored
}

fn split_subtitle_entries(tokens: &[SubtitleToken]) -> Vec<SubtitleEntry> {
    if tokens.is_empty() {
        return Vec::new();
    }

    const MIN_CHARS: usize = 40;
    const MAX_CHARS: usize = 100;

    let finalize_range =
        |entries: &mut Vec<SubtitleEntry>, tokens: &[SubtitleToken], start: usize, end: usize| {
            if start > end || start >= tokens.len() {
                return;
            }
            let text: String = tokens[start..=end]
                .iter()
                .map(|token| token.text.as_str())
                .collect();
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return;
            }
            entries.push(SubtitleEntry {
                text: trimmed.to_string(),
                start_time: tokens[start].start_time,
                end_time: tokens[end].end_time,
            });
        };

    let mut entries = Vec::new();
    let mut start_idx = 0usize;

    while start_idx < tokens.len() {
        let mut current_len = 0usize;
        let mut last_sentence_break: Option<usize> = None;
        let mut last_soft_break: Option<usize> = None;
        let mut last_segment_break: Option<usize> = None;
        let mut end_idx = start_idx;
        let mut did_split = false;

        while end_idx < tokens.len() {
            let token = &tokens[end_idx];
            current_len += visible_char_count(&token.text);

            if current_len >= MIN_CHARS {
                if is_sentence_break_text(&token.text) {
                    last_sentence_break = Some(end_idx);
                }
                if is_soft_break_text(&token.text) {
                    last_soft_break = Some(end_idx);
                }
                if token.segment_boundary_after {
                    last_segment_break = Some(end_idx);
                }

                if is_sentence_break_text(&token.text) || token.segment_boundary_after {
                    finalize_range(&mut entries, tokens, start_idx, end_idx);
                    start_idx = end_idx + 1;
                    did_split = true;
                    break;
                }
            }

            if current_len > MAX_CHARS {
                let cut_idx = last_sentence_break
                    .or(last_segment_break)
                    .or(last_soft_break)
                    .unwrap_or(end_idx);
                finalize_range(&mut entries, tokens, start_idx, cut_idx);
                start_idx = cut_idx + 1;
                did_split = true;
                break;
            }

            end_idx += 1;
        }

        if !did_split {
            finalize_range(&mut entries, tokens, start_idx, tokens.len() - 1);
            break;
        }
    }

    entries
}

fn generate_srt_from_aligned_batches(batches: &[AlignedBatch]) -> String {
    let mut tokens = Vec::new();
    for batch in batches {
        let mut batch_tokens = restore_tokens_from_text(&batch.text, &batch.items, true);
        if batch_tokens.is_empty() && !batch.items.is_empty() {
            batch_tokens.push(SubtitleToken {
                text: batch.text.clone(),
                start_time: batch.items[0].start_time,
                end_time: batch.end_time,
                segment_boundary_after: true,
            });
        }
        tokens.extend(batch_tokens);
    }

    let entries = split_subtitle_entries(&tokens);
    generate_srt_from_entries(&entries)
}

fn generate_srt_from_entries(entries: &[SubtitleEntry]) -> String {
    let mut srt_content = String::new();
    for (idx, entry) in entries.iter().enumerate() {
        let next_start = entries.get(idx + 1).map(|entry| entry.start_time);
        let end_time = next_start
            .map(|next_start| entry.end_time.max((entry.end_time + 0.25).min(next_start)))
            .unwrap_or(entry.end_time + 0.25);

        srt_content.push_str(&format!("{}\n", idx + 1));
        srt_content.push_str(&format!(
            "{} --> {}\n",
            format_srt_time(entry.start_time),
            format_srt_time(end_time)
        ));
        srt_content.push_str(&format!("{}\n\n", entry.text));
    }
    srt_content
}

fn finalize_subtitle(
    srt_content: &mut String,
    subtitle_index: &mut usize,
    current_sentence: &str,
    sentence_start: Option<f64>,
    sentence_end: Option<f64>,
    next_item_start: Option<f64>,
) {
    if current_sentence.trim().is_empty() {
        return;
    }

    const MAX_TAIL_HOLD: f64 = 0.25;

    let start_time = sentence_start.unwrap_or(0.0);
    let raw_end_time = sentence_end.unwrap_or(start_time);
    let end_time = next_item_start
        .map(|next_start| raw_end_time.max((raw_end_time + MAX_TAIL_HOLD).min(next_start)))
        .unwrap_or(raw_end_time + MAX_TAIL_HOLD);

    srt_content.push_str(&format!("{}\n", *subtitle_index));
    srt_content.push_str(&format!(
        "{} --> {}\n",
        format_srt_time(start_time),
        format_srt_time(end_time)
    ));
    srt_content.push_str(&format!("{}\n\n", current_sentence.trim()));
    *subtitle_index += 1;
}

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
    let mut previous_item_text: Option<&str> = None;

    // Max constraints for a single subtitle entry
    const MAX_CHARS: usize = 60;
    const MAX_DURATION: f64 = 10.0;
    const MAX_GAP: f64 = 1.0;

    for (idx, item) in items.iter().enumerate() {
        let next_item_start = items.get(idx + 1).map(|next| next.start_time);
        let is_punctuation = item.text.chars().any(|c| SENTENCE_ENDINGS.contains(&c));
        let too_long = !current_sentence.is_empty()
            && current_sentence.chars().count() + item.text.chars().count() > MAX_CHARS;
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
            finalize_subtitle(
                &mut srt_content,
                &mut subtitle_index,
                &current_sentence,
                sentence_start,
                sentence_end,
                Some(item.start_time),
            );
            current_sentence.clear();
            sentence_start = None;
            sentence_end = None;
            previous_item_text = None;
        }

        // Add word to current segment
        if let Some(prev_text) = previous_item_text {
            if !current_sentence.is_empty() && needs_space_between(prev_text, &item.text) {
                current_sentence.push(' ');
            }
        } else if !current_sentence.is_empty() && !current_sentence.ends_with(' ') {
            current_sentence.push(' ');
        }
        current_sentence.push_str(&item.text);

        if sentence_start.is_none() {
            sentence_start = Some(item.start_time);
        }
        sentence_end = Some(item.end_time);
        previous_item_text = Some(&item.text);

        // If we need to split AFTER adding this word (due to punctuation)
        if is_punctuation {
            finalize_subtitle(
                &mut srt_content,
                &mut subtitle_index,
                &current_sentence,
                sentence_start,
                sentence_end,
                next_item_start,
            );
            current_sentence.clear();
            sentence_start = None;
            sentence_end = None;
            previous_item_text = None;
        }
    }

    // Handle any remaining content
    if !current_sentence.is_empty() {
        finalize_subtitle(
            &mut srt_content,
            &mut subtitle_index,
            &current_sentence,
            sentence_start,
            sentence_end,
            None,
        );
    }

    srt_content
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_srt_keeps_cjk_compact() {
        let items = vec![
            AlignItem {
                text: "你".into(),
                start_time: 0.0,
                end_time: 0.2,
            },
            AlignItem {
                text: "好".into(),
                start_time: 0.2,
                end_time: 0.4,
            },
            AlignItem {
                text: "世".into(),
                start_time: 0.4,
                end_time: 0.6,
            },
            AlignItem {
                text: "界".into(),
                start_time: 0.6,
                end_time: 0.8,
            },
        ];

        let srt = generate_srt_from_align_items(&items);
        assert!(srt.contains("你好世界"));
        assert!(!srt.contains("你 好 世 界"));
    }

    #[test]
    fn generate_srt_restores_punctuation_and_splits_on_it() {
        let batch = AlignedBatch {
            text: "你好，世界。再见，朋友。".into(),
            items: vec![
                AlignItem {
                    text: "你".into(),
                    start_time: 0.0,
                    end_time: 0.2,
                },
                AlignItem {
                    text: "好".into(),
                    start_time: 0.2,
                    end_time: 0.4,
                },
                AlignItem {
                    text: "世".into(),
                    start_time: 0.4,
                    end_time: 0.6,
                },
                AlignItem {
                    text: "界".into(),
                    start_time: 0.6,
                    end_time: 0.8,
                },
                AlignItem {
                    text: "再".into(),
                    start_time: 0.8,
                    end_time: 1.0,
                },
                AlignItem {
                    text: "见".into(),
                    start_time: 1.0,
                    end_time: 1.2,
                },
                AlignItem {
                    text: "朋".into(),
                    start_time: 1.2,
                    end_time: 1.4,
                },
                AlignItem {
                    text: "友".into(),
                    start_time: 1.4,
                    end_time: 1.6,
                },
            ],
            end_time: 1.6,
        };

        let tokens = restore_tokens_from_text(&batch.text, &batch.items, true);
        let text: String = tokens.iter().map(|token| token.text.as_str()).collect();
        assert_eq!(text, "你好，世界。再见，朋友。");
    }

    #[test]
    fn generate_srt_holds_end_until_next_item() {
        let items = vec![
            AlignItem {
                text: "hello".into(),
                start_time: 0.0,
                end_time: 0.5,
            },
            AlignItem {
                text: "world".into(),
                start_time: 10.2,
                end_time: 10.6,
            },
        ];

        let srt = generate_srt_from_align_items(&items);
        assert!(srt.contains("00:00:00,000 --> 00:00:00,750"));
        assert!(srt.contains("00:00:10,200 --> 00:00:10,850"));
    }

    #[test]
    fn split_subtitle_entries_prefers_punctuation_within_target_window() {
        let mut tokens = Vec::new();
        let mut current_time = 0.0;
        let chunks = vec![
            "甲".repeat(40),
            "乙".repeat(45) + "。",
            "丙".repeat(35),
            "丁".repeat(50) + "。",
        ];

        for (idx, chunk) in chunks.into_iter().enumerate() {
            let duration = 0.5;
            tokens.push(SubtitleToken {
                text: chunk,
                start_time: current_time,
                end_time: current_time + duration,
                segment_boundary_after: idx == 3,
            });
            current_time += duration;
        }

        let entries = split_subtitle_entries(&tokens);
        assert_eq!(entries.len(), 2);
        assert_eq!(visible_char_count(&entries[0].text), 86);
        assert!(entries[0].text.ends_with('。'));
        assert_eq!(visible_char_count(&entries[1].text), 86);
        assert!(entries[1].text.ends_with('。'));
    }
}
