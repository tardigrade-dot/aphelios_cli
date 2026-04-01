use serde::Serialize;

pub mod audio;
pub mod frontend;
pub mod sensevoice;
pub mod tokenizer;
pub mod vad;

pub fn language_id_from_code(code: &str) -> i32 {
    // Python mapping: {"auto":0,"zh":3,"en":4,"yue":7,"ja":11,"ko":12,"nospeech":13}
    match code.to_lowercase().as_str() {
        "auto" => 0,
        "zh" => 3,
        "en" => 4,
        "yue" => 7,
        "ja" => 11,
        "ko" => 12,
        "nospeech" => 13,
        _ => 0,
    }
}

pub struct SenseVoiceConfig {
    pub num_threads: usize,

    pub language: String,

    pub use_itn: bool,

    pub vad_int8: bool,

    pub vad_threshold: f32,

    pub vad_min_speech_ms: f32,

    pub vad_min_silence_ms: f32,

    pub vad_speech_pad_ms: f32,

    pub vad_merge_gap_ms: f32,

    pub channels: usize,
}
impl SenseVoiceConfig {
    pub fn default() -> Self {
        Self {
            num_threads: 1,
            language: "en".to_string(),
            use_itn: false,
            vad_int8: false,
            vad_threshold: 0.5,
            vad_min_speech_ms: 400.0,
            vad_min_silence_ms: 200.0,
            vad_speech_pad_ms: 120.0,
            vad_merge_gap_ms: 1200.0,
            channels: 1,
        }
    }
}

#[derive(Serialize, Clone)]
pub struct WordTimestamp {
    pub word: String,
    pub start_sec: f32,
    pub end_sec: f32,
}

pub fn extract_tags(text: &str) -> (String, Vec<String>) {
    let chars: Vec<char> = text.chars().collect();
    let mut tags = Vec::new();
    let mut clean = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '<' {
            let mut j = i + 1;
            let mut closed = false;
            while j < chars.len() {
                if chars[j] == '>' {
                    closed = true;
                    break;
                }
                j += 1;
            }
            if closed {
                let tag_content: String = chars[i + 1..j].iter().collect();
                let tag = tag_content.trim();
                if !tag.is_empty() {
                    tags.push(tag.to_string());
                }
                i = j + 1;
                continue;
            }
        }
        clean.push(chars[i]);
        i += 1;
    }
    let cleaned = clean.trim().to_string();
    (cleaned, tags)
}

use anyhow::Result;
use aphelios_core::init_logging;
use ndarray::Axis;
use std::{path::PathBuf, time::Instant};
use tracing::{debug, error, info, warn};

use crate::sensevoice::{
    audio::{decode_audio_multi, downmix_to_mono, resample_channels},
    frontend::{FeaturePipeline, FrontendConfig},
    sensevoice::SensevoiceEncoder,
    tokenizer::TokenDecoder,
    vad::{SileroVad, VadConfig, VadSegment},
};

pub struct SenseVoiceResult {
    pub text: String,
    pub timestamp: Vec<WordTimestamp>,
    pub words: Vec<String>,
}

pub fn sensevoice_asr(
    sensevoice_model_path: &str,
    audio_path: &str,
    sense_voice_config: SenseVoiceConfig,
) -> Result<SenseVoiceResult> {
    init_logging();

    let input_audio = PathBuf::from(audio_path);

    let encoder_path = PathBuf::from(sensevoice_model_path).join("model.int8.onnx");
    let tokens_path = PathBuf::from(sensevoice_model_path).join("tokens.txt");

    if !encoder_path.exists() || !tokens_path.exists() {
        error!("model/resource files missing in snapshot. Please check repository contents.");
    }

    let fe_cfg = FrontendConfig::default();
    let target_sample_rate = fe_cfg.sample_rate as u32;
    let vad_config = VadConfig::new(
        sense_voice_config.vad_threshold,
        sense_voice_config.vad_min_silence_ms,
        sense_voice_config.vad_min_speech_ms,
        sense_voice_config.vad_speech_pad_ms,
        sense_voice_config.vad_merge_gap_ms,
    );
    let vad_model_path = PathBuf::from(sensevoice_model_path).join("vad-model.onnx");
    let mut silero_vad = SileroVad::new(
        &vad_model_path,
        fe_cfg.sample_rate,
        sense_voice_config.num_threads,
        vad_config,
    )?;

    let vad_config = silero_vad.config();

    debug!(
        use_int8 = sense_voice_config.vad_int8,
        vad_model = %vad_model_path.display(),
        threshold = format!("{:.3}", vad_config.threshold),
        min_speech_ms = format!("{:.1}", vad_config.min_speech_ms),
        min_silence_ms = format!("{:.1}", vad_config.min_silence_ms),
        speech_pad_ms = format!("{:.1}", vad_config.speech_pad_ms),
        "Silero VAD model initialized"
    );

    // 4) ORT Session for encoder + tokenizer
    let mut encoder = SensevoiceEncoder::new(&encoder_path, sense_voice_config.num_threads)?;
    let decoder = TokenDecoder::new(&tokens_path)?;
    let lang_id = language_id_from_code(&sense_voice_config.language);
    // 2) Audio decode and downmix to mono
    let t0 = Instant::now();
    let (decoded_sample_rate, total_channels, samples_per_channel) =
        decode_audio_multi(&input_audio)?;

    let mut ch = downmix_to_mono(samples_per_channel);
    if ch.is_empty() {
        anyhow::bail!("no audio samples available for transcription");
    }

    let audio_duration_sec = ch.len() as f32 / decoded_sample_rate as f32;
    if decoded_sample_rate != target_sample_rate {
        debug!(
            "resampling audio from {} Hz to {} Hz",
            decoded_sample_rate, target_sample_rate
        );
        let resampled = resample_channels(vec![ch], decoded_sample_rate, target_sample_rate)?;
        ch = resampled.into_iter().next().unwrap();
    }
    debug!(
        "decoded audio: {} Hz, {} ch (downmixed to 1), duration ~{:.2}s",
        decoded_sample_rate, total_channels, audio_duration_sec
    );

    // 3) Frontend: fbank + LFR + CMVN (Kaldi-like defaults)
    let mut fe = FeaturePipeline::new(fe_cfg);

    let mut consolidated_text = String::new();
    let mut consolidated_timestamps = Vec::new();
    let mut consolidated_words = Vec::new();

    let mut segments = silero_vad.collect_segments(&ch)?;
    if segments.is_empty() {
        warn!("no speech detected by VAD, falling back to full audio");
        segments.push(VadSegment {
            start: 0,
            end: ch.len(),
        });
    }

    for seg in segments {
        let start = seg.start.min(ch.len());
        let end = seg.end.min(ch.len());
        if end <= start {
            continue;
        }
        let segment_samples = &ch[start..end];
        if segment_samples.is_empty() {
            continue;
        }

        let feats = match fe.compute_features(segment_samples, target_sample_rate) {
            Ok(f) => f,
            Err(e) => {
                warn!(start = start, end = end, error = %e, "feature extraction failed for segment");
                continue;
            }
        };
        if feats.is_empty() {
            warn!(
                start = start,
                end = end,
                "empty feature matrix for segment, skipping"
            );
            continue;
        }
        let feats = feats.insert_axis(Axis(0));
        let (raw_text, word_timestamps) = encoder.run_and_decode_with_timestamps(
            &decoder,
            feats.view(),
            lang_id,
            sense_voice_config.use_itn,
        )?;

        let (clean_text, _tags) = extract_tags(&raw_text);
        if !consolidated_text.is_empty() && !clean_text.is_empty() {
            consolidated_text.push(' ');
        }
        consolidated_text.push_str(&clean_text);

        let offset_sec = start as f32 / target_sample_rate as f32;
        for wt in word_timestamps {
            consolidated_words.push(wt.token.clone());
            consolidated_timestamps.push(WordTimestamp {
                word: wt.token,
                start_sec: wt.start_sec + offset_sec,
                end_sec: wt.end_sec + offset_sec,
            });
        }
    }

    let sensevoice_result = SenseVoiceResult {
        text: consolidated_text,
        timestamp: consolidated_timestamps,
        words: consolidated_words,
    };
    let elapsed = t0.elapsed();
    let rtf = elapsed.as_secs_f32() / audio_duration_sec.max(1e-6);

    info!("time: {:?}, rtf: {:.3}", elapsed, rtf);
    Ok(sensevoice_result)
}
