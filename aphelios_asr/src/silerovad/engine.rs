use std::{fs, path::Path};

use anyhow::{anyhow, Context, Result};
use aphelios_core::utils::common::get_available_ep;
use ndarray::{Array1, Array2, Array3};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use tracing::warn;

use crate::VadSegment;

/// VAD configuration parameters.
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    /// Speech probability threshold (0.0 - 1.0).
    pub threshold: f32,
    /// Minimum silence duration in milliseconds before ending a speech segment.
    pub min_silence_ms: f32,
    /// Minimum speech duration in milliseconds to be considered a valid segment.
    pub min_speech_ms: f32,
    /// Padding added to speech segment boundaries in milliseconds.
    pub speech_pad_ms: f32,
    /// Maximum gap in milliseconds between two segments to merge them.
    pub merge_gap_ms: f32,
    /// Sliding window size (number of frames) for smoothing.
    pub window_size: usize,
    /// Threshold for sliding window majority vote.
    pub window_threshold: f32,
}

impl VadConfig {
    pub fn new(
        threshold: f32,
        min_silence_ms: f32,
        min_speech_ms: f32,
        speech_pad_ms: f32,
        merge_gap_ms: f32,
    ) -> Self {
        Self {
            threshold,
            min_silence_ms,
            min_speech_ms,
            speech_pad_ms,
            merge_gap_ms,
            ..Self::default()
        }
    }
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_silence_ms: 500.0,
            min_speech_ms: 200.0,
            speech_pad_ms: 100.0,
            merge_gap_ms: 300.0,
            window_size: 10,
            window_threshold: 0.5,
        }
    }
}

impl VadConfig {
    /// Create a config optimized for SenseVoice-style usage (sample-index based).
    pub fn for_sensevoice(
        threshold: f32,
        min_silence_ms: f32,
        min_speech_ms: f32,
        speech_pad_ms: f32,
        merge_gap_ms: f32,
    ) -> Self {
        Self {
            threshold,
            min_silence_ms,
            min_speech_ms,
            speech_pad_ms,
            merge_gap_ms,
            window_size: 5,
            window_threshold: 0.4,
        }
    }

    /// Create a config optimized for general VAD pipeline usage (second-based).
    pub fn for_pipeline() -> Self {
        Self {
            threshold: 0.5,
            min_silence_ms: 500.0,
            min_speech_ms: 200.0,
            speech_pad_ms: 100.0,
            merge_gap_ms: 300.0,
            window_size: 10,
            window_threshold: 0.5,
        }
    }
}

/// Core Silero VAD engine: handles ONNX inference and segment detection on raw PCM.
pub struct SileroVadEngine {
    session: Session,
    state: Array3<f32>,
    sample_rate: usize,
    window_size: usize,
    config: VadConfig,
    // Pre-computed from config
    threshold: f32,
    min_silence_samples: usize,
    min_speech_samples: usize,
    speech_pad_samples: usize,
    merge_gap_samples: usize,
}

impl SileroVadEngine {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        sample_rate: usize,
        intra_threads: usize,
        config: VadConfig,
    ) -> Result<Self> {
        let window_size = match sample_rate {
            8000 => 256,
            16000 => 512,
            32000 => 1024,
            44100 | 48000 => 1536,
            other => {
                return Err(anyhow!(
                    "unsupported sample rate {} for Silero VAD (expected 8k/16k/32k/44.1k/48k)",
                    other
                ))
            }
        };

        let session = build_session_with_ort_cache(model_path.as_ref(), intra_threads)
            .with_context(|| {
                format!(
                    "prepare Silero VAD session for {}",
                    model_path.as_ref().display()
                )
            })?;

        let state = Array3::<f32>::zeros((2, 1, 128));
        let sanitized_config = sanitize_config(config);
        let threshold = sanitized_config.threshold;
        let min_silence_samples =
            ms_to_samples(sanitized_config.min_silence_ms, sample_rate).max(1);
        let min_speech_samples = ms_to_samples(sanitized_config.min_speech_ms, sample_rate).max(1);
        let speech_pad_samples = ms_to_samples(sanitized_config.speech_pad_ms, sample_rate);
        let merge_gap_samples = ms_to_samples(sanitized_config.merge_gap_ms, sample_rate);

        Ok(Self {
            session,
            state,
            sample_rate,
            window_size,
            config: sanitized_config,
            threshold,
            min_silence_samples,
            min_speech_samples,
            speech_pad_samples,
            merge_gap_samples,
        })
    }

    /// Reset internal RNN state.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    pub fn config(&self) -> VadConfig {
        self.config
    }

    /// Collect speech segments from raw PCM audio.
    /// Returns segments as sample indices.
    pub fn collect_segments(&mut self, pcm: &[f32]) -> Result<Vec<VadSegment>> {
        self.reset();
        if pcm.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Run inference to get raw probabilities
        let raw_probs = self.infer_probabilities(pcm)?;

        // 2. Apply sliding window smoothing
        let smoothed_probs = self.apply_smoothing(&raw_probs);

        // 3. State machine to extract segments
        let segments = self.detect_segments_from_probs(&smoothed_probs);

        Ok(segments)
    }

    /// Run inference on all audio chunks, return raw probabilities per frame.
    fn infer_probabilities(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        let mut probs = Vec::new();
        let mut frame = vec![0.0f32; self.window_size];
        let mut offset = 0;

        while offset < pcm.len() {
            let frame_end = (offset + self.window_size).min(pcm.len());
            frame.iter_mut().for_each(|v| *v = 0.0);
            let slice = &pcm[offset..frame_end];
            frame[..slice.len()].copy_from_slice(slice);

            let prob = self.predict_frame(&frame)?;
            probs.push(prob);

            offset += self.window_size;
        }

        Ok(probs)
    }

    /// Apply sliding window majority vote smoothing to reduce false positives.
    fn apply_smoothing(&self, raw_probs: &[f32]) -> Vec<f32> {
        let ws = self.config.window_size;
        if ws <= 1 {
            return raw_probs.to_vec();
        }

        let mut smoothed = Vec::with_capacity(raw_probs.len());

        for i in 0..raw_probs.len() {
            let start = i.saturating_sub(ws / 2);
            let end = (i + ws / 2).min(raw_probs.len() - 1);
            let window = &raw_probs[start..=end];

            let active_count = window.iter().filter(|&&p| p >= self.threshold).count();
            let ratio = active_count as f32 / window.len() as f32;

            if ratio >= self.config.window_threshold {
                smoothed.push(raw_probs[i]);
            } else {
                smoothed.push(0.0);
            }
        }
        smoothed
    }

    /// State machine: detect speech segments from smoothed probability sequence.
    fn detect_segments_from_probs(&self, probs: &[f32]) -> Vec<VadSegment> {
        let mut segments = Vec::new();
        let mut triggered = false;
        let mut speech_start_frame = 0;
        let mut silence_frames = 0;

        let max_silence_frames = self.min_silence_samples / self.window_size;
        let min_speech_frames = self.min_speech_samples / self.window_size;

        for (i, &prob) in probs.iter().enumerate() {
            if prob >= self.threshold {
                if !triggered {
                    triggered = true;
                    speech_start_frame = i;
                    silence_frames = 0;
                } else {
                    silence_frames = 0;
                }
            } else if triggered {
                silence_frames += 1;
                if silence_frames >= max_silence_frames {
                    triggered = false;
                    let end_frame = i.saturating_sub(silence_frames).saturating_add(1);
                    // Convert frames to sample indices with padding
                    let start_sample = speech_start_frame
                        .saturating_sub(self.speech_pad_samples / self.window_size)
                        .saturating_mul(self.window_size);
                    let end_sample = (end_frame
                        .saturating_add(self.speech_pad_samples / self.window_size)
                        .saturating_add(1))
                    .saturating_mul(self.window_size)
                    .min(pcm_len_from_probs(probs, self.window_size));

                    let duration = end_sample.saturating_sub(start_sample);
                    if duration >= self.min_speech_samples {
                        segments.push(VadSegment::new(start_sample as i64, end_sample as i64, 0.0));
                    }
                    silence_frames = 0;
                }
            }
        }

        // Handle trailing speech
        if triggered {
            let start_sample = speech_start_frame
                .saturating_sub(self.speech_pad_samples / self.window_size)
                .saturating_mul(self.window_size);
            let end_sample = pcm_len_from_probs(probs, self.window_size);
            let duration = end_sample.saturating_sub(start_sample);
            if duration >= self.min_speech_samples {
                segments.push(VadSegment::new(start_sample as i64, end_sample as i64, 0.0));
            }
        }

        // Merge segments
        if segments.len() > 1 {
            segments = self.merge_segment_pairs(segments);
        }

        segments
    }

    /// Merge pairs of segments within merge_gap distance.
    fn merge_segment_pairs(&self, segments: Vec<VadSegment>) -> Vec<VadSegment> {
        let mut merged: Vec<VadSegment> = Vec::with_capacity(segments.len());
        for seg in segments {
            if let Some(last) = merged.last_mut() {
                if seg.start <= last.end {
                    if seg.end > last.end {
                        last.end = seg.end;
                    }
                    continue;
                }
                let gap = seg.start.saturating_sub(last.end);
                if gap <= self.merge_gap_samples as i64 {
                    if seg.end > last.end {
                        last.end = seg.end;
                    }
                    continue;
                }
            }
            merged.push(seg);
        }
        merged
    }

    /// Run inference on a single frame, updating internal state.
    fn predict_frame(&mut self, samples: &[f32]) -> Result<f32> {
        let input = Array2::<f32>::from_shape_vec((1, samples.len()), samples.to_vec())?;
        let sample_rate = Array1::<i64>::from(vec![self.sample_rate as i64]);

        let input_value = Value::from_array(input).map_err(|e| anyhow!("ORT tensor error: {e}"))?;
        let sr_value =
            Value::from_array(sample_rate).map_err(|e| anyhow!("ORT tensor error: {e}"))?;
        let state_value =
            Value::from_array(self.state.clone()).map_err(|e| anyhow!("ORT tensor error: {e}"))?;

        let inputs = ort::inputs![
            "input" => input_value,
            "sr" => sr_value,
            "state" => state_value,
        ];
        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| anyhow!("ORT run error: {e}"))?;

        let (_prob_shape, prob_data) = outputs
            .get("output")
            .ok_or_else(|| anyhow!("Output 'output' not found"))?
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("ORT extract tensor error: {e}"))?;
        let probability = prob_data[0];

        let (state_shape, state_data) = outputs
            .get("stateN")
            .ok_or_else(|| anyhow!("Output 'stateN' not found"))?
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("ORT extract tensor error: {e}"))?;

        let state_array = Array3::<f32>::from_shape_vec(
            (
                state_shape[0] as usize,
                state_shape[1] as usize,
                state_shape[2] as usize,
            ),
            state_data.to_vec(),
        )?;
        self.state.assign(&state_array);

        Ok(probability)
    }
}

fn pcm_len_from_probs(probs: &[f32], window_size: usize) -> usize {
    probs.len().saturating_mul(window_size)
}

// ======================== Session building ========================

fn build_session_with_ort_cache(model_path: &Path, intra_threads: usize) -> Result<Session> {
    let ort_path = model_path.with_extension("ort");

    if ort_path.exists() {
        let session_attempt = Session::builder()
            .map_err(|e| anyhow!("ORT session builder error: {e}"))?
            .with_intra_threads(intra_threads)
            .map_err(|e| anyhow!("ORT intra threads error: {e}"))?
            .commit_from_file(&ort_path);

        match session_attempt {
            Ok(session) => return Ok(session),
            Err(err) => {
                warn!(
                    ort = %ort_path.display(),
                    model = %model_path.display(),
                    error = %err,
                    "failed to load cached VAD ORT graph, regenerating"
                );
                let _ = fs::remove_file(&ort_path);
            }
        }
    }

    let builder = Session::builder()
        .map_err(|e| anyhow!("ORT session builder error: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("ORT optimization level error: {e}"))?
        .with_intra_threads(intra_threads)
        .map_err(|e| anyhow!("ORT intra threads error: {e}"))?;

    if let Ok(builder_with_cache) = builder.with_optimized_model_path(&ort_path) {
        match builder_with_cache
            .with_execution_providers(get_available_ep())?
            .commit_from_file(model_path)
        {
            Ok(session) => return Ok(session),
            Err(err) => {
                warn!(
                    ort = %ort_path.display(),
                    model = %model_path.display(),
                    error = %err,
                    "failed to rebuild VAD session with ORT cache, retrying without cache"
                );
                let _ = fs::remove_file(&ort_path);
            }
        }
    }

    let fallback_builder = Session::builder()
        .map_err(|e| anyhow!("ORT session builder error: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("ORT optimization level error: {e}"))?
        .with_intra_threads(intra_threads)
        .map_err(|e| anyhow!("ORT intra threads error: {e}"))?;

    let model_bytes = fs::read(model_path)
        .with_context(|| format!("read Silero VAD model {}", model_path.display()))?;
    fallback_builder
        .commit_from_memory(&model_bytes)
        .map_err(|e| anyhow!("ORT load model error: {e}"))
}

// ======================== Config sanitization ========================

fn sanitize_config(config: VadConfig) -> VadConfig {
    let defaults = VadConfig::default();
    VadConfig {
        threshold: sanitize_threshold(config.threshold, defaults.threshold),
        min_silence_ms: sanitize_duration(config.min_silence_ms, defaults.min_silence_ms),
        min_speech_ms: sanitize_duration(config.min_speech_ms, defaults.min_silence_ms),
        speech_pad_ms: sanitize_duration(config.speech_pad_ms, defaults.speech_pad_ms),
        merge_gap_ms: sanitize_duration(config.merge_gap_ms, defaults.merge_gap_ms),
        window_size: if config.window_size > 0 {
            config.window_size
        } else {
            defaults.window_size
        },
        window_threshold: sanitize_threshold(config.window_threshold, defaults.window_threshold),
    }
}

fn sanitize_threshold(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        fallback
    }
}

fn sanitize_duration(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        fallback
    }
}

fn ms_to_samples(ms: f32, sample_rate: usize) -> usize {
    if ms <= 0.0 {
        return 0;
    }
    ((sample_rate as f32) * (ms / 1000.0)).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_config_default() {
        let cfg = VadConfig::default();
        assert_eq!(cfg.threshold, 0.5);
        assert_eq!(cfg.min_silence_ms, 500.0);
        assert_eq!(cfg.min_speech_ms, 200.0);
    }

    #[test]
    fn test_ms_to_samples() {
        assert_eq!(ms_to_samples(1000.0, 16000), 16000);
        assert_eq!(ms_to_samples(500.0, 16000), 8000);
        assert_eq!(ms_to_samples(0.0, 16000), 0);
    }
}
