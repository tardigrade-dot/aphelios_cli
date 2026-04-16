//! Full VAD detection pipeline: audio loading → resampling → inference → segments.

use anyhow::Result;
use aphelios_core::{
    audio::{MonoBuffer, ResampleQuality},
    AudioLoader, Resampler,
};
use tracing::info;

use crate::{AudioBatch, VadSegment};

use super::engine::{SileroVadEngine, VadConfig};

/// VAD detector with full audio file processing pipeline.
pub struct VadDetector {
    engine: SileroVadEngine,
    sample_rate: usize,
}

impl VadDetector {
    /// Create a new VAD detector.
    pub fn new(model_path: &str, config: VadConfig) -> Result<Self> {
        // Silero VAD works best at 16kHz
        let sample_rate = 16000usize;

        info!("Loading VAD model from: {}", model_path);
        let engine = SileroVadEngine::new(model_path, sample_rate, 1, config)?;
        info!("VAD model loaded successfully");

        Ok(Self {
            engine,
            sample_rate,
        })
    }

    /// Create a new VAD detector with default config.
    pub fn new_default(model_path: &str) -> Result<Self> {
        Self::new(model_path, VadConfig::default())
    }

    /// Detect speech activity from an audio file.
    /// Returns segments in seconds. Audio is loaded at 16kHz mono.
    pub fn detect_from_file(&mut self, audio_path: &str) -> Result<Vec<VadSegment>> {
        // AudioLoader now always returns 16kHz mono AudioBuffer
        let audio = AudioLoader::new().load(audio_path)?;
        let mono = audio.into_mono();

        let samples_segments = self.engine.collect_segments(mono.samples.as_slice())?;
        // Convert sample indices to milliseconds
        Ok(samples_segments
            .into_iter()
            .map(|s| VadSegment {
                start: (s.start as i64 * 1000) / self.sample_rate as i64,
                end: (s.end as i64 * 1000) / self.sample_rate as i64,
                avg_prob: 0.0, // Not tracked in this mode
            })
            .collect())
    }

    /// Detect speech activity from raw PCM samples.
    /// Returns segments in seconds.
    pub fn detect(&mut self, samples: &[f32]) -> Result<Vec<VadSegment>> {
        let samples_segments = self.engine.collect_segments(samples)?;
        Ok(samples_segments
            .into_iter()
            .map(|s| VadSegment {
                start: (s.start as i64 * 1000) / self.sample_rate as i64,
                end: (s.end as i64 * 1000) / self.sample_rate as i64,
                avg_prob: 0.0,
            })
            .collect())
    }

    /// Detect from raw MonoBuffer.
    pub fn detect_mono(&mut self, audio: MonoBuffer) -> Result<Vec<VadSegment>> {
        let audio = if audio.sample_rate != self.sample_rate as u32 {
            let resampler = Resampler::new().with_quality(ResampleQuality::Fast);
            resampler.resample_mono(&audio, self.sample_rate as u32)?
        } else {
            audio
        };
        self.detect(&audio.samples)
    }

    pub fn config(&self) -> VadConfig {
        self.engine.config()
    }
}

/// VAD processor for the qwenasr pipeline: supports segment aggregation into batches.
pub struct VadProcessor {
    engine: SileroVadEngine,
    sample_rate: usize,
}

impl VadProcessor {
    pub fn new_default(model_dir: &str) -> Result<Self> {
        let model_path = format!("{}/model.onnx", model_dir);
        let config = VadConfig::for_pipeline();
        Self::new(&model_path, config)
    }

    pub fn new(model_path: &str, config: VadConfig) -> Result<Self> {
        let sample_rate = 16000usize;
        info!("Loading VAD model from: {}", model_path);
        let engine = SileroVadEngine::new(model_path, sample_rate, 1, config)?;
        info!("VAD model loaded successfully");
        Ok(Self {
            engine,
            sample_rate,
        })
    }

    /// Process audio file and return VAD segments (in milliseconds).
    /// Audio is loaded at 16kHz mono - no additional resampling needed.
    pub fn process_from_file(&mut self, audio_path: &str) -> Result<Vec<VadSegment>> {
        // AudioLoader always returns 16kHz mono, no resampling needed
        let audio = AudioLoader::new().load(audio_path)?;
        let mono = audio.into_mono();

        let segments = self.engine.collect_segments(&mono.samples)?;
        // Convert sample indices to milliseconds
        let ms_segments: Vec<VadSegment> = segments
            .into_iter()
            .map(|s| VadSegment {
                start: (s.start as i64 * 1000) / self.sample_rate as i64,
                end: (s.end as i64 * 1000) / self.sample_rate as i64,
                avg_prob: 0.0,
            })
            .collect();

        Ok(ms_segments)
    }

    /// Process raw samples and return VAD segments (in milliseconds).
    pub fn process(&mut self, samples: &[f32]) -> Result<Vec<VadSegment>> {
        let segments = self.engine.collect_segments(samples)?;
        let ms_segments: Vec<VadSegment> = segments
            .into_iter()
            .map(|s| VadSegment {
                start: (s.start as i64 * 1000) / self.sample_rate as i64,
                end: (s.end as i64 * 1000) / self.sample_rate as i64,
                avg_prob: 0.0,
            })
            .collect();
        Ok(ms_segments)
    }

    /// Aggregate VAD segments into batches of approximately `target_duration` seconds.
    /// Segments are expected to be in milliseconds.
    pub fn aggregate_segments(
        &self,
        segments: &[VadSegment],
        target_duration: f64,
        max_silence_merge: f64,
    ) -> Vec<AudioBatch> {
        if segments.is_empty() {
            return Vec::new();
        }

        // Convert to milliseconds for internal calculation
        let target_duration_ms = (target_duration * 1000.0) as i64;
        let max_silence_merge_ms = (max_silence_merge * 1000.0) as i64;

        let mut batches = Vec::new();
        let mut current_batch_start = segments[0].start;
        let mut last_segment_end = segments[0].end;
        let mut count = 0;

        for i in 1..segments.len() {
            let seg = &segments[i];
            let gap = seg.start - last_segment_end;
            let potential_duration = seg.end - current_batch_start;

            if potential_duration <= target_duration_ms || gap < max_silence_merge_ms {
                last_segment_end = seg.end;
                count += 1;
            } else {
                batches.push(AudioBatch {
                    start: current_batch_start as f64 / 1000.0,
                    end: last_segment_end as f64 / 1000.0,
                    duration: (last_segment_end - current_batch_start) as f64 / 1000.0,
                    segments_count: count + 1,
                });

                current_batch_start = segments[i].start;
                last_segment_end = segments[i].end;
                count = 0;
            }
        }

        // Last batch
        batches.push(AudioBatch {
            start: current_batch_start as f64 / 1000.0,
            end: last_segment_end as f64 / 1000.0,
            duration: (last_segment_end - current_batch_start) as f64 / 1000.0,
            segments_count: count + 1,
        });

        batches
    }

    pub fn config(&self) -> VadConfig {
        self.engine.config()
    }
}

/// Run VAD detection with path (for testing).
pub fn run_vad_with_path(audio_path: &str, label: &str) -> Result<()> {
    use aphelios_core::utils::init_logging;

    info!("=== VAD Test: {} ===", label);
    info!("Audio file: {}", audio_path);

    let mut detector = VadDetector::new_default(audio_path)?;
    let segments = detector.detect_from_file(audio_path)?;

    // Compute audio duration (AudioLoader always returns 16kHz mono)
    let audio = AudioLoader::new().load(audio_path)?;
    let mono = audio.into_mono();
    let duration = mono.samples.len() as f64 / 16000.0;

    let total_speech: f64 = segments
        .iter()
        .map(|s| (s.end - s.start) as f64 / 1000.0)
        .sum();
    let speech_ratio = if duration > 0.0 {
        total_speech / duration
    } else {
        0.0
    };

    info!("Audio duration: {:.2}s", duration);
    info!(
        "Speech duration: {:.2}s ({:.1}%)",
        total_speech,
        speech_ratio * 100.0
    );
    info!("Segments detected: {}", segments.len());

    for (i, segment) in segments.iter().enumerate() {
        info!(
            "  Segment {}: {:.2}s - {:.2}s",
            i + 1,
            segment.start,
            segment.end
        );
    }

    assert!(
        !segments.is_empty(),
        "Should detect at least one speech segment"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_detector_default() {
        // Just verify the struct can be created (model path doesn't need to exist for compile check)
        let config = VadConfig::default();
        assert_eq!(config.threshold, 0.5);
    }
}
