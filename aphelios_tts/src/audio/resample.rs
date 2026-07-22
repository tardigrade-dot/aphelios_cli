//! Audio resampling using rubato
//!
//! Provides high-quality resampling for converting between different sample rates.

use anyhow::Result;
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Async, FixedAsync, Indexing, PolynomialDegree, Resampler, SincInterpolationParameters, SincInterpolationType, WindowFunction};

use super::AudioBuffer;

/// Resampling quality preset
#[derive(Debug, Clone, Copy, Default)]
pub enum ResampleQuality {
    /// Fast resampling, lower quality
    Fast,
    /// Balanced speed and quality
    #[default]
    Normal,
    /// High quality, slower
    High,
}

/// Audio resampler
pub struct AudioResampler {
    quality: ResampleQuality,
}

impl AudioResampler {
    /// Create a new resampler
    pub fn new(quality: ResampleQuality) -> Self {
        Self {
            quality,
        }
    }

    /// Resample audio to a target sample rate
    pub fn resample(&self, audio: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
        if audio.sample_rate == target_rate {
            return Ok(audio.clone());
        }

        let ratio = target_rate as f64 / audio.sample_rate as f64;

        if matches!(self.quality, ResampleQuality::Fast) {
            self.resample_fast(audio, target_rate, ratio)
        } else {
            self.resample_sinc(audio, target_rate, ratio)
        }
    }

    /// Fast polynomial resampling
    fn resample_fast(&self, audio: &AudioBuffer, target_rate: u32, ratio: f64) -> Result<AudioBuffer> {
        let chunk_size = 1024;
        let mut resampler = Async::<f32>::new_poly(ratio, 1.0, PolynomialDegree::Cubic, chunk_size, 1, FixedAsync::Input).map_err(|e| anyhow::anyhow!("Failed to create fast resampler: {}", e))?;

        let output = process_resample(&mut resampler, &audio.samples, chunk_size)?;
        Ok(AudioBuffer::new(output, target_rate))
    }

    /// High-quality sinc resampling
    fn resample_sinc(&self, audio: &AudioBuffer, target_rate: u32, ratio: f64) -> Result<AudioBuffer> {
        let chunk_size = 1024;
        let sinc_len = if matches!(self.quality, ResampleQuality::High) {
            256
        } else {
            128
        };
        let oversampling = if matches!(self.quality, ResampleQuality::High) {
            256
        } else {
            128
        };

        let params = SincInterpolationParameters::new(sinc_len, WindowFunction::BlackmanHarris2)
            .f_cutoff(0.95)
            .interpolation(SincInterpolationType::Linear)
            .oversampling_factor(oversampling);

        let mut resampler = Async::<f32>::new_sinc(ratio, 1.0, &params, chunk_size, 1, FixedAsync::Input).map_err(|e| anyhow::anyhow!("Failed to create sinc resampler: {}", e))?;

        let output = process_resample(&mut resampler, &audio.samples, chunk_size)?;
        Ok(AudioBuffer::new(output, target_rate))
    }
}

impl Default for AudioResampler {
    fn default() -> Self {
        Self::new(ResampleQuality::Normal)
    }
}

/// Process mono audio through a rubato Async resampler
fn process_resample(resampler: &mut Async<f32>, samples: &[f32], _chunk_size: usize) -> Result<Vec<f32>> {
    let n_samples = samples.len();
    if n_samples == 0 {
        return Ok(Vec::new());
    }

    let input_adapter = InterleavedSlice::new(samples, 1, n_samples).map_err(|e| anyhow::anyhow!("Failed to create input adapter: {}", e))?;

    // Estimate output size
    let estimated_out = 2 * n_samples + 4096;
    let mut output_data = vec![0.0f32; estimated_out.max(1024)];
    let out_len = output_data.len();
    let mut output_adapter = InterleavedSlice::new_mut(&mut output_data, 1, out_len).map_err(|e| anyhow::anyhow!("Failed to create output adapter: {}", e))?;

    let mut indexing = Indexing::new();
    let mut input_left = n_samples;
    let mut input_frames_next = resampler.input_frames_next();

    while input_left >= input_frames_next {
        let (nbr_in, nbr_out) = resampler
            .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
            .map_err(|e| anyhow::anyhow!("Resampling failed: {}", e))?;
        if nbr_in == 0 {
            break;
        }
        indexing.input_offset += nbr_in;
        indexing.output_offset += nbr_out;
        input_left -= nbr_in;
        input_frames_next = resampler.input_frames_next();
    }

    if input_left > 0 {
        indexing.partial_len = Some(input_left);
        if let Ok((_, nbr_out)) = resampler.process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing)) {
            indexing.output_offset += nbr_out;
        }
    }

    output_data.truncate(indexing.output_offset);
    Ok(output_data)
}

/// Convenience function to resample audio
pub fn resample(audio: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
    AudioResampler::default().resample(audio, target_rate)
}

/// Resample to Qwen3-TTS's native 24kHz
pub fn resample_to_24k(audio: &AudioBuffer) -> Result<AudioBuffer> {
    resample(audio, 24000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_resample_quality_default() {
        let resampler = AudioResampler::default();
        // Should use Normal quality
        assert!(matches!(resampler.quality, ResampleQuality::Normal));
    }

    #[test]
    fn test_resample_quality_variants() {
        let _fast = AudioResampler::new(ResampleQuality::Fast);
        let _normal = AudioResampler::new(ResampleQuality::Normal);
        let _high = AudioResampler::new(ResampleQuality::High);
    }

    #[test]
    fn test_no_resample_needed() {
        let audio = AudioBuffer::new(vec![0.0; 1000], 24000);
        let result = resample(&audio, 24000).unwrap();
        assert_eq!(result.sample_rate, 24000);
        assert_eq!(result.len(), audio.len());
    }

    #[test]
    fn test_downsample() {
        // 48kHz -> 24kHz (half)
        let audio = AudioBuffer::new(vec![0.0; 4800], 48000);
        let result = resample(&audio, 24000).unwrap();
        assert_eq!(result.sample_rate, 24000);
        // Should be approximately half the samples
        assert!(result.len() > 2000 && result.len() < 3000);
    }

    #[test]
    fn test_upsample() {
        // 16kHz -> 24kHz (1.5x)
        let audio = AudioBuffer::new(vec![0.0; 1600], 16000);
        let result = resample(&audio, 24000).unwrap();
        assert_eq!(result.sample_rate, 24000);
        // Should be approximately 1.5x the samples (padding may add extra)
        assert!(result.len() > 2000 && result.len() < 4000);
    }

    #[test]
    fn test_resample_to_24k() {
        let audio = AudioBuffer::new(vec![0.0; 1600], 16000);
        let result = resample_to_24k(&audio).unwrap();
        assert_eq!(result.sample_rate, 24000);
    }

    #[test]
    fn test_resample_fast_quality() {
        let resampler = AudioResampler::new(ResampleQuality::Fast);
        let audio = AudioBuffer::new(vec![0.0; 2048], 48000);
        let result = resampler
            .resample(&audio, 24000)
            .unwrap();
        assert_eq!(result.sample_rate, 24000);
    }

    #[test]
    fn test_resample_high_quality() {
        let resampler = AudioResampler::new(ResampleQuality::High);
        let audio = AudioBuffer::new(vec![0.0; 2048], 48000);
        let result = resampler
            .resample(&audio, 24000)
            .unwrap();
        assert_eq!(result.sample_rate, 24000);
    }

    #[test]
    fn test_resample_preserves_sine_wave() {
        // Create a low frequency sine wave that should survive resampling
        let freq = 100.0; // 100 Hz - well below Nyquist for both sample rates
        let audio = AudioBuffer::new(
            (0..4800)
                .map(|i| (2.0 * PI * freq * i as f32 / 48000.0).sin())
                .collect(),
            48000,
        );

        let result = resample(&audio, 24000).unwrap();

        // Check that output has non-zero values
        let max_val = result
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        assert!(max_val > 0.5); // Sine wave should maintain amplitude
    }

    #[test]
    fn test_resample_empty_audio() {
        let audio = AudioBuffer::new(vec![], 24000);
        let result = resample(&audio, 48000).unwrap();
        // Should handle empty gracefully
        assert_eq!(result.sample_rate, 48000);
    }

    #[test]
    fn test_resample_small_audio() {
        let audio = AudioBuffer::new(vec![0.5, -0.5], 24000);
        let result = resample(&audio, 48000).unwrap();
        assert_eq!(result.sample_rate, 48000);
    }
}
