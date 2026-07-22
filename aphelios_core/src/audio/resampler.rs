//! 音频重采样器
//!
//! 提供多种质量级别的重采样算法

use anyhow::Result;
use tracing::info;

use super::types::{MonoBuffer, StereoBuffer};

/// 重采样质量级别
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleQuality {
    /// 快速线性插值
    Fast,
    /// 高质量 SINC 插值
    High,
}

impl Default for ResampleQuality {
    fn default() -> Self {
        Self::Fast
    }
}

/// 音频重采样器
pub struct Resampler {
    quality: ResampleQuality,
}

impl Resampler {
    pub fn new() -> Self {
        Self {
            quality: ResampleQuality::default(),
        }
    }

    pub fn with_quality(mut self, quality: ResampleQuality) -> Self {
        self.quality = quality;
        self
    }

    /// 重采样单声道音频
    pub fn resample_mono(&self, input: &MonoBuffer, target_rate: u32) -> Result<MonoBuffer> {
        if input.sample_rate == target_rate {
            return Ok(input.clone());
        }

        info!("Resampling mono audio: {}Hz -> {}Hz ({} samples)", input.sample_rate, target_rate, input.samples.len());

        let samples = match self.quality {
            ResampleQuality::Fast => self.linear_resample(&input.samples, input.sample_rate, target_rate),
            ResampleQuality::High => self.sinc_resample(&input.samples, input.sample_rate, target_rate),
        };

        info!("Resampled mono audio: {}Hz -> {}Hz ({} samples)", input.sample_rate, target_rate, samples.len());
        Ok(MonoBuffer::new(samples, target_rate))
    }

    /// 重采样立体声音频
    pub fn resample_stereo(&self, input: &StereoBuffer, target_rate: u32) -> Result<StereoBuffer> {
        if input.sample_rate == target_rate {
            return Ok(input.clone());
        }

        info!("Resampling stereo audio: {}Hz -> {}Hz ({} samples)", input.sample_rate, target_rate, input.left.len());

        let left = match self.quality {
            ResampleQuality::Fast => self.linear_resample(&input.left, input.sample_rate, target_rate),
            ResampleQuality::High => self.sinc_resample(&input.left, input.sample_rate, target_rate),
        };

        let right = match self.quality {
            ResampleQuality::Fast => self.linear_resample(&input.right, input.sample_rate, target_rate),
            ResampleQuality::High => self.sinc_resample(&input.right, input.sample_rate, target_rate),
        };

        Ok(StereoBuffer::new(left, right, target_rate))
    }

    /// 线性插值重采样（快速）
    fn linear_resample(&self, samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
        let ratio = dst_rate as f64 / src_rate as f64;
        let new_len = (samples.len() as f64 * ratio).floor() as usize;

        if new_len == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_idx = i as f64 / ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(samples.len() - 1);
            let frac = src_idx - idx0 as f64;

            let sample = samples[idx0] as f64 * (1.0 - frac) + samples[idx1] as f64 * frac;
            result.push(sample as f32);
        }

        result
    }

    /// SINC 重采样（高质量）
    fn sinc_resample(&self, samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
        // 使用 rubato 库进行高质量重采样
        use rubato::audioadapter_buffers::direct::InterleavedSlice;
        use rubato::{Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, SincInterpolationType, WindowFunction};

        let ratio = dst_rate as f64 / src_rate as f64;

        let params = SincInterpolationParameters::new(256, WindowFunction::BlackmanHarris2)
            .f_cutoff(0.95)
            .oversampling_factor(256)
            .interpolation(SincInterpolationType::Linear);

        let chunk_size = 4096;
        let mut resampler = Async::<f32>::new_sinc(ratio, 2.0, &params, chunk_size, 1, FixedAsync::Input).expect("Failed to create resampler");

        let input_frames = samples.len();
        if input_frames == 0 {
            return Vec::new();
        }

        // Create input adapter wrapping the entire sample buffer
        let input_adapter = InterleavedSlice::new(samples, 1, input_frames).expect("Failed to create input adapter");

        // Pre-allocate output buffer with generous capacity to avoid overflow
        let estimated_frames = 2 * ((input_frames as f64) * ratio).ceil() as usize + resampler.output_delay() + chunk_size;
        let mut output_data = vec![0.0f32; estimated_frames.max(1024)];
        let output_adapter_frames = output_data.len(); // mono: frames = len
        let mut output_adapter = InterleavedSlice::new_mut(&mut output_data, 1, output_adapter_frames).expect("Failed to create output adapter");

        let mut indexing = Indexing::new();
        let mut input_frames_left = input_frames;
        let mut input_frames_next = resampler.input_frames_next();

        // Process full chunks
        while input_frames_left >= input_frames_next {
            let (nbr_in, nbr_out) = resampler
                .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
                .unwrap_or_else(|e| {
                    tracing::warn!("Resampling error: {:?}", e);
                    (0, 0)
                });
            if nbr_in == 0 {
                break;
            }
            indexing.input_offset += nbr_in;
            indexing.output_offset += nbr_out;
            input_frames_left -= nbr_in;
            input_frames_next = resampler.input_frames_next();
        }

        // Process remaining partial chunk
        if input_frames_left > 0 {
            indexing.partial_len = Some(input_frames_left);
            if let Ok((_nbr_in, nbr_out)) = resampler.process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing)) {
                indexing.output_offset += nbr_out;
            }
        }

        // Truncate to actual output size
        output_data.truncate(indexing.output_offset);

        // 调整到精确的目标长度
        let target_len = (input_frames as f64 * ratio).floor() as usize;
        if output_data.len() > target_len {
            output_data.truncate(target_len);
        }

        output_data
    }
}

impl Default for Resampler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_resample() {
        let resampler = Resampler::new().with_quality(ResampleQuality::Fast);
        let input = MonoBuffer::new(vec![1.0, 0.5, 0.0, -0.5, -1.0], 16000);

        let output = resampler
            .resample_mono(&input, 8000)
            .unwrap();
        assert_eq!(output.sample_rate, 8000);
        assert!(output.len() > 0);
    }

    #[test]
    fn test_resample_same_rate() {
        let resampler = Resampler::new();
        let input = MonoBuffer::new(vec![1.0, 0.5, 0.0], 16000);

        let output = resampler
            .resample_mono(&input, 16000)
            .unwrap();
        assert_eq!(output.sample_rate, 16000);
        assert_eq!(output.len(), input.len());
    }
}
