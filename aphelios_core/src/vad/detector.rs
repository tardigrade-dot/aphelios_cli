//! Silero VAD 检测器实现

use anyhow::Result;
use ndarray::Array;
use ort::{
    ep::CoreML,
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use tracing::{debug, info};

use crate::audio::{AudioLoader, MonoBuffer, ResampleQuality, Resampler};
use crate::utils::ScopedTimer;

use super::types::{VadResult, VadSegment};

/// VAD 配置
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// 模型路径
    pub model_path: String,
    /// 采样率（Silero VAD 需要 16000Hz）
    pub sample_rate: u32,
    /// 分块大小（采样点数）
    pub chunk_size: usize,
    /// 语音检测阈值
    pub threshold: f32,
    /// 最小片段时长（秒）
    pub min_segment_duration: f64,
    /// 合并相邻片段的最大间隔（秒）
    pub merge_gap: f64,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            model_path: "/Volumes/sw/onnx_models/silero-vad/onnx/model.onnx".to_string(),
            sample_rate: 16000,
            chunk_size: 512, // 32ms @ 16kHz
            threshold: 0.5,
            min_segment_duration: 0.1,
            merge_gap: 0.3,
        }
    }
}

/// VAD 检测器
pub struct VadDetector {
    config: VadConfig,
    session: Session,
}

impl VadDetector {
    /// 创建新的 VAD 检测器
    pub fn new(config: VadConfig) -> Result<Self> {
        let _timer = ScopedTimer::new("VadDetector::new");

        info!("Loading VAD model from: {}", config.model_path);

        let coreml_options = CoreML::default().with_subgraphs(true);
        let session = Session::builder()?
            .with_execution_providers([coreml_options.build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(&config.model_path)?;

        info!("VAD model loaded successfully");

        Ok(Self { config, session })
    }

    /// 从文件检测语音活动
    pub fn detect_from_file(&mut self, audio_path: &str) -> Result<VadResult> {
        let _timer = ScopedTimer::new("VadDetector::detect_from_file");

        // 加载音频
        let audio = AudioLoader::new().load(audio_path)?;
        let mono = audio.to_stereo().to_mono();

        self.detect(mono)
    }

    /// 检测语音活动
    pub fn detect(&mut self, audio: MonoBuffer) -> Result<VadResult> {
        let _timer = ScopedTimer::new("VadDetector::detect");

        info!(
            "Starting VAD detection: {} samples @ {}Hz ({:.2}s)",
            audio.samples.len(),
            audio.sample_rate,
            audio.duration_secs()
        );

        // 重采样到目标采样率
        let audio = if audio.sample_rate != self.config.sample_rate {
            let resampler = Resampler::new().with_quality(ResampleQuality::Fast);
            resampler.resample_mono(&audio, self.config.sample_rate)?
        } else {
            audio
        };

        // 分块处理
        let probabilities = self.process_audio(&audio.samples)?;

        // 检测片段边界
        let mut segments = self.detect_segments(&probabilities);

        info!("Detected {} speech segments", segments.len());

        // 合并相邻片段
        if self.config.merge_gap > 0.0 {
            let mut result = VadResult::new(audio.duration_secs(), segments);
            result.merge_adjacent(self.config.merge_gap);
            segments = result.segments;
        }

        let result = VadResult::new(audio.duration_secs(), segments);

        info!(
            "VAD completed: {:.2}s total audio, {:.2}s speech ({:.1}%)",
            result.audio_duration,
            result.total_speech_duration,
            result.speech_ratio * 100.0
        );

        Ok(result)
    }

    /// 处理音频并返回每帧的语音概率
    fn process_audio(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let chunk_size = self.config.chunk_size;
        let sample_rate = self.config.sample_rate as i64;

        // 准备初始状态 [2, 1, 128]
        let mut state = Array::<f32, _>::zeros((2, 1, 128));
        let sr = Array::<i64, _>::from_elem((1,), sample_rate);

        let chunks: Vec<&[f32]> = samples.chunks(chunk_size).collect();
        let mut probabilities = Vec::with_capacity(chunks.len());

        debug!("Processing {} audio chunks", chunks.len());

        for (i, chunk) in chunks.iter().enumerate() {
            // 准备输入
            let mut chunk_data = chunk.to_vec();
            if chunk_data.len() < chunk_size {
                chunk_data.resize(chunk_size, 0.0);
            }

            let input_tensor = Array::<f32, _>::from_shape_vec((1, chunk_size), chunk_data)?;
            let input_value = Value::from_array(input_tensor)?;
            let sr_value = Value::from_array(sr.clone())?;
            let state_value = Value::from_array(state.clone())?;

            // 运行推理
            let outputs = self.session.run(inputs![
                "input" => input_value,
                "sr" => sr_value,
                "state" => state_value,
            ])?;

            // 提取概率
            let (_, output_data) = outputs["output"].try_extract_tensor::<f32>()?;
            probabilities.push(output_data[0]);

            // 更新状态
            let (_, next_state_data) = outputs["stateN"].try_extract_tensor::<f32>()?;
            state = Array::from_shape_vec((2, 1, 128), next_state_data.to_vec())?;

            // 进度日志
            if (i + 1) % 200 == 0 {
                debug!(
                    "VAD progress: {}/{} chunks ({:.1}%)",
                    i + 1,
                    chunks.len(),
                    (i + 1) as f32 / chunks.len() as f32 * 100.0
                );
            }
        }

        Ok(probabilities)
    }

    /// 从概率序列中检测语音片段
    fn detect_segments(&self, probabilities: &[f32]) -> Vec<VadSegment> {
        let seconds_per_frame = self.config.chunk_size as f64 / self.config.sample_rate as f64;
        let threshold = self.config.threshold;
        let min_duration = self.config.min_segment_duration;

        let mut segments = Vec::new();
        let mut in_speech = false;
        let mut start_frame = 0;
        let mut speech_probs = Vec::new();

        for (i, &prob) in probabilities.iter().enumerate() {
            if prob >= threshold {
                if !in_speech {
                    // 语音开始
                    in_speech = true;
                    start_frame = i;
                    speech_probs.clear();
                }
                speech_probs.push(prob);
            } else {
                if in_speech {
                    // 语音结束
                    in_speech = false;
                    let start = start_frame as f64 * seconds_per_frame;
                    let end = i as f64 * seconds_per_frame;
                    let duration = end - start;

                    if duration >= min_duration {
                        let avg_prob = speech_probs.iter().sum::<f32>() / speech_probs.len() as f32;
                        segments.push(VadSegment::new(start, end, avg_prob));
                    }
                }
            }
        }

        // 处理最后一段
        if in_speech {
            let start = start_frame as f64 * seconds_per_frame;
            let end = probabilities.len() as f64 * seconds_per_frame;
            let duration = end - start;

            if duration >= min_duration {
                let avg_prob = speech_probs.iter().sum::<f32>() / speech_probs.len() as f32;
                segments.push(VadSegment::new(start, end, avg_prob));
            }
        }

        segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_config_default() {
        let config = VadConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.threshold, 0.5);
    }
}
