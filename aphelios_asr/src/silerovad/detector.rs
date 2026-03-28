//! Silero VAD 检测器实现

use anyhow::Result;
use aphelios_core::{
    audio::{MonoBuffer, ResampleQuality},
    AudioLoader, Resampler, ScopedTimer,
};
use ndarray::Array;
use ndarray::{Array2, Array3};
use ort::{
    ep::CoreML,
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use tracing::{debug, info};

use super::types::VadResult;
use crate::base::VadSegment;

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

        // 1. 预创建静态输入：采样率 Tensor 只需要创建一次
        let sr_tensor = Array::from_elem((1,), sample_rate);
        let sr_value = Value::from_array(sr_tensor)?;

        // 2. 初始化状态：直接使用 ndarray 维持
        let mut state = Array3::<f32>::zeros((2, 1, 128));

        // 3. 预分配结果空间
        let num_chunks = (samples.len() + chunk_size - 1) / chunk_size;
        let mut probabilities = Vec::with_capacity(num_chunks);

        // 4. 用于处理最后一个不足 chunk 的缓冲区
        let mut padding_buffer = vec![0.0f32; chunk_size];

        debug!("Processing {} audio chunks", num_chunks);

        for (i, chunk) in samples.chunks(chunk_size).enumerate() {
            // 处理输入数据：如果是最后一帧且长度不足，复制到 padding_buffer
            let input_tensor = if chunk.len() == chunk_size {
                // 直接从 slice 创建 View，避免 to_vec() 拷贝
                Array2::from_shape_vec((1, chunk_size), chunk.to_vec())?
                // 注意：ort 1.x 某些版本可能需要具体所有权的 Array，
                // 如果支持 View，推荐使用 ArrayView2::from_shape((1, chunk_size), chunk)?
            } else {
                padding_buffer[..chunk.len()].copy_from_slice(chunk);
                padding_buffer[chunk.len()..].fill(0.0);
                Array2::from_shape_vec((1, chunk_size), padding_buffer.clone())?
            };

            // 5. 运行推理：注意 state 不需要 clone，传入引用即可
            let input_value = Value::from_array(input_tensor)?;
            let state_value = Value::from_array(state.clone())?;
            let outputs = self.session.run(inputs![
                "input" => input_value,
                "sr" => &sr_value,
                "state" => state_value,
            ])?;

            // 6. 提取概率
            let (output_shape, output_data) = outputs["output"].try_extract_tensor::<f32>()?;
            probabilities.push(output_data[0]);

            // 7. 高效更新状态：直接将输出的 Tensor 转换为 ndarray 并赋值给 state
            let (state_shape, state_data) = outputs["stateN"].try_extract_tensor::<f32>()?;
            // 从 slice 数据重建 Array3
            let shape = state_shape.to_vec();
            state = Array3::from_shape_vec(
                (shape[0] as usize, shape[1] as usize, shape[2] as usize),
                state_data.to_vec(),
            )?;

            if (i + 1) % 200 == 0 {
                debug!(
                    "VAD progress: {}/{} chunks ({:.1}%)",
                    i + 1,
                    num_chunks,
                    (i + 1) as f32 / num_chunks as f32 * 100.0
                );
            }
        }

        Ok(probabilities)
    }

    /// 从概率序列中检测语音片段
    fn detect_segments(&self, probabilities: &[f32]) -> Vec<VadSegment> {
        let seconds_per_frame = self.config.chunk_size as f64 / self.config.sample_rate as f64;
        let threshold = self.config.threshold;
        let min_speech_duration = self.config.min_segment_duration; // 最小语音长度

        // 新增配置：最小静音长度（建议 0.3s - 1.0s），只有静音超过这个时间才切断
        let min_silence_duration = 0.5;
        // 新增配置：首尾缓冲时间（建议 0.1s），防止切掉辅音
        let speech_pad_ms = 0.1;

        let mut segments = Vec::new();
        let mut in_speech = false;
        let mut start_frame = 0;
        let mut silence_frames = 0;
        let max_silence_frames = (min_silence_duration / seconds_per_frame) as usize;

        for (i, &prob) in probabilities.iter().enumerate() {
            if prob >= threshold {
                if !in_speech {
                    in_speech = true;
                    start_frame = i;
                }
                silence_frames = 0; // 重置静音计数
            } else {
                if in_speech {
                    silence_frames += 1;
                    // 只有当静音持续帧数超过阈值，才认为这段说话结束了
                    if silence_frames >= max_silence_frames {
                        in_speech = false;

                        // 计算时间戳，减去多算的静音帧
                        let end_frame = i - silence_frames + 1;
                        self.push_segment(
                            &mut segments,
                            start_frame,
                            end_frame,
                            seconds_per_frame,
                            speech_pad_ms,
                            min_speech_duration,
                            probabilities,
                        );
                    }
                }
            }
        }

        // 处理最后残留的片段
        if in_speech {
            self.push_segment(
                &mut segments,
                start_frame,
                probabilities.len(),
                seconds_per_frame,
                speech_pad_ms,
                min_speech_duration,
                probabilities,
            );
        }

        segments
    }

    // 辅助函数：处理 Padding 和 长度校验
    fn push_segment(
        &self,
        segments: &mut Vec<VadSegment>,
        start_f: usize,
        end_f: usize,
        spf: f64,
        pad: f64,
        min_dur: f64,
        probs: &[f32],
    ) {
        // 加上 Padding 并且防止越界
        let start = (start_f as f64 * spf - pad).max(0.0);
        let end = (end_f as f64 * spf + pad).min(probs.len() as f64 * spf);
        let duration = end - start;

        if duration >= min_dur {
            let avg_prob = probs[start_f..end_f].iter().sum::<f32>() / (end_f - start_f) as f32;
            segments.push(VadSegment::new(start, end, avg_prob));
        }
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
