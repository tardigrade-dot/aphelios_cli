use anyhow::Result;
use aphelios_core::AudioLoader;
use ndarray::{Array, Array2, Array3};
use ort::{
    ep::CoreML,
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};

use crate::base::{AudioBatch, VadSegment};

pub struct VadConfig {
    pub sample_rate: i64,      // 16000
    pub chunk_size: usize,     // 512, 1024 或 1536
    pub threshold: f32,        // 0.5 - 0.6
    pub min_speech_ms: f64,    // 建议 200.0
    pub min_silence_ms: f64,   // 建议 500.0 - 1000.0
    pub window_size: usize,    // 滑动窗口帧数，建议 5
    pub window_threshold: f32, // 滑动窗口阈值，建议 0.4
}

pub struct VadProcessor {
    session: Session,
    config: VadConfig,
}

impl VadProcessor {
    pub fn new_default() -> Result<Self> {
        let model_path = "/Volumes/sw/onnx_models/silero-vad/onnx/model.onnx";
        let coreml_options = CoreML::default().with_subgraphs(true);
        let session = Session::builder()?
            .with_execution_providers([coreml_options.build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        // 创建配置
        let config = VadConfig {
            sample_rate: 16000,
            chunk_size: 512,
            threshold: 0.5,
            min_speech_ms: 200.0,
            min_silence_ms: 500.0,
            window_size: 10,
            window_threshold: 0.5,
        };
        Ok(Self { session, config })
    }
    pub fn new(session: Session, config: VadConfig) -> Self {
        Self { session, config }
    }

    pub fn process_from_file(&mut self, audio_path: &str) -> Result<Vec<VadSegment>> {
        use aphelios_core::audio::{ResampleQuality, Resampler};
        let audio = AudioLoader::new().load(audio_path)?;
        let mono = audio.to_stereo().to_mono();
        
        let resampled = if mono.sample_rate != 16000 {
            Resampler::new()
                .with_quality(ResampleQuality::High)
                .resample_mono(&mono, 16000)?
        } else {
            mono
        };

        self.process(&resampled.samples)
    }
    /// 核心处理入口：音频流 -> 概率流 -> 平滑处理 -> 时间戳切片
    pub fn process(&mut self, samples: &[f32]) -> Result<Vec<VadSegment>> {
        // 1. 推理，获取原始概率流 (传递 RNN State)
        let raw_probabilities = self.infer_probabilities(samples)?;

        // 2. 滑动窗口平滑 (75% 比例法)
        let smoothed_probabilities = self.apply_smoothing(&raw_probabilities);

        // 3. 状态机提取片段
        let segments = self.detect_segments(&smoothed_probabilities);

        Ok(segments)
    }
    /// 将零散的 VAD 片段聚合为约 target_duration 秒的长片段
    pub fn aggregate_segments(
        &self,
        segments: &[VadSegment],
        target_duration: f64,
        max_silence_merge: f64, // 两个片段间距小于多少秒时，强制视为连续
    ) -> Vec<AudioBatch> {
        if segments.is_empty() {
            return Vec::new();
        }

        let mut batches = Vec::new();
        let mut current_batch_start = segments[0].start;
        let mut last_segment_end = segments[0].end;
        let mut count = 0;

        for i in 1..segments.len() {
            let seg = &segments[i];
            let gap = seg.start - last_segment_end;
            let potential_duration = seg.end - current_batch_start;

            // 聚合条件：
            // 1. 加上当前段后，总时长未超过目标时长 (如 30s)
            // 2. 或者两段之间的间隙非常小，必须合并以保持连贯
            if potential_duration <= target_duration || gap < max_silence_merge {
                last_segment_end = seg.end;
                count += 1;
            } else {
                // 结束当前 batch
                batches.push(AudioBatch {
                    start: current_batch_start,
                    end: last_segment_end,
                    duration: last_segment_end - current_batch_start,
                    segments_count: count + 1,
                });

                // 开启新 batch
                current_batch_start = seg.start;
                last_segment_end = seg.end;
                count = 0;
            }
        }

        // 别忘了最后一个 batch
        batches.push(AudioBatch {
            start: current_batch_start,
            end: last_segment_end,
            duration: last_segment_end - current_batch_start,
            segments_count: count + 1,
        });

        batches
    }
    /// 推理函数：负责状态管理和 Tensor 转换
    fn infer_probabilities(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let chunk_size = self.config.chunk_size;
        let sample_rate = Array::from_elem((1,), self.config.sample_rate);
        let sr_value = Value::from_array(sample_rate)?;

        // 初始化状态 [2, 1, 128] - 注意：Silero v5 是 128, v4 是 64
        let mut state = Array3::<f32>::zeros((2, 1, 128));

        let num_chunks = (samples.len() + chunk_size - 1) / chunk_size;
        let mut probs = Vec::with_capacity(num_chunks);
        let mut padding_buffer = vec![0.0f32; chunk_size];

        for chunk in samples.chunks(chunk_size) {
            let input_tensor = if chunk.len() == chunk_size {
                Array2::from_shape_vec((1, chunk_size), chunk.to_vec())?
            } else {
                padding_buffer[..chunk.len()].copy_from_slice(chunk);
                padding_buffer[chunk.len()..].fill(0.0);
                Array2::from_shape_vec((1, chunk_size), padding_buffer.clone())?
            };

            let input_value = Value::from_array(input_tensor)?;
            let state_value = Value::from_array(state.clone())?;
            let outputs = self.session.run(inputs![
                "input" => input_value,
                "sr" => &sr_value,
                "state" => state_value,
            ])?;

            // 提取结果并更新 State
            let (_, output_data) = outputs["output"].try_extract_tensor::<f32>()?;
            probs.push(output_data[0]);

            let (state_shape, state_data) = outputs["stateN"].try_extract_tensor::<f32>()?;
            let shape = state_shape.to_vec();
            state = Array3::from_shape_vec(
                (shape[0] as usize, shape[1] as usize, shape[2] as usize),
                state_data.to_vec(),
            )?;
        }

        Ok(probs)
    }

    /// 平滑函数：使用滑动窗口和 75% 多数投票法过滤杂音
    fn apply_smoothing(&self, raw_probs: &[f32]) -> Vec<f32> {
        let ws = self.config.window_size;
        let mut smoothed = Vec::with_capacity(raw_probs.len());

        for i in 0..raw_probs.len() {
            let start = i.saturating_sub(ws / 2);
            let end = (i + ws / 2).min(raw_probs.len() - 1);
            let window = &raw_probs[start..=end];

            // 统计窗口内达标的帧数
            let active_count = window
                .iter()
                .filter(|&&p| p >= self.config.threshold)
                .count();
            let ratio = active_count as f32 / window.len() as f32;

            if ratio >= self.config.window_threshold {
                smoothed.push(raw_probs[i]);
            } else {
                smoothed.push(0.0);
            }
        }
        smoothed
    }

    /// 状态机：支持滞后检测（Hysteresis）和最小长度过滤
    fn detect_segments(&self, probs: &[f32]) -> Vec<VadSegment> {
        let spf = self.config.chunk_size as f64 / self.config.sample_rate as f64;
        let mut segments = Vec::new();

        let mut in_speech = false;
        let mut start_frame = 0;
        let mut silence_frames = 0;

        let max_silence_frames = (self.config.min_silence_ms / 1000.0 / spf) as usize;
        let min_speech_frames = (self.config.min_speech_ms / 1000.0 / spf) as usize;

        for (i, &prob) in probs.iter().enumerate() {
            if prob >= self.config.threshold {
                if !in_speech {
                    in_speech = true;
                    start_frame = i;
                }
                silence_frames = 0;
            } else if in_speech {
                silence_frames += 1;
                if silence_frames >= max_silence_frames {
                    in_speech = false;
                    let end_frame = i - silence_frames + 1;
                    self.push_if_valid(
                        &mut segments,
                        start_frame,
                        end_frame,
                        spf,
                        min_speech_frames,
                        probs,
                    );
                }
            }
        }

        if in_speech {
            self.push_if_valid(
                &mut segments,
                start_frame,
                probs.len(),
                spf,
                min_speech_frames,
                probs,
            );
        }

        segments
    }

    fn push_if_valid(
        &self,
        segments: &mut Vec<VadSegment>,
        start_f: usize,
        end_f: usize,
        spf: f64,
        min_frames: usize,
        probs: &[f32],
    ) {
        if end_f - start_f < min_frames {
            return;
        }

        let chunk = &probs[start_f..end_f];
        let avg_prob = chunk.iter().sum::<f32>() / chunk.len() as f32;

        // 最终防线：如果平均概率太低或整段中没有一个极高点，则抛弃
        let max_prob = chunk.iter().fold(0.0f32, |a, &b| a.max(b));
        if avg_prob < 0.35 || max_prob < self.config.threshold + 0.1 {
            return;
        }

        segments.push(VadSegment::new(
            start_f as f64 * spf,
            end_f as f64 * spf,
            avg_prob,
        ));
    }
}
