//! VAD + ASR 联合测试 - 先使用 VAD 检测语音片段，再对每个片段进行 ASR 识别
//!
//! 流程：
//! 1. 使用 Silero VAD 检测音频中的语音片段
//! 2. 根据检测到的片段边界截取音频
//! 3. 对每个语音片段运行 Whisper ASR
//! 4. 合并所有结果并生成 SRT 文件

use anyhow::Result;
use aphelios_core::{
    asr::{Segment, generate_srt, run_whisper_with_pcm},
    utils,
};
use candle_core::Device;
use hound::{WavReader, WavSpec, WavWriter};
use ndarray::Array;
use ort::{
    ep::CoreML,
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use tracing::{error, info};

// ==================== 配置常量 ====================

const VAD_MODEL_PATH: &str = "/Volumes/sw/onnx_models/silero-vad/onnx/model.onnx";
const ASR_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/distil-large-v3.5";
const AUDIO_16K: &str = "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4_16k.wav";
const AUDIO_44K: &str = "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4.wav";

const TARGET_SAMPLE_RATE: u32 = 16000;
const CHUNK_SIZE: usize = 512;
const SPEECH_THRESHOLD: f32 = 0.5;
const MIN_SEGMENT_DURATION: f32 = 0.5;
const PADDING_DURATION: f32 = 0.1;

// ==================== 语音片段结构 ====================

#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub start: f64,
    pub end: f64,
    pub duration: f64,
}

// ==================== 音频处理函数 ====================

/// 加载 WAV 音频文件并返回 f32 格式的 PCM 数据和采样率
fn load_wav_audio(path: &str) -> Result<(Vec<f32>, u32)> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    info!("音频规格：{:?} channels={}", spec, spec.channels);

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|s| s.unwrap() as f32 / i16::MAX as f32)
                .collect(),
            32 => reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / i32::MAX as f32)
                .collect(),
            _ => anyhow::bail!("Unsupported bit depth: {}", spec.bits_per_sample),
        },
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
    };

    Ok((samples, spec.sample_rate))
}

/// 将音频分割成固定大小的 chunk
fn chunk_audio(audio: &[f32], chunk_size: usize) -> Vec<Vec<f32>> {
    audio
        .chunks(chunk_size)
        .map(|chunk| {
            let mut c = chunk.to_vec();
            if c.len() < chunk_size {
                c.resize(chunk_size, 0.0);
            }
            c
        })
        .collect()
}

/// 从完整音频中截取指定时间范围的片段
fn extract_audio_segment(
    audio: &[f32],
    sample_rate: u32,
    start: f64,
    end: f64,
    padding: f64,
) -> Vec<f32> {
    let start_sample = ((start - padding).max(0.0) * sample_rate as f64) as usize;
    let end_sample = ((end + padding) * sample_rate as f64) as usize;
    let end_sample = end_sample.min(audio.len());

    if start_sample >= end_sample {
        return vec![];
    }

    audio[start_sample..end_sample].to_vec()
}

/// 保存 PCM 数据到临时 WAV 文件
fn save_pcm_to_wav(pcm: &[f32], sample_rate: u32, path: &str) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;
    for &sample in pcm {
        writer.write_sample((sample * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

// ==================== VAD 处理 ====================

/// 使用 Silero VAD 检测语音片段
fn detect_voice_segments(audio: &[f32], sample_rate: u32) -> Result<Vec<SpeechSegment>> {
    info!("加载 VAD 模型：{}", VAD_MODEL_PATH);

    let coreml_options = CoreML::default().with_subgraphs(true);
    let coreml_provider = coreml_options.build();

    let mut session = Session::builder()?
        .with_execution_providers([coreml_provider])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .commit_from_file(VAD_MODEL_PATH)?;

    info!("VAD 模型加载完成");

    // 准备初始状态 [2, 1, 128]
    let mut state = Array::<f32, _>::zeros((2, 1, 128));
    let sr = Array::<i64, _>::from_elem((1,), sample_rate as i64);

    // 分块处理
    let chunks = chunk_audio(audio, CHUNK_SIZE);
    info!("音频分块数量：{}", chunks.len());

    let mut probabilities: Vec<f32> = Vec::with_capacity(chunks.len());

    for (i, chunk) in chunks.iter().enumerate() {
        let input_tensor = Array::<f32, _>::from_shape_vec((1, CHUNK_SIZE), chunk.clone())?;

        let input_value = Value::from_array(input_tensor)?;
        let sr_value = Value::from_array(sr.clone())?;
        let state_value = Value::from_array(state.clone())?;

        let outputs = session.run(inputs![
            "input" => input_value,
            "sr" => sr_value,
            "state" => state_value,
        ])?;

        let (_, output_data) = outputs["output"].try_extract_tensor::<f32>()?;
        let probability = output_data[0];
        probabilities.push(probability);

        // 更新状态
        let (_, next_state_data) = outputs["stateN"].try_extract_tensor::<f32>()?;
        state = Array::from_shape_vec((2, 1, 128), next_state_data.to_vec())?;

        // 进度日志
        if (i + 1) % 200 == 0 {
            info!(
                "VAD 处理进度：{}/{} 帧 ({:.1}%)",
                i + 1,
                chunks.len(),
                (i + 1) as f32 / chunks.len() as f32 * 100.0
            );
        }
    }

    // 检测语音片段边界
    let segments = detect_speech_segments(&probabilities, CHUNK_SIZE, sample_rate as i64);
    info!("检测到 {} 个语音片段", segments.len());

    Ok(segments
        .into_iter()
        .map(|(start, end, _)| SpeechSegment {
            start,
            end,
            duration: end - start,
        })
        .collect())
}

/// 检测语音片段边界
fn detect_speech_segments(
    probabilities: &[f32],
    chunk_size: usize,
    sample_rate: i64,
) -> Vec<(f64, f64, f64)> {
    let mut segments = Vec::new();
    let mut in_speech = false;
    let mut start_frame = 0;
    let seconds_per_frame = chunk_size as f64 / sample_rate as f64;

    for (i, &prob) in probabilities.iter().enumerate() {
        if prob > SPEECH_THRESHOLD {
            if !in_speech {
                in_speech = true;
                start_frame = i;
            }
        } else {
            if in_speech {
                in_speech = false;
                let start_sec = start_frame as f64 * seconds_per_frame;
                let end_sec = i as f64 * seconds_per_frame;
                let duration = end_sec - start_sec;
                if duration > MIN_SEGMENT_DURATION as f64 {
                    segments.push((start_sec, end_sec, duration));
                }
            }
        }
    }

    // 处理最后一段
    if in_speech {
        let start_sec = start_frame as f64 * seconds_per_frame;
        let end_sec = probabilities.len() as f64 * seconds_per_frame;
        let duration = end_sec - start_sec;
        if duration > MIN_SEGMENT_DURATION as f64 {
            segments.push((start_sec, end_sec, duration));
        }
    }

    segments
}

// ==================== ASR 处理 ====================

/// 对单个语音片段运行 ASR
fn transcribe_segment(
    pcm: &[f32],
    sample_rate: u32,
    segment: &SpeechSegment,
    temp_wav_path: &str,
) -> Result<Vec<Segment>> {
    // 截取带填充的音频片段
    let segment_pcm = extract_audio_segment(
        pcm,
        sample_rate,
        segment.start,
        segment.end,
        PADDING_DURATION as f64,
    );

    if segment_pcm.is_empty() {
        info!("跳过空片段：{:.2}s - {:.2}s", segment.start, segment.end);
        return Ok(vec![]);
    }

    // 保存为临时 WAV 文件
    save_pcm_to_wav(&segment_pcm, sample_rate, temp_wav_path)?;

    // 创建 Metal 设备
    let device = Device::new_metal(0)?;

    // 运行 ASR
    info!(
        "ASR 处理片段：{:.2}s - {:.2}s (时长：{:.2}s, 采样点：{})",
        segment.start,
        segment.end,
        segment.duration,
        segment_pcm.len()
    );

    let segments = run_whisper_with_pcm(ASR_MODEL_DIR, &segment_pcm, sample_rate, &device)?;

    // 调整时间戳为全局时间
    let offset = segment.start - PADDING_DURATION as f64;
    let mut adjusted_segments = Vec::new();

    for mut seg in segments {
        seg.start += offset;
        for sub in &mut seg.sub_segments {
            sub.start += offset;
            sub.end += offset;
        }
        adjusted_segments.push(seg);
    }

    // 清理临时文件
    let _ = std::fs::remove_file(temp_wav_path);

    Ok(adjusted_segments)
}

// ==================== 主测试函数 ====================

/// 运行 VAD+ASR 联合测试的核心逻辑
fn run_vad_asr_pipeline(audio_path: &str, output_suffix: &str) -> Result<()> {
    utils::init_logging();

    info!("========== VAD + ASR 联合处理流程 ==========");
    info!("输入音频：{}", audio_path);

    // 1. 加载音频（自动重采样到 16kHz）
    info!("加载音频文件...");
    let (audio_data, src_sample_rate) = load_wav_audio(audio_path)?;
    info!(
        "音频加载完成：{} 采样点，原始采样率：{}Hz",
        audio_data.len(),
        src_sample_rate
    );
    info!(
        "音频总时长：{:.2} 秒",
        audio_data.len() as f64 / src_sample_rate as f64
    );

    // 2. VAD 检测语音片段（使用 16kHz）
    info!("\n========== 步骤 1: VAD 语音检测 ==========");
    let voice_segments = detect_voice_segments(&audio_data, TARGET_SAMPLE_RATE)?;

    if voice_segments.is_empty() {
        info!("未检测到任何语音片段");
        return Ok(());
    }

    info!("\n检测到的语音片段:");
    for (i, seg) in voice_segments.iter().enumerate() {
        info!(
            "  片段 {}: {:.2}s - {:.2}s (时长：{:.2}s)",
            i + 1,
            seg.start,
            seg.end,
            seg.duration
        );
    }

    // 3. 对每个语音片段运行 ASR
    info!("\n========== 步骤 2: ASR 语音识别 ==========");
    let temp_wav_path = "/tmp/vad_asr_temp.wav";
    let mut all_segments: Vec<Segment> = Vec::new();

    for (i, segment) in voice_segments.iter().enumerate() {
        info!("\n处理片段 {}/{}", i + 1, voice_segments.len());
        match transcribe_segment(&audio_data, TARGET_SAMPLE_RATE, segment, temp_wav_path) {
            Ok(segments) => {
                for seg in &segments {
                    info!(
                        "  [{:.2}s - {:.2}s] {}",
                        seg.start,
                        seg.start + seg.duration,
                        seg.dr.text.trim()
                    );
                }
                all_segments.extend(segments);
            }
            Err(e) => {
                error!("片段 {} 处理失败：{}", i + 1, e);
            }
        }
    }

    // 4. 合并结果并排序
    all_segments.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());

    // 5. 生成 SRT 文件
    let output_srt_path = format!("{}{}", audio_path.trim_end_matches(".wav"), output_suffix);
    info!("\n========== 步骤 3: 生成 SRT 文件 ==========");
    match generate_srt(&all_segments, &output_srt_path) {
        Ok(_) => info!("SRT 文件已保存：{}", output_srt_path),
        Err(e) => error!("生成 SRT 文件失败：{}", e),
    }

    // 6. 打印最终统计
    info!("\n========== 处理完成 ==========");
    info!("总语音片段数：{}", voice_segments.len());
    info!("总 ASR 片段数：{}", all_segments.len());

    let total_duration: f64 = all_segments.iter().map(|s| s.duration).sum();
    info!("识别内容总时长：{:.2} 秒", total_duration);

    Ok(())
}

#[test]
fn test_vad_asr_16k() -> Result<()> {
    info!("=== VAD+ASR Test with 16kHz audio ===");
    run_vad_asr_pipeline(AUDIO_16K, "_vad_asr_16k.srt")
}

#[test]
fn test_vad_asr_44k_to_16k() -> Result<()> {
    info!("=== VAD+ASR Test with 44.1kHz audio (resampled to 16kHz) ===");
    run_vad_asr_pipeline(AUDIO_44K, "_vad_asr_44k.srt")
}
