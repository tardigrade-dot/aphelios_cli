//! Silero VAD 测试 - 语音活动检测
//!
//! 模型：Silero VAD v4/v5 (ONNX)
//! 输入：单声道 16kHz WAV 音频
//! 输出：每帧的语音概率

use anyhow::Result;
use hound::WavReader;
use ndarray::Array;
use ort::{
    ep::CoreML,
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};

const MODEL_PATH: &str = "/Volumes/sw/onnx_models/silero-vad/onnx/model.onnx";
const AUDIO_PATH: &str = "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4_16k.wav";
const SAMPLE_RATE: i64 = 16000;
const CHUNK_SIZE: usize = 512; // Silero VAD 推荐：512 (32ms @ 16kHz)
const SPEECH_THRESHOLD: f32 = 0.5;

/// 加载 WAV 音频文件并返回 f32 格式的 PCM 数据
fn load_wav_audio(path: &str) -> Result<Vec<f32>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    eprintln!("音频规格：{:?} channels={}", spec, spec.channels);

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

    Ok(samples)
}

/// 将音频分割成固定大小的 chunk
fn chunk_audio(audio: &[f32], chunk_size: usize) -> Vec<Vec<f32>> {
    audio
        .chunks(chunk_size)
        .map(|chunk| {
            let mut c = chunk.to_vec();
            // 最后一个 chunk 可能不足，补零
            if c.len() < chunk_size {
                c.resize(chunk_size, 0.0);
            }
            c
        })
        .collect()
}

#[test]
fn test_silero_vad() -> Result<()> {
    eprintln!("加载模型：{}", MODEL_PATH);

    let coreml_options = CoreML::default().with_subgraphs(true);
    let coreml_provider = coreml_options.build();

    // 1. 初始化推理会话
    let mut session = Session::builder()?
        .with_execution_providers([coreml_provider])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .commit_from_file(MODEL_PATH)?;

    eprintln!("模型加载完成");

    // 2. 加载音频
    let audio_data = load_wav_audio(AUDIO_PATH)?;
    eprintln!("音频加载完成：{} 采样点", audio_data.len());

    // 3. 准备初始状态 [2, 1, 128] - Silero VAD v4/v5 使用 128 维状态
    let mut state = Array::<f32, _>::zeros((2, 1, 128));
    let sr = Array::<i64, _>::from_elem((1,), SAMPLE_RATE);

    // 4. 分块处理
    let chunks = chunk_audio(&audio_data, CHUNK_SIZE);
    eprintln!("分块数量：{}", chunks.len());

    let mut speech_frames = 0;
    let mut total_frames = 0;
    let mut probabilities: Vec<f32> = Vec::with_capacity(chunks.len());

    for (i, chunk) in chunks.iter().enumerate() {
        let input_tensor = Array::<f32, _>::from_shape_vec((1, CHUNK_SIZE), chunk.clone())?;

        // 转换为 ort::Value
        let input_value = Value::from_array(input_tensor)?;
        let sr_value = Value::from_array(sr.clone())?;
        let state_value = Value::from_array(state.clone())?;

        let outputs = session.run(inputs![
            "input" => input_value,
            "sr" => sr_value,
            "state" => state_value,
        ])?;

        // 获取输出概率 [1, 1]
        let (_, output_data) = outputs["output"].try_extract_tensor::<f32>()?;
        let probability = output_data[0];
        probabilities.push(probability);

        // 更新状态 [2, 1, 128]
        let (_, next_state_data) = outputs["stateN"].try_extract_tensor::<f32>()?;
        state = Array::from_shape_vec((2, 1, 128), next_state_data.to_vec())?;

        if probability > SPEECH_THRESHOLD {
            speech_frames += 1;
        }
        total_frames += 1;

        // 每 100 帧打印一次进度
        if (i + 1) % 100 == 0 {
            eprintln!(
                "处理进度：{}/{} 帧 ({:.1}%)",
                i + 1,
                chunks.len(),
                (i + 1) as f32 / chunks.len() as f32 * 100.0
            );
        }
    }

    // 5. 统计结果
    let speech_ratio = speech_frames as f32 / total_frames as f32;
    eprintln!("\n========== VAD 检测结果 ==========");
    eprintln!("总帧数：{}", total_frames);
    eprintln!("语音帧数：{} ({:.1}%)", speech_frames, speech_ratio * 100.0);
    eprintln!(
        "音频时长：{:.2} 秒",
        audio_data.len() as f32 / SAMPLE_RATE as f32
    );
    eprintln!("================================");

    // 6. 检测语音片段
    let segments = detect_speech_segments(&probabilities, CHUNK_SIZE, SAMPLE_RATE);
    eprintln!("\n检测到的语音片段数量：{}", segments.len());
    for (i, (start, end, duration)) in segments.iter().enumerate() {
        eprintln!(
            "  片段 {}: {:.2}s - {:.2}s (时长：{:.2}s)",
            i + 1,
            start,
            end,
            duration
        );
    }

    Ok(())
}

/// 检测语音片段边界
fn detect_speech_segments(
    probabilities: &[f32],
    chunk_size: usize,
    sample_rate: i64,
) -> Vec<(f32, f32, f32)> {
    let mut segments = Vec::new();
    let mut in_speech = false;
    let mut start_frame = 0;
    let ms_per_frame = (chunk_size as f32 / sample_rate as f32) * 1000.0;

    for (i, &prob) in probabilities.iter().enumerate() {
        if prob > SPEECH_THRESHOLD {
            if !in_speech {
                // 语音开始
                in_speech = true;
                start_frame = i;
            }
        } else {
            if in_speech {
                // 语音结束
                in_speech = false;
                let start_sec = start_frame as f32 * ms_per_frame / 1000.0;
                let end_sec = i as f32 * ms_per_frame / 1000.0;
                let duration = end_sec - start_sec;
                if duration > 0.1 {
                    // 过滤太短的片段
                    segments.push((start_sec, end_sec, duration));
                }
            }
        }
    }

    // 处理最后一段
    if in_speech {
        let start_sec = start_frame as f32 * ms_per_frame / 1000.0;
        let end_sec = probabilities.len() as f32 * ms_per_frame / 1000.0;
        let duration = end_sec - start_sec;
        if duration > 0.1 {
            segments.push((start_sec, end_sec, duration));
        }
    }

    segments
}
