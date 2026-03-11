//! DIA (Speaker Diarization) 测试
use anyhow::Result;
use aphelios_core::audio::ResampleQuality;
use aphelios_core::{init_logging, AudioLoader, Resampler};
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer, SpeakerSegment};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;
use tracing::{error, info};

/// 测试音频文件路径
pub const TEST_AUDIO_16K: &str = "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4_16k.wav";
pub const TEST_AUDIO_44K: &str = "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4.wav";

/// 运行 DIA 测试
fn run_dia_test(audio_path: &str, label: &str) -> Result<()> {
    setup();

    info!("=== DIA Test: {} ===", label);
    info!("Audio file: {}", audio_path);

    let config = TestConfig::default();
    let result = dia_process(audio_path, &config.dia_model());

    match result {
        Ok(segments) => {
            info!("Detected {} speaker segments", segments.len());

            for (i, seg) in segments.iter().enumerate() {
                info!(
                    "  Segment {}: {:.2}s - {:.2}s (Speaker {})",
                    i + 1,
                    seg.start,
                    seg.end,
                    seg.speaker_id
                );
            }

            // 保存结果
            let output_path = format!("{}_dia_output.txt", audio_path.trim_end_matches(".wav"));
            let file = File::create(&output_path)?;
            let mut writer = BufWriter::new(file);

            for seg in segments {
                writeln!(
                    writer,
                    "[{:06.2}s - {:06.2}s] Speaker {}",
                    seg.start, seg.end, seg.speaker_id
                )?;
            }
            writer.flush()?;

            info!("Results saved to: {}", output_path);
        }
        Err(e) => {
            error!("DIA failed: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

#[test]
fn test_dia_16k() -> Result<()> {
    run_dia_test(TEST_AUDIO_16K, "16kHz audio (native)")
}

#[test]
fn test_dia_44k_to_16k() -> Result<()> {
    run_dia_test(TEST_AUDIO_44K, "44.1kHz audio (resampled to 16kHz)")
}

#[test]
fn test_dia_arbitrary_wav() -> Result<()> {
    // 保留原有的任意 WAV 文件测试
    let audio_path = std::env::var("ARBITRARY_WAV_PATH").unwrap_or(TEST_AUDIO_44K.to_string());
    run_dia_test(&audio_path, "arbitrary WAV file")
}

pub fn setup() {
    init_logging();
}

/// 加载测试音频文件（自动重采样到目标采样率）
pub fn load_test_audio(path: &str, target_sample_rate: u32) -> aphelios_core::audio::StereoBuffer {
    let audio = AudioLoader::new()
        .load(path)
        .expect("Failed to load audio file");

    let stereo = audio.to_stereo();

    if stereo.sample_rate != target_sample_rate {
        let resampler = Resampler::new().with_quality(ResampleQuality::Fast);
        resampler
            .resample_stereo(&stereo, target_sample_rate)
            .expect("Failed to resample audio")
    } else {
        stereo
    }
}

/// 测试配置
#[derive(Debug, Clone)]
pub struct TestConfig {
    pub model_dir: String,
    pub output_dir: String,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            model_dir: "/Volumes/sw/pretrained_models".to_string(),
            output_dir: "./test_output".to_string(),
        }
    }
}

impl TestConfig {
    pub fn whisper_model(&self) -> String {
        format!("{}/distil-large-v3.5", self.model_dir)
    }

    pub fn vad_model(&self) -> String {
        format!("{}/silero-vad/onnx/model.onnx", self.model_dir)
    }

    pub fn demucs_model(&self) -> String {
        "/Users/larry/coderesp/aphelios_cli/aphelios_core/onnx_models/demucs/htdemucs_embedded.onnx"
            .to_string()
    }

    pub fn dia_model(&self) -> String {
        "/Users/larry/coderesp/aphelios_cli/aphelios_core/onnx_models/dia/diar_streaming_sortformer_4spk-v2.onnx".to_string()
    }
}

pub fn dia_process(audio_path: &str, model_path: &str) -> Result<Vec<SpeakerSegment>> {
    let target_sample_rate = 16000;

    // 加载音频
    let audio = AudioLoader::new().load(audio_path)?;
    let mut stereo = audio.to_stereo();

    // 重采样到 16kHz
    if stereo.sample_rate != target_sample_rate {
        let resampler = Resampler::new().with_quality(ResampleQuality::Fast);
        stereo = resampler.resample_stereo(&stereo, target_sample_rate)?;
    }

    // 转换为交错格式
    let audio: Vec<f32> = stereo
        .left
        .iter()
        .zip(stereo.right.iter())
        .flat_map(|(&l, &r)| vec![l, r])
        .collect();

    let duration = audio.len() as f32 / target_sample_rate as f32 / 2.0;
    println!(
        "Loaded {} samples ({} Hz, 2 channels, {:.1}s)",
        audio.len(),
        target_sample_rate,
        duration
    );

    println!("{}", "=".repeat(80));
    println!("Step 2/3: Performing speaker diarization with Sortformer v2 (streaming)...");

    // Create Sortformer with default config (callhome)
    let mut sortformer = Sortformer::with_config(
        model_path,
        None, // default exec config
        DiarizationConfig::callhome(),
    )?;

    let start = Instant::now();
    // For streaming/real-time use cases (e.g. with VAD-based chunking), you can use
    // sortformer.diarize_chunk(&audio_16k_mono) which preserves internal state (FIFO,
    // speaker cache) across calls for consistent speaker IDs over time.
    let speaker_segments = sortformer.diarize(audio.clone(), target_sample_rate, 2)?;

    println!(
        "Found {} speaker segments from Sortformer, {:.1}s)",
        speaker_segments.len(),
        (Instant::now() - start).as_secs(),
    );
    Ok(speaker_segments)
}
