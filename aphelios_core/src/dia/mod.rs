use std::time::Instant;

use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};

use anyhow::{Ok, Result};
use crate::common::core_utils::load_and_resample_audio;

pub fn dia_process(audio_path: &str, model_path: &str) -> Result<()> {
    let target_sample_rate = 16000;
    // 使用一站式加载函数 (处理了位深转换、声道对齐、重采样)
    let audio_data = load_and_resample_audio(audio_path, target_sample_rate)?;

    // audio_data 结构为 [Left_Channel, Right_Channel]，转换为交错格式
    let audio: Vec<f32> = audio_data[0]
        .iter()
        .zip(audio_data[1].iter())
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

    // Print raw diarization segments
    println!("\nRaw diarization segments:");
    for seg in &speaker_segments {
        println!(
            "  [{:06.2}s - {:06.2}s] Speaker {}",
            seg.start, seg.end, seg.speaker_id
        );
    }
    Ok(())
}
