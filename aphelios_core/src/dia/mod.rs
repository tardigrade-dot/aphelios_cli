use std::time::Instant;

use parakeet_rs::sortformer::{DiarizationConfig, Sortformer, SpeakerSegment};

use crate::audio::{AudioLoader, Resampler, ResampleQuality, StereoBuffer};
use anyhow::{Ok, Result};

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
    let audio: Vec<f32> = stereo.left
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
