use std::time::Instant;

use anyhow::Result;
use aphelios_core::AudioLoader;
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer, SpeakerSegment};

pub fn dia_process(audio_path: &str, model_path: &str) -> Result<Vec<SpeakerSegment>> {
    let target_sample_rate = 16000;

    // 加载音频
    let audio = AudioLoader::new().load(audio_path)?;
    let mono = audio.into_mono();
    let duration = mono.len() as f32 / target_sample_rate as f32 / 2.0;
    println!(
        "Loaded {} samples ({} Hz, 2 channels, {:.1}s)",
        mono.len(),
        target_sample_rate,
        duration
    );

    println!("{}", "=".repeat(80));
    println!("Step 2/3: Performing speaker diarization with Sortformer v2 (streaming)...");

    let mut sortformer = Sortformer::with_config(
        model_path,
        None, // default exec config
        DiarizationConfig::callhome(),
    )?;

    let start = Instant::now();
    let speaker_segments = sortformer.diarize(mono.samples(), target_sample_rate, 1)?;

    println!(
        "Found {} speaker segments from Sortformer, {:.1}s)",
        speaker_segments.len(),
        (Instant::now() - start).as_secs(),
    );
    Ok(speaker_segments)
}
