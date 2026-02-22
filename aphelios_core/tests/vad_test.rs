//! VAD (Voice Activity Detection) 测试
//!
//! 测试 Silero VAD 模型在不同采样率音频上的表现

mod common;

use anyhow::Result;
use aphelios_core::vad::{VadConfig, VadDetector};
use common::{TEST_AUDIO_16K, TEST_AUDIO_44K, setup};
use tracing::info;

/// 运行 VAD 测试
fn run_vad_test(audio_path: &str, label: &str) -> Result<()> {
    setup();

    info!("=== VAD Test: {} ===", label);
    info!("Audio file: {}", audio_path);

    let mut detector = VadDetector::new(VadConfig::default())?;
    let result = detector.detect_from_file(audio_path)?;

    info!("Audio duration: {:.2}s", result.audio_duration);
    info!(
        "Speech duration: {:.2}s ({:.1}%)",
        result.total_speech_duration,
        result.speech_ratio * 100.0
    );
    info!("Segments detected: {}", result.segments.len());

    for (i, segment) in result.segments.iter().enumerate() {
        info!(
            "  Segment {}: {:.2}s - {:.2}s (prob: {:.2})",
            i + 1,
            segment.start,
            segment.end,
            segment.avg_probability
        );
    }

    assert!(
        !result.segments.is_empty(),
        "Should detect at least one speech segment"
    );
    Ok(())
}

#[test]
fn test_vad_16k() -> Result<()> {
    run_vad_test(TEST_AUDIO_16K, "16kHz audio (native)")
}

#[test]
fn test_vad_44k_to_16k() -> Result<()> {
    run_vad_test(TEST_AUDIO_44K, "44.1kHz audio (resampled to 16kHz)")
}
