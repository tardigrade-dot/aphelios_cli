//! VAD (Voice Activity Detection) 测试
//!
//! 测试 Silero VAD 模型在不同采样率音频上的表现

use anyhow::Result;
use aphelios_asr::vad::{VadConfig, VadDetector};
use aphelios_core::utils::init_logging;
use tracing::info;

const BIG_WAV_PATH: &str = "/Volumes/sw/tts_result/gcsjdls/gcsjdls-4_1.wav";
const SMALL_WAV_PATH: &str = "/Users/larry/Documents/resources/qinsheng.wav";

mod tests {
    use super::*;

    #[test]
    fn vad_big_test() -> Result<()> {
        run_vad_test(BIG_WAV_PATH, "16kHz audio (native)")
    }

    #[test_log::test]
    fn vad_small_test() -> Result<()> {
        run_vad_test(SMALL_WAV_PATH, "16kHz audio (native)")
    }
}

fn run_vad_test(audio_path: &str, label: &str) -> Result<()> {
    init_logging();

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
            "  Segment {}: {:.2}s - {:.2}s (duration: {:.2}s, avg_prob: {:.2})",
            i + 1,
            segment.start,
            segment.end,
            segment.duration,
            segment.avg_probability
        );
    }

    assert!(
        !result.segments.is_empty(),
        "Should detect at least one speech segment"
    );
    Ok(())
}
