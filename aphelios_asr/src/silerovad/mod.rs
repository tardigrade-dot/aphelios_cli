//! VAD (Voice Activity Detection) 模块
//!
//! 使用 Silero VAD 模型进行语音活动检测

use anyhow::Result;
use aphelios_core::utils::init_logging;
use tracing::info;

pub mod detector;
pub mod types;
pub mod vadprocess;

pub use detector::{VadConfig, VadDetector};
pub use types::VadResult;
pub use vadprocess::{VadConfig as VadProcessorConfig, VadProcessor};

pub fn run_vad_with_path(audio_path: &str, label: &str) -> Result<()> {
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
            "  Segment {}: {:.2}s - {:.2}s (avg_prob: {:.2})",
            i + 1,
            segment.start,
            segment.end,
            segment.avg_prob
        );
    }

    assert!(
        !result.segments.is_empty(),
        "Should detect at least one speech segment"
    );
    Ok(())
}
