//! VAD (Voice Activity Detection) 测试
//!
//! 测试 Silero VAD 模型在不同采样率音频上的表现

use anyhow::Result;
use aphelios_asr::silerovad::VadProcessor;
use aphelios_core::utils::init_logging;
use tracing::info;

const BIG_WAV_PATH: &str = "/Volumes/sw/tts_result/gcsjdls/gcsjdls-4_1.wav";
const SMALL_WAV_PATH: &str = "/Volumes/sw/video/mQlxALUw3h4_16k.wav";
static STALIN_WAV_PATH: &str = "/Volumes/sw/video/Why Does Joseph Stalin Matter？.wav";

mod tests {
    use std::path::Path;

    use aphelios_asr::{
        silerovad::run_vad_with_path,
        whisper::{generate_srt, run_whisper_with_segments},
    };

    use super::*;

    #[test]
    fn vad_big_test() -> Result<()> {
        run_vad_with_path(BIG_WAV_PATH, "16kHz audio (native)")
    }

    #[test_log::test]
    fn vad_small_test() -> Result<()> {
        run_vad_with_path(SMALL_WAV_PATH, "16kHz audio (native)")
    }

    #[test_log::test]
    fn vad_audio_path_test() -> Result<()> {
        run_vad_with_path(STALIN_WAV_PATH, "16kHz audio (native)")
    }

    // 873.35s
    #[tokio::test]
    async fn vad_and_whisper_test() -> Result<()> {
        init_logging();
        info!("Audio file: {}", SMALL_WAV_PATH);
        let mut vad = VadProcessor::new_default()?;
        let segments = vad.process_from_file(SMALL_WAV_PATH)?;
        info!("Detected {} speech segments", segments.len());
        for (i, segment) in segments.iter().enumerate() {
            info!(
                "  Segment {}: {:.2}s - {:.2}s (duration: {:.2}s, avg_prob: {:.2})",
                i + 1,
                segment.start,
                segment.end,
                segment.end - segment.start,
                segment.avg_prob
            );
        }

        assert!(
            !segments.is_empty(),
            "Should detect at least one speech segment"
        );
        let batches = vad.aggregate_segments(&segments, 30.0, 0.3);
        info!("Aggregated {} batches", batches.len());
        for (i, batch) in batches.iter().enumerate() {
            info!(
                "  Batch {}: {:.2}s - {:.2}s (duration: {:.2}s, segments_count: {})",
                i + 1,
                batch.start,
                batch.end,
                batch.duration,
                batch.segments_count
            );
        }
        const ASR_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/distil-large-v3.5";

        info!("Running whisper with segments ...");
        let res = run_whisper_with_segments(ASR_MODEL_DIR, SMALL_WAV_PATH, batches, Some("en"))?;
        for (i, segment) in res.iter().enumerate() {
            for (j, sub) in segment.sub_segments.iter().enumerate() {
                let start = segment.start + sub.start;
                let end = segment.start + sub.end;
                let duration = sub.end - sub.start;
                info!(
                    "  Segment {}.{}: {:.2}s - {:.2}s (duration: {:.2}s, text: {})",
                    i + 1,
                    j + 1,
                    start,
                    end,
                    duration,
                    sub.text
                );
            }
        }
        let output_path = Path::new(SMALL_WAV_PATH)
            .with_file_name(
                Path::new(SMALL_WAV_PATH)
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap(),
            )
            .with_extension("srt")
            .to_str()
            .unwrap()
            .to_string();
        generate_srt(&res, &output_path).await?;
        Ok(())
    }
}
