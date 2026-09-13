//! VAD (Voice Activity Detection) 测试
//!
//! 测试 Silero VAD 模型在不同采样率音频上的表现

use anyhow::Result;
use aphelios_asr::silerovad::VadProcessor;
use tracing::info;

fn init_logging() {
    let _ = tracing_subscriber::fmt::try_init();
}

const STALIN_WAV_PATH: &str = "/Volumes/sw/video/Why Does Joseph Stalin Matter？.wav";

mod tests {
    use std::path::Path;

    use aphelios_asr::{
        silerovad::run_vad_with_path,
        srt::generate_vad_srt,
    };

    use super::*;

    #[test]
    fn vad_audio_path_test() -> Result<()> {
        run_vad_with_path(STALIN_WAV_PATH, "16kHz audio (native)")
    }

    // 生成vad形式的伪字幕文件.只有时间,没有文本
    #[tokio::test]
    async fn vad_srt_check_test() -> Result<()> {
        let audio_path = STALIN_WAV_PATH;
        init_logging();
        info!("Audio file: {}", audio_path);
        let mut vad = VadProcessor::new_default(Some("/Volumes/sw/onnx_models/silero-vad/onnx"))?;
        let segments = vad.process_from_file(audio_path)?;
        info!("Detected {} speech segments", segments.len());
        for (i, segment) in segments.iter().enumerate() {
            info!("  Segment {}: {:.2}s - {:.2}s (duration: {:.2}s, avg_prob: {:.2})", i + 1, segment.start, segment.end, segment.end - segment.start, segment.avg_prob);
        }
        let output_path = Path::new(audio_path)
            .with_file_name(
                Path::new(audio_path)
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap(),
            )
            .with_extension("vad.srt")
            .to_str()
            .unwrap()
            .to_string();
        generate_vad_srt(&segments, &output_path).await?;
        Ok(())
    }
}
