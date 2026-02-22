//! DIA (Speaker Diarization) 测试

mod common;

use anyhow::Result;
use aphelios_core::dia::dia_process;
use common::{TEST_AUDIO_16K, TEST_AUDIO_44K, TestConfig, setup};
use std::fs::File;
use std::io::{BufWriter, Write};
use tracing::{error, info};

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
