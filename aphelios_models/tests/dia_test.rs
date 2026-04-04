use anyhow::Result;
use aphelios_asr::dia::dia_process;
use aphelios_core::utils::logger;
use std::fs::File;
use std::io::{BufWriter, Write};
use tracing::{error, info};

#[test]
fn run_dia_test() -> Result<()> {
    logger::init_logging();

    let audio_path = "/Volumes/sw/video/Stalin at War - Stephen Kotkin.wav";
    let dia_model_path =
        "/Volumes/sw/aphelios_cli_models/onnx_models/dia/diar_streaming_sortformer_4spk-v2.onnx";

    let result = dia_process(audio_path, dia_model_path);

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
