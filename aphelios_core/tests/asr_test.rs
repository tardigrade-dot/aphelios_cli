// https://github.com/openai/whisper/blob/main/whisper/model.py/rgs
// TODO:
// - Batch size greater than 1.
// - More token filters (SuppressBlanks, ApplyTimestampRules).

use std::path::Path;

use anyhow::Result;
use aphelios_core::{
    asr::{generate_srt, run_whisper},
    utils,
};
use candle_core::Device;

use tracing::{error, info};

const ASR_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/distil-large-v3.5";
const AUDIO_16K: &str = "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4_16k.wav";
// const AUDIO_44K: &str = "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4.wav";
const AUDIO_44K: &str = "/Users/larry/coderesp/aphelios_cli/test_data/RYdrPg6xdYo_16k.wav";

fn run_asr_test(input: &str, output_suffix: &str) -> Result<()> {
    let device = Device::new_metal(0)?;

    let output_path = Path::new(input)
        .with_file_name(format!(
            "{}{}",
            Path::new(input).file_stem().unwrap().to_str().unwrap(),
            output_suffix
        ))
        .with_extension("srt")
        .to_str()
        .unwrap()
        .to_string();

    println!("正在处理音频：{}", input);
    let result = run_whisper(ASR_MODEL_DIR, input, &device);
    info!("print asr result");
    match result {
        Ok(segments) => {
            for segment in &segments {
                println!(
                    "\n[Segment: {:.2}s - {:.2}s]",
                    segment.start,
                    segment.start + segment.duration
                );

                if !segment.sub_segments.is_empty() {
                    for sub in &segment.sub_segments {
                        println!(
                            "  => [{:>6.2}s -> {:>6.2}s]  {}",
                            sub.start,
                            sub.end,
                            sub.text.trim()
                        );
                    }
                } else {
                    let clean_text = segment.dr.text.replace(
                        |c: char| c == '<' || c == '|' || c == '>' || c.is_numeric() || c == '.',
                        "",
                    );
                    if !clean_text.trim().is_empty() {
                        println!("  (Raw): {}", clean_text.trim());
                    }
                }
            }
            let _ = generate_srt(&segments, output_path.as_str());
        }
        Err(e) => {
            error!("ASR task failed: {:?}", e);
        }
    }

    Ok(())
}

#[test]
fn asr_test_16k() -> Result<()> {
    utils::init_logging();
    info!("=== ASR Test with 16kHz audio ===");
    run_asr_test(AUDIO_16K, "_asr_16k")
}

#[test]
fn asr_test_44k_to_16k() -> Result<()> {
    utils::init_logging();
    info!("=== ASR Test with 44.1kHz audio (auto resampled to 16kHz) ===");
    run_asr_test(AUDIO_44K, "_asr_44k")
}
