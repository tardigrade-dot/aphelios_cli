use anyhow::{Ok, Result, anyhow};
use aphelios_core::{common::core_utils::get_append_filename_with_ext, dia};
use std::fs::File;
use std::io::{BufWriter, Write};
use tracing::error;

#[test]
fn dia_test() -> Result<()> {
    // Use a more generic test file or allow any WAV file
    let audio_path = std::env::var("TEST_AUDIO_PATH")
        .unwrap_or("/Users/larry/coderesp/aphelios_cli/test_data/RYdrPg6xdYo.wav".to_string());
    let model_path = "/Users/larry/coderesp/aphelios_cli/aphelios_core/onnx_models/dia/diar_streaming_sortformer_4spk-v2.onnx";

    let _ = dia::dia_process(&audio_path, model_path);
    Ok(())
}

// Additional test that can handle any WAV file
#[test]
fn dia_test_arbitrary_wav() -> Result<()> {
    let audio_path = "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4.wav";
    // This test can be run with any WAV file by setting the environment variable
    let audio_path = std::env::var("ARBITRARY_WAV_PATH").unwrap_or(audio_path.to_string());

    // Check if the file exists
    if !std::path::Path::new(&audio_path).exists() {
        eprintln!("Warning: Audio file does not exist: {}", audio_path);
        return Ok(());
    }

    let model_path = "/Users/larry/coderesp/aphelios_cli/aphelios_core/onnx_models/dia/diar_streaming_sortformer_4spk-v2.onnx";

    let dia_file = get_append_filename_with_ext(&audio_path, "_dia", ".txt");
    let res = dia::dia_process(&audio_path, model_path);

    let file = File::create(dia_file)?;
    let mut writer = BufWriter::new(file);
    match res {
        core::result::Result::Ok(ss_list) => {
            for seg in ss_list {
                let line: String = format!(
                    "[{:06.2}s - {:06.2}s] Speaker {}",
                    seg.start, seg.end, seg.speaker_id
                );
                writeln!(writer, "{}", line)?;
            }
        }
        Err(e) => {
            error!("{}", e);
        }
    }
    writer.flush()?;
    Ok(())
}
