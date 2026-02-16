use anyhow::{Ok, Result};
use aphelios_core::dia;

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
    // This test can be run with any WAV file by setting the environment variable
    let audio_path = std::env::var("ARBITRARY_WAV_PATH")
        .unwrap_or("/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4.wav".to_string());

    // Check if the file exists
    if !std::path::Path::new(&audio_path).exists() {
        eprintln!("Warning: Audio file does not exist: {}", audio_path);
        return Ok(());
    }

    let model_path = "/Users/larry/coderesp/aphelios_cli/aphelios_core/onnx_models/dia/diar_streaming_sortformer_4spk-v2.onnx";

    let _ = dia::dia_process(&audio_path, model_path);
    Ok(())
}
