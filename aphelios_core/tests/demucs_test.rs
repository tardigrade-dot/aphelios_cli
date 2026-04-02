use aphelios_core::{
    demucs::{run_demucs, run_vocal_separation},
    init_logging,
};

const AUDIO_16K: &str = "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4_16k.wav";
const AUDIO_PATH: &str = "/Volumes/sw/video/mQlxALUw3h4.wav";

const MODEL_DIR: &str = "/Volumes/sw/aphelios_cli_models/onnx_models/demucs";

#[test]
fn test_demucs_separation_16k_to_44k() {
    run_demucs(MODEL_DIR, AUDIO_16K, None).expect("Demucs separation failed");
}

#[test]
fn test_vocal_separation_44k() {
    init_logging();
    run_vocal_separation(MODEL_DIR, AUDIO_PATH, None).expect("Vocal separation failed");
}
