use anyhow::Result;

const BIG_WAV_PATH: &str = "/Volumes/sw/tts_result/gcsjdls/gcsjdls-4_1.wav";
const SMALL_WAV_PATH: &str = "/Users/larry/Documents/resources/qinsheng.wav";
const GCSJDLS_16K_WAV_PATH: &str = "/Users/larry/coderesp/FireRedVAD/gcsjdls-4_1_16k.wav";

mod tests {
    use aphelios_asr::fireredvad::firered_vad_test;
    use aphelios_core::utils::core_utils;

    use super::*;

    #[test]
    fn big_test() -> Result<()> {
        core_utils::init_tracing();
        firered_vad_test(BIG_WAV_PATH, "16kHz audio (native)")
    }

    #[test]
    fn gcsjdls_16k_test() -> Result<()> {
        core_utils::init_tracing();
        firered_vad_test(GCSJDLS_16K_WAV_PATH, "16kHz audio (native)")
    }

    #[test]
    fn small_test() -> Result<()> {
        core_utils::init_tracing();
        firered_vad_test(SMALL_WAV_PATH, "16kHz audio (native)")
    }
}
