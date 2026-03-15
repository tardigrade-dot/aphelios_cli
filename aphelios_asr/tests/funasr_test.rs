use anyhow::Result;
use aphelios_asr::funasr::funasr_infer;
use aphelios_core::utils::core_utils;

#[test]
fn test_funasr_big() -> Result<()> {
    core_utils::init_logging();
    let big_wav_path = "/Users/larry/Documents/resources/gcsjdls-1_0.wav";
    funasr_infer(big_wav_path)
}

#[test]
fn test_funasr_small() -> Result<()> {
    core_utils::init_logging();
    let small_zh_wav_path = "/Users/larry/Documents/resources/qinsheng.wav";
    funasr_infer(small_zh_wav_path)
}
