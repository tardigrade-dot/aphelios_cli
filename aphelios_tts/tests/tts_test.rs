use anyhow::Result;
use aphelios_core::utils::logger;
use aphelios_tts::qwen_tts::qwen_tts_v2::generate_voice;

#[test]
fn qwen_tts_test() -> Result<()> {
    logger::init_logging();
    let model_path = "/Volumes/sw/pretrained_models/Qwen3-TTS-12Hz-0.6B-Base";
    let ref_audio = "/Users/larry/Documents/resources/qinsheng-4s-enhanced.wav";
    let ref_text = "写这本书的目的在于通过我的走访和观察";
    let text_to_speech = "你可以重新运行构建脚本查看效果";
    let output_path = "/Users/larry/coderesp/aphelios_cli/output/aaa.wav";

    let _ = generate_voice(
        model_path,
        ref_audio,
        ref_text,
        text_to_speech,
        output_path,
        None,
    );
    Ok(())
}
