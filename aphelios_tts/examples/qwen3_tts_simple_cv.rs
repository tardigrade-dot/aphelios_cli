use aphelios_core::utils::core_utils;
use aphelios_tts::{Language, Qwen3TTS, Speaker};

fn main() -> anyhow::Result<()> {
    core_utils::init_tracing();

    let device = core_utils::get_default_device(false)?;

    let model_path = "/Volumes/sw/pretrained_models/Qwen3-TTS-12Hz-1.7B-CustomVoice";
    let text_to_speech = "你好，这是语音克隆测试";

    tracing::info!("加载模型...");
    let model = Qwen3TTS::from_pretrained(model_path, device)?;

    let speaks = vec![
        Speaker::Ryan,
        Speaker::Aiden,
        Speaker::UncleFu,
        Speaker::Serena, //较适合中文
        Speaker::Sohee,
    ];
    for speaker in speaks {
        let audio =
            model.synthesize_with_voice(text_to_speech, speaker, Language::Chinese, None)?;
        audio.save(format!("output/CustomVoice-output-{}.wav", speaker))?;
    }

    Ok(())
}
