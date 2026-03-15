use aphelios_core::{measure_time, utils::core_utils};
use aphelios_tts::{AudioBuffer, Language, Qwen3TTS};

fn main() -> anyhow::Result<()> {
    core_utils::init_logging();
    let device = core_utils::get_default_device(false)?;

    let model_path = "/Volumes/sw/pretrained_models/Qwen3-TTS-12Hz-0.6B-Base";
    // let ref_audio = "/Users/larry/coderesp/aphelios_cli/test_data/newvoice.wav";
    let ref_audio = "/Users/larry/Documents/resources/qinsheng-4s-enhanced.wav";
    let ref_text = "写这本书的目的在于通过我的走访和观察";
    let text_to_speech = "同樣是在北京協和醫院裡，半個世紀後發生了另一件同樣深富象徵性但意義全然不同的事件。一九七一年，《紐約時報》記者雷斯頓在尼克森總統歷史性的中國行之前參與一支先遣團隊造訪中國，卻在北京接受一個緊急的闌尾切除手術。";

    let model = Qwen3TTS::from_pretrained(model_path, device)?;

    // Load reference audio
    let ref_audio = AudioBuffer::load(ref_audio)?;

    // ICL mode: full voice cloning with reference text
    let prompt = model.create_voice_clone_prompt(&ref_audio, Some(ref_text))?;

    // x_vector_only: faster, speaker embedding only (no reference text needed)
    // let prompt = model.create_voice_clone_prompt(&ref_audio, None)?;

    let audio = measure_time!(
        "语音合成",
        model.synthesize_voice_clone(text_to_speech, &prompt, Language::Chinese, None)?
    );
    audio.save("output/test3-output2.wav")?;
    // 音频21s, 合成耗时65s
    Ok(())
}
