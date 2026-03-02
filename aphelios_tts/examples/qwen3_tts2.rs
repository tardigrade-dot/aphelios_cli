use aphelios_core::{measure_time, utils::core_utils};
use aphelios_tts::{AudioBuffer, Language, Qwen3TTS};

fn main() -> anyhow::Result<()> {
    core_utils::init_tracing();
    let device = core_utils::get_default_device(false)?;

    let model_path = "/Volumes/sw/pretrained_models/Qwen3-TTS-12Hz-0.6B-Base";
    let ref_audio = "/Users/larry/coderesp/aphelios_cli/test_data/newvoice.wav";
    let ref_text = "写这本书的目的在于通过我的走访和观察";
    let text_to_speech =
        "近年来的研究表明，在中国文化中“家”的含义十分广泛，学者们对其所下定义也是五花八门";

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
    audio.save("newvoice-output.wav")?;
    Ok(())
}
