use aphelios_core::{measure_time, utils::core_utils};
use aphelios_tts::{AudioBuffer, Language, Qwen3TTS};

fn main() -> anyhow::Result<()> {
    core_utils::init_tracing();
    let device = core_utils::get_default_device(false)?;

    let model_path = "/Volumes/sw/pretrained_models/Qwen3-TTS-12Hz-0.6B-Base";
    let ref_audio = "/Users/larry/coderesp/aphelios_cli/test_data/newvoice.wav";
    let ref_text = "写这本书的目的在于通过我的走访和观察";
    let text_to_speech =
        "近来研究表明，那种庞大、复杂、联合式的宗族在中国并不普遍，可能只存在于华南及江南的某些地区。
    研究者发现，北方那样的多族共居村庄在新界殖民地仍很普遍。对旧的宗族研究范式的批评甚至比这些论点还要深入。";

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
    audio.save("output/newvoice-output2.wav")?;
    // 音频21s, 合成耗时65s
    Ok(())
}
