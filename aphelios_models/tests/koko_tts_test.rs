use anyhow::Result;
use aphelios_core::{audio::MonoBuffer, AudioSaver};
use aphelios_tts::kokoro::{KokoroLanguage, KokoroModel, KokoroVoices};
use std::fs;
use std::path::Path;

const OUTPUT_DIR: &str = "/Users/larry/coderesp/aphelios_cli/output";

#[test]
fn test_koko_tts_full_pipeline() -> Result<()> {
    let model_dir = "/Volumes/sw/onnx_models/Kokoro-82M-v1.0-ONNX";
    if !Path::new(model_dir).exists() {
        return Ok(());
    }

    // Ensure output directory exists
    fs::create_dir_all(OUTPUT_DIR)?;

    let mut model = KokoroModel::new(model_dir)?;

    // Test cases: (Voice, Text, Language, Filename)
    let test_cases = vec![
        (
            "af_jessica",
            "Hello world. This is a British voice using misaki-rs.",
            KokoroLanguage::EnUs,
            format!("{}/koko_full_us.wav", OUTPUT_DIR),
        ),
        (
            "bm_george",
            "Hello world. This is a cold test of the full pipeline.",
            KokoroLanguage::EnGb,
            format!("{}/koko_full_uk.wav", OUTPUT_DIR),
        ),
    ];

    for (voice_name, text, lang, filename) in test_cases {
        let voice_path = format!("{}/voices/{}.bin", model_dir, voice_name);
        if !Path::new(&voice_path).exists() {
            continue;
        }

        let voices = KokoroVoices::from_file(&voice_path)?;
        let style = voices.get_style(0).unwrap();

        println!("Generating full pipeline audio for {}...", voice_name);
        let audio_data = model.generate(text, lang, style, 1.0)?;

        let buffer = MonoBuffer::new(audio_data.to_vec(), 24000);
        AudioSaver::new().save_mono(&buffer, &filename)?;

        println!("Successfully generated and saved to {}", &filename);
        assert!(Path::new(&filename).exists());
    }

    Ok(())
}

#[test]
fn test_koko_tts_speed_variation_ipa() -> Result<()> {
    let model_dir = "/Volumes/sw/onnx_models/Kokoro-82M-v1.0-ONNX";
    if !Path::new(model_dir).exists() {
        return Ok(());
    }

    let mut model = KokoroModel::new(model_dir)?;
    let voices = KokoroVoices::from_file(format!("{}/voices/af_nicole.bin", model_dir))?;
    let style = voices.get_style(0).unwrap();

    // "Life is like a box of chocolates."
    let ipa = "laɪf ɪz laɪk ə bɒks ɒv tʃɒkləts.";

    println!("Testing generation at 0.5x speed (IPA)...");
    let slow_audio = model.generate_from_ipa(ipa, style, 0.5)?;

    println!("Testing generation at 2.0x speed (IPA)...");
    let fast_audio = model.generate_from_ipa(ipa, style, 2.0)?;

    assert!(slow_audio.len() > fast_audio.len());

    fs::create_dir_all("/Users/larry/coderesp/aphelios_cli/output")?;
    AudioSaver::new().save_mono(
        &MonoBuffer::new(slow_audio.to_vec(), 24000),
        "/Users/larry/coderesp/aphelios_cli/output/koko_slow_ipa.wav",
    )?;
    AudioSaver::new().save_mono(
        &MonoBuffer::new(fast_audio.to_vec(), 24000),
        "/Users/larry/coderesp/aphelios_cli/output/koko_fast_ipa.wav",
    )?;

    Ok(())
}
