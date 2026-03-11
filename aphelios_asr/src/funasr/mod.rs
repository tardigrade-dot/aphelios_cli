pub mod audio_adaptor;
pub mod audio_encoder;
pub mod frontend;
pub mod llm;
pub mod pipeline;
use crate::funasr::pipeline::FunASR;
use anyhow::Result;
use aphelios_core::utils::core_utils;
use candle_core::Tensor;
use candle_nn::VarBuilder;

pub fn funasr_infer(wav_path: &str) -> Result<()> {
    let safetensors_path = "/Volumes/sw/pretrained_models/Fun-ASR-Nano-2512/model.safetensors";

    if !std::path::Path::new(safetensors_path).exists() || !std::path::Path::new(wav_path).exists()
    {
        println!("Skipping test: model or wav file not found.");
        return Ok(());
    }

    let device = core_utils::get_default_device(false)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[safetensors_path], candle_core::DType::F32, &device)?
    };

    let mut model = FunASR::load(vb, "/Volumes/sw/pretrained_models/Fun-ASR-Nano-2512")?;

    // Load WAV
    use aphelios_core::audio::{AudioBuffer, AudioLoader};
    let loader = AudioLoader::new().with_normalize(true);
    let audio = loader.load(wav_path)?;

    let sample_rate = audio.sample_rate();
    let pcm = match audio {
        AudioBuffer::Mono(m) => {
            let len = m.samples.len();
            Tensor::from_vec(m.samples, (1, len), &device)?
        }
        AudioBuffer::Stereo(s) => {
            // Use left channel for mono ASR
            let len = s.left.len();
            Tensor::from_vec(s.left, (1, len), &device)?
        }
    };

    println!("Starting inference on {}...", wav_path);
    let result = model.inference(&pcm, sample_rate, &device)?;
    println!("Transcription: {}", result.text);
    for entry in result.characters.iter().take(20) {
        println!(
            "  char: {}, start: {:.3}, end: {:.3}, score: {:.3}",
            entry.char, entry.start, entry.end, entry.score
        );
    }
    Ok(())
}
