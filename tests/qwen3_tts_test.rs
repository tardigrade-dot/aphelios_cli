use std::path::Path;

use anyhow::Result;
use anyhow::bail;
use candle_core::{DType, Device};
use qwen_tts::io::batch_items::load_batch_items;
use qwen_tts::io::model_path::get_model_path;
use qwen_tts::io::output_path::get_output_path;
use qwen_tts::io::{GenerationArgs, IoArgs};
use qwen_tts::model::loader::{LoaderConfig, ModelLoader};
use qwen_tts::nn::mt_rng::set_seed;
use qwen_tts::synthesis::detect_mode::{DetectedMode, determine_mode};
use qwen_tts::synthesis::synthesize_voice::{
    synthesize_custom_voice_item, synthesize_voice_clone_item, synthesize_voice_design_item,
};
use qwen_tts::synthesis::tokenizer::{TokenizerCommand, run_tokenizer};

#[test]
fn qwen3_tts_test() -> Result<()> {
    let device = Device::new_metal(0).unwrap();
    let mut dtype = DType::BF16;

    let loader_config: LoaderConfig = LoaderConfig {
        dtype,
        load_tokenizer: true,
        load_text_tokenizer: true,
        load_generate_config: true,
        use_flash_attn: false,
    };

    let model_path = "/Volumes/sw/pretrained_models/Qwen3-TTS-12Hz-0.6B-Base";
    let loader = ModelLoader::from_local_dir(&model_path)
        .map_err(|e| anyhow::anyhow!("Failed to create model loader: {}", e))?;

    let mut model = loader
        .load_tts_model(&device, &loader_config)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

    println!("Model loaded!");
    tracing::info!(
        model_type = ?model.model_type(),
        has_text_tokenizer = model.has_text_processor(),
        "Model details"
    );

    let synth_item = qwen_tts::io::SynthesisItem {
        text: "吉拉斯事件是国际共运史上的一个重要事件".to_string(),
        language: "chinese".to_string(),
        output: Path::new("/Users/larry/coderesp/aphelios_cli/output/bbb.wav").to_path_buf(),
    };

    let clone_params = qwen_tts::io::VoiceCloneParams {
        ref_audio: Some(Path::new("/Volumes/sw/MyDrive/data_src/youyi-15s.wav").to_path_buf()),
        ref_text: Some("不过事实上也不完全如此,从政治体制的角度看,他们俩都在为君主制寻找新的基础.或者说都希望出现一种不同于以往的新的君主制.".into()),
        x_vector_only: false,
        save_prompt: false
    };

    let gen_args = GenerationArgs {
        max_tokens: 2048,
        temperature: Option::None,
        top_k: Option::None,
        top_p: Option::None,
        repetition_penalty: Option::None,
        seed: Option::None,
        greedy: false,
        subtalker_temperature: Option::None,
        subtalker_top_k: Option::None,
        subtalker_top_p: Option::None,
        no_subtalker_sample: true,
    };
    let io_args = IoArgs {
        output: Path::new("/Users/larry/coderesp/aphelios_cli/output/aaa.wav").to_path_buf(),
        file: Option::None,
        output_dir: Option::None,
        save_prompt: Option::None,
        load_prompt: Option::None,
        debug: true,
        tracing: true,
    };
    let _ =
        synthesize_voice_clone_item(&mut model, &gen_args, &io_args, &clone_params, &synth_item)?;
    Ok(())
}
