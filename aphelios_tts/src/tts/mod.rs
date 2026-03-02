use anyhow::Result;
use aphelios_core::measure_time;
use aphelios_core::utils::core_utils;
use candle_core::DType;
use qwen_tts::io::{GenerationArgs, IoArgs};
use qwen_tts::model::loader::{LoaderConfig, ModelLoader};
use qwen_tts::synthesis::synthesize_voice::synthesize_voice_clone_item;
use std::path::Path;
use std::time::Instant;
use tracing::info;

pub fn qwen3_tts(text_to_speech: String) -> Result<()> {
    let device = core_utils::get_default_device(false)?;
    let dtype = DType::BF16;

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

    info!(
        model_type = ?model.model_type(),
        has_text_tokenizer = model.has_text_processor(),
        "Model details"
    );

    let synth_item1 = qwen_tts::io::SynthesisItem {
        text: text_to_speech.to_owned(),
        language: "chinese".to_string(),
        output: Path::new("/Users/larry/coderesp/aphelios_cli/output/newvoice-12.wav")
            .to_path_buf(),
    };

    let synth_item2 = qwen_tts::io::SynthesisItem {
        text: text_to_speech.to_owned(),
        language: "chinese".to_string(),
        output: Path::new("/Users/larry/coderesp/aphelios_cli/output/newvoice-22.wav")
            .to_path_buf(),
    };

    let clone_params = qwen_tts::io::VoiceCloneParams {
        ref_audio: Some(
            Path::new("/Users/larry/coderesp/aphelios_cli/test_data/newvoice.wav").to_path_buf(),
        ),
        ref_text: Some("写这本书的目的在于通过我的走访和观察".into()),
        x_vector_only: false,
        save_prompt: false,
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
    let io_args = IoArgs::default();
    let _start = Instant::now();
    let _ = measure_time!(
        "TTS 耗时1",
        synthesize_voice_clone_item(&mut model, &gen_args, &io_args, &clone_params, &synth_item1)?
    );

    let _ = measure_time!(
        "TTS 耗时2",
        synthesize_voice_clone_item(&mut model, &gen_args, &io_args, &clone_params, &synth_item2)?
    );

    let du = (Instant::now() - _start).as_secs();
    info!("tts 耗时: {}秒", du);

    Ok(())
}
