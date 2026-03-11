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

/// TTS 合成函数，支持自定义输出路径和参考音频
pub fn qwen3_tts_with_output(
    text: &str,
    model_path: &str,
    output_path: &str,
    ref_audio_path: Option<&str>,
    ref_text: Option<&str>,
) -> Result<()> {
    let device = core_utils::get_default_device(false)?;
    let dtype = DType::BF16;

    let loader_config: LoaderConfig = LoaderConfig {
        dtype,
        load_tokenizer: true,
        load_text_tokenizer: true,
        load_generate_config: true,
        use_flash_attn: false,
    };

    let loader = ModelLoader::from_local_dir(model_path)
        .map_err(|e| anyhow::anyhow!("Failed to create model loader: {}", e))?;

    let mut model = loader
        .load_tts_model(&device, &loader_config)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

    info!(
        model_type = ?model.model_type(),
        has_text_tokenizer = model.has_text_processor(),
        "Model details"
    );

    let synth_item = qwen_tts::io::SynthesisItem {
        text: text.to_owned(),
        language: "chinese".to_string(),
        output: Path::new(output_path).to_path_buf(),
    };

    let clone_params = qwen_tts::io::VoiceCloneParams {
        ref_audio: ref_audio_path.map(|p| Path::new(p).to_path_buf()),
        ref_text: ref_text.map(|t| t.to_string()),
        x_vector_only: ref_audio_path.is_some(),
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
        "TTS 耗时",
        synthesize_voice_clone_item(&mut model, &gen_args, &io_args, &clone_params, &synth_item)?
    );

    let du = (Instant::now() - _start).as_secs();
    info!("tts 耗时：{}秒", du);

    Ok(())
}

pub fn qwen3_tts(text_to_speech: String) -> Result<()> {
    qwen3_tts_with_output(
        &text_to_speech,
        "/Volumes/sw/pretrained_models/Qwen3-TTS-12Hz-0.6B-Base",
        "/Users/larry/coderesp/aphelios_cli/output/newvoice-BF16.wav",
        Some("/Users/larry/coderesp/aphelios_cli/test_data/newvoice.wav"),
        Some("写这本书的目的在于通过我的走访和观察"),
    )
}
