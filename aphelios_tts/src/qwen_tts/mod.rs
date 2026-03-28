pub mod qwen_tts_v2;

use anyhow::Result;
use aphelios_core::measure_time;
use aphelios_core::utils::base;
use candle_core::{DType, Device};
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
    // 使用 Metal 设备（如果可用）
    let device = base::get_default_device(false)?;

    // 根据设备选择合适的数据类型
    // Metal 上 BF16 可能不如 F32 稳定，CPU 上必须用 F32
    let dtype = match &device {
        Device::Metal(_) => DType::F32, // Metal 上用 F32 更稳定
        Device::Cpu => DType::F32,
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => DType::BF16, // CUDA 上可以用 BF16 加速
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => DType::F32, // 如果没有 CUDA 特性，回退到 F32
    };

    info!("TTS 推理设备：{:?}, 数据类型：{:?}", device, dtype);

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
        temperature: Some(0.8), // 添加温度参数，增加随机性
        top_k: Some(50),
        top_p: Some(0.95),
        repetition_penalty: Some(1.1),
        seed: Option::None,
        greedy: false,
        subtalker_temperature: Some(0.8),
        subtalker_top_k: Some(50),
        subtalker_top_p: Some(0.95),
        no_subtalker_sample: false, // 允许子说话人采样，增加自然度
    };
    let io_args = IoArgs::default();

    info!("开始 TTS 合成任务...");
    info!("  - 文本：{}", text);
    info!("  - 模型：{}", model_path);
    info!("  - 输出：{}", output_path);
    if let Some(ref_audio) = ref_audio_path {
        info!("  - 参考音频：{}", ref_audio);
    }
    if let Some(ref_txt) = ref_text {
        info!("  - 参考文本：{}", ref_txt);
    }

    let _start = Instant::now();
    let _ = measure_time!(
        "TTS 任务",
        synthesize_voice_clone_item(&mut model, &gen_args, &io_args, &clone_params, &synth_item)?
    );

    let du = (Instant::now() - _start).as_secs();
    info!("TTS 总耗时：{}秒", du);

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
