pub mod qwen_tts_infer;

use anyhow::Result;
use candle_core::{DType, Device};
use qwen_tts::io::{GenerationArgs, IoArgs};
use qwen_tts::model::loader::{LoaderConfig, ModelLoader};
use qwen_tts::synthesis::synthesize_voice::synthesize_voice_clone_item;
use std::path::Path;
use std::time::Instant;
use tracing::info;

pub fn qwen3_tts(text_to_speech: String) -> Result<()> {
    let device: Device = Device::new_metal(0).unwrap();
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

    let synth_item = qwen_tts::io::SynthesisItem {
        text: text_to_speech.to_owned(),
        language: "chinese".to_string(),
        output: Path::new("/Users/larry/coderesp/aphelios_cli/output/bbb2.wav").to_path_buf(),
    };

    let clone_params = qwen_tts::io::VoiceCloneParams {
        ref_audio: Some(Path::new("/Volumes/sw/MyDrive/data_src/youyi-15s.wav").to_path_buf()),
        ref_text: Some("不过事实上也不完全如此,从政治体制的角度看,他们俩都在为君主制寻找新的基础.或者说都希望出现一种不同于以往的新的君主制.".into()),
        x_vector_only: false,
        save_prompt: false
    };

    // let clone_params = qwen_tts::io::VoiceCloneParams {
    //     ref_audio: Some(Path::new("/Volumes/sw/MyDrive/data_src/qinsheng.wav").to_path_buf()),
    //     ref_text: Some("写这本书的目的在于通过我的走访和观察,了解中东问题的发生和发展过程,但本书并不赞同把中东问题归结为一两个简单的范式:宗教矛盾、民主问题、经济原因,都无法解释中东发生的一切.".into()),
    //     x_vector_only: false,
    //     save_prompt: false
    // };

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
    // let io_args = IoArgs {
    //     output: Path::new("/Users/larry/coderesp/aphelios_cli/output/aaa.wav").to_path_buf(),
    //     file: Option::None,
    //     output_dir: Option::None,
    //     save_prompt: Option::None,
    //     load_prompt: Option::None,
    //     debug: true,
    //     tracing: true,
    // };
    let _start = Instant::now();
    let _ =
        synthesize_voice_clone_item(&mut model, &gen_args, &io_args, &clone_params, &synth_item)?;

    let du = (Instant::now() - _start).as_secs();
    info!("tts 耗时: {}秒", du);

    Ok(())
}
