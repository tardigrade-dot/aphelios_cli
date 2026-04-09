use anyhow::{Context, Result};
use aphelios_tts::longcat_audiodit::{
    GuidanceMethod, LongCatAudioDiT, LongCatInferenceConfig, LongCatPromptDebugOutputs,
    LongCatSynthesisRequest,
};
use candle_core::{DType, Tensor};
use clap::Parser;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/LongCat-AudioDiT-1B";
const DEFAULT_TOKENIZER: &str = "/Volumes/sw/pretrained_models/umt5-base/tokenizer.json";

#[derive(Debug, Parser)]
#[command(name = "longcat-prompt-dump")]
struct Args {
    #[arg(long)]
    text: String,
    #[arg(long = "prompt_text", alias = "prompt-text")]
    prompt_text: Option<String>,
    #[arg(long = "prompt_audio", alias = "prompt-audio")]
    prompt_audio: PathBuf,
    #[arg(long = "dump_dir", alias = "dump-dir")]
    dump_dir: PathBuf,
    #[arg(long = "model_dir", alias = "model-dir", default_value = DEFAULT_MODEL_DIR)]
    model_dir: PathBuf,
    #[arg(long = "tokenizer_path", alias = "tokenizer-path", default_value = DEFAULT_TOKENIZER)]
    tokenizer_path: Option<PathBuf>,
    #[arg(long, default_value_t = 1024)]
    seed: u64,
    #[arg(long)]
    cpu: bool,
}

#[derive(Debug, Serialize)]
struct TensorManifestEntry {
    dtype: String,
    shape: Vec<usize>,
    file: String,
}

#[derive(Debug, Serialize)]
struct DumpManifest {
    tensors: BTreeMap<String, TensorManifestEntry>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let model = LongCatAudioDiT::from_pretrained(
        &args.model_dir,
        LongCatInferenceConfig {
            cpu: args.cpu,
            tokenizer_path: args.tokenizer_path.clone(),
            dtype: None,
        },
    )?;

    let request = LongCatSynthesisRequest {
        text: args.text,
        prompt_text: args.prompt_text,
        prompt_audio: Some(args.prompt_audio),
        duration: None,
        steps: 16,
        cfg_strength: 4.0,
        guidance_method: GuidanceMethod::Cfg,
        seed: args.seed,
    };

    let debug = model.collect_prompt_debug_tensors(&request, None)?;
    write_dump(&args.dump_dir, &debug)?;
    println!("Dumped LongCat prompt tensors to {}", args.dump_dir.display());
    Ok(())
}

fn write_dump(dir: &Path, debug: &LongCatPromptDebugOutputs) -> Result<()> {
    fs::create_dir_all(dir)
        .with_context(|| format!("failed to create dump dir {}", dir.display()))?;

    let mut manifest = DumpManifest {
        tensors: BTreeMap::new(),
    };
    write_tensor(
        dir,
        &mut manifest,
        "prompt_audio_padded",
        &debug.prompt_audio_padded,
    )?;
    write_tensor(
        dir,
        &mut manifest,
        "prompt_latent_vae_output",
        &debug.prompt_latent_vae_output,
    )?;
    write_tensor(dir, &mut manifest, "prompt_latent", &debug.prompt_latent)?;

    let manifest_path = dir.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?).with_context(|| {
        format!(
            "failed to write prompt dump manifest {}",
            manifest_path.display()
        )
    })?;
    Ok(())
}

fn write_tensor(
    dir: &Path,
    manifest: &mut DumpManifest,
    name: &str,
    tensor: &Tensor,
) -> Result<()> {
    let tensor = tensor.to_device(&candle_core::Device::Cpu)?;
    let tensor = tensor.to_dtype(DType::F32)?;
    let shape = tensor.shape().dims().to_vec();
    let file = format!("{name}.bin");
    let path = dir.join(&file);
    let values = tensor.flatten_all()?.to_vec1::<f32>()?;
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(&path, bytes)
        .with_context(|| format!("failed to write prompt tensor {}", path.display()))?;
    manifest.tensors.insert(
        name.to_string(),
        TensorManifestEntry {
            dtype: "f32".to_string(),
            shape,
            file,
        },
    );
    Ok(())
}
