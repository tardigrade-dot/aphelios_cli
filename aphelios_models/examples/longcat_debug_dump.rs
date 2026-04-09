use anyhow::{bail, Context, Result};
use aphelios_tts::longcat_audiodit::{
    GuidanceMethod, LongCatAudioDiT, LongCatDebugOutputs, LongCatDebugOverrides,
    LongCatInferenceConfig, LongCatSynthesisRequest,
};
use candle_core::{DType, Tensor};
use clap::Parser;
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/LongCat-AudioDiT-1B";
const DEFAULT_TOKENIZER: &str = "/Volumes/sw/pretrained_models/umt5-base/tokenizer.json";

#[derive(Debug, Parser)]
#[command(name = "longcat-debug-dump")]
struct Args {
    #[arg(long)]
    text: String,
    #[arg(long = "prompt_text", alias = "prompt-text")]
    prompt_text: Option<String>,
    #[arg(long = "prompt_audio", alias = "prompt-audio")]
    prompt_audio: Option<PathBuf>,
    #[arg(long = "dump_dir", alias = "dump-dir")]
    dump_dir: PathBuf,
    #[arg(long = "model_dir", alias = "model-dir", default_value = DEFAULT_MODEL_DIR)]
    model_dir: PathBuf,
    #[arg(long = "tokenizer_path", alias = "tokenizer-path", default_value = DEFAULT_TOKENIZER)]
    tokenizer_path: Option<PathBuf>,
    #[arg(long, default_value_t = 16)]
    nfe: usize,
    #[arg(
        long = "guidance_strength",
        alias = "guidance-strength",
        default_value_t = 4.0
    )]
    guidance_strength: f64,
    #[arg(
        long = "guidance_method",
        alias = "guidance-method",
        default_value = "cfg"
    )]
    guidance_method: String,
    #[arg(long, default_value_t = 1024)]
    seed: u64,
    #[arg(long)]
    duration: Option<usize>,
    #[arg(long)]
    cpu: bool,
    #[arg(long = "override_from", alias = "override-from")]
    override_from: Option<PathBuf>,
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
    scalars: BTreeMap<String, usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let guidance_method = GuidanceMethod::from_cli(&args.guidance_method)?;

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
        prompt_audio: args.prompt_audio,
        duration: args.duration,
        steps: args.nfe,
        cfg_strength: args.guidance_strength,
        guidance_method,
        seed: args.seed,
    };

    let override_storage = if let Some(path) = args.override_from.as_ref() {
        Some(load_overrides(path, &model.device)?)
    } else {
        None
    };
    let debug = model.collect_debug_tensors(&request, override_storage.as_ref())?;
    write_dump(&args.dump_dir, &debug)?;
    println!(
        "Dumped LongCat debug tensors to {}",
        args.dump_dir.display()
    );
    Ok(())
}

fn load_overrides(path: &Path, device: &candle_core::Device) -> Result<LongCatDebugOverrides> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read override dump {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("failed to parse safetensors {}", path.display()))?;

    Ok(LongCatDebugOverrides {
        text_condition: load_optional_tensor(&tensors, "text_condition", device)?,
        prompt_latent: load_optional_tensor(&tensors, "prompt_latent", device)?,
        y0: load_optional_tensor(&tensors, "y0", device)?,
        total_frames: load_optional_usize(&tensors, "duration")?,
    })
}

fn load_optional_tensor(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &candle_core::Device,
) -> Result<Option<Tensor>> {
    let Ok(view) = tensors.tensor(name) else {
        return Ok(None);
    };
    let shape = view.shape().to_vec();
    let tensor = match view.dtype() {
        SafeDtype::F32 => {
            let values = view
                .data()
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            Tensor::from_vec(values, shape, device)?
        }
        SafeDtype::I64 => {
            let values = view
                .data()
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            Tensor::from_vec(values, shape, device)?
        }
        SafeDtype::I32 => {
            let values = view
                .data()
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            Tensor::from_vec(values, shape, device)?
        }
        other => bail!("unsupported override tensor dtype for {name}: {other:?}"),
    };
    Ok(Some(tensor))
}

fn load_optional_usize(tensors: &SafeTensors<'_>, name: &str) -> Result<Option<usize>> {
    let Ok(view) = tensors.tensor(name) else {
        return Ok(None);
    };
    let value = match view.dtype() {
        SafeDtype::I64 => i64::from_le_bytes(view.data()[0..8].try_into().unwrap()) as usize,
        SafeDtype::I32 => i32::from_le_bytes(view.data()[0..4].try_into().unwrap()) as usize,
        SafeDtype::U32 => u32::from_le_bytes(view.data()[0..4].try_into().unwrap()) as usize,
        other => bail!("unsupported scalar dtype for {name}: {other:?}"),
    };
    Ok(Some(value))
}

fn write_dump(dir: &Path, debug: &LongCatDebugOutputs) -> Result<()> {
    fs::create_dir_all(dir)
        .with_context(|| format!("failed to create dump dir {}", dir.display()))?;

    let mut manifest = DumpManifest {
        tensors: BTreeMap::new(),
        scalars: BTreeMap::new(),
    };

    write_tensor(dir, &mut manifest, "input_ids", &debug.input_ids)?;
    write_tensor(dir, &mut manifest, "attention_mask", &debug.attention_mask)?;
    write_tensor(dir, &mut manifest, "text_condition", &debug.text_condition)?;
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
    write_tensor(dir, &mut manifest, "y0", &debug.y0)?;
    write_tensor(dir, &mut manifest, "latent_cond", &debug.latent_cond)?;
    write_tensor(dir, &mut manifest, "prompt_noise", &debug.prompt_noise)?;
    write_tensor(
        dir,
        &mut manifest,
        "transformer_out_t0",
        &debug.transformer_out_t0,
    )?;
    write_tensor(dir, &mut manifest, "null_pred_t0", &debug.null_pred_t0)?;
    println!("Rust null_pred_t0 shape: {:?}", debug.null_pred_t0.shape().dims());
    write_tensor(dir, &mut manifest, "velocity_zero", &debug.velocity_zero)?;
    write_tensor(dir, &mut manifest, "output_latent", &debug.output_latent)?;
    write_tensor(
        dir,
        &mut manifest,
        "output_waveform",
        &debug.output_waveform,
    )?;
    manifest
        .scalars
        .insert("duration".to_string(), debug.duration);

    let manifest_path = dir.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?).with_context(|| {
        format!(
            "failed to write LongCat debug manifest {}",
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
    let shape = tensor.shape().dims().to_vec();
    let (dtype_name, file_name, bytes) = match tensor.dtype() {
        DType::F32 => {
            let values: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            ("f32".to_string(), format!("{name}.f32.bin"), bytes)
        }
        DType::U32 => {
            let values: Vec<u32> = tensor.flatten_all()?.to_vec1()?;
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            ("u32".to_string(), format!("{name}.u32.bin"), bytes)
        }
        DType::I64 => {
            let values: Vec<i64> = tensor.flatten_all()?.to_vec1()?;
            let mut bytes = Vec::with_capacity(values.len() * 8);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            ("i64".to_string(), format!("{name}.i64.bin"), bytes)
        }
        other => bail!("unsupported tensor dtype for dump {name}: {other:?}"),
    };

    let path = dir.join(&file_name);
    fs::write(&path, bytes)
        .with_context(|| format!("failed to write tensor dump {}", path.display()))?;
    manifest.tensors.insert(
        name.to_string(),
        TensorManifestEntry {
            dtype: dtype_name,
            shape,
            file: file_name,
        },
    );
    Ok(())
}
