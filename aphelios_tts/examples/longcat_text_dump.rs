use anyhow::{Context, Result};
use aphelios_tts::longcat_audiodit::{default_dtype, select_device};
use aphelios_tts::longcat_audiodit::loader::ModelPaths;
use aphelios_tts::longcat_audiodit::text_encoder::{ensure_tokenizer_path, LongCatTextEncoder};
use candle_core::{DType, Tensor};
use clap::Parser;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(name = "longcat-text-dump")]
struct Args {
    #[arg(long)]
    text: String,
    #[arg(long = "prompt_text", alias = "prompt-text")]
    prompt_text: Option<String>,
    #[arg(long = "model_dir", alias = "model-dir", default_value = "/Volumes/sw/pretrained_models/LongCat-AudioDiT-1B")]
    model_dir: PathBuf,
    #[arg(long = "tokenizer_path", alias = "tokenizer-path", default_value = "/Volumes/sw/pretrained_models/umt5-base/tokenizer.json")]
    tokenizer_path: Option<PathBuf>,
    #[arg(long = "dump_dir", alias = "dump-dir")]
    dump_dir: PathBuf,
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
    let device = select_device(args.cpu)?;
    let dtype = default_dtype(&device);
    let paths = ModelPaths::discover(&args.model_dir, args.tokenizer_path.as_ref())?;
    let config = paths.load_config()?;
    let tokenizer_path = ensure_tokenizer_path(paths.tokenizer.as_deref())?;
    let text_encoder =
        LongCatTextEncoder::load(&config, &paths.weights, &tokenizer_path, DType::F32, &device)?;

    let text = LongCatTextEncoder::normalize_text(&args.text);
    let full_text = match args.prompt_text.as_deref() {
        Some(prompt_text) => {
            let prompt = LongCatTextEncoder::normalize_text(prompt_text);
            format!("{prompt} {text}")
        }
        None => text,
    };
    let mut debug_encoded = text_encoder
        .encode_batch_debug(&[full_text], &device)
        .context("failed to encode LongCat text condition")?;
    let mut encoded = debug_encoded.batch;
    encoded.hidden_states = encoded.hidden_states.to_dtype(dtype)?;

    fs::create_dir_all(&args.dump_dir)?;
    let mut manifest = DumpManifest {
        tensors: BTreeMap::new(),
    };
    write_tensor(&args.dump_dir, &mut manifest, "input_ids", &encoded.input_ids)?;
    write_tensor(
        &args.dump_dir,
        &mut manifest,
        "attention_mask",
        &encoded.attention_mask,
    )?;
    write_tensor(&args.dump_dir, &mut manifest, "lengths", &encoded.lengths)?;
    write_tensor(
        &args.dump_dir,
        &mut manifest,
        "embedding_output",
        &debug_encoded.embedding_output,
    )?;
    write_tensor(
        &args.dump_dir,
        &mut manifest,
        "raw_last_hidden",
        &debug_encoded.raw_last_hidden,
    )?;
    write_tensor(
        &args.dump_dir,
        &mut manifest,
        "first_block_hidden",
        &debug_encoded.first_block_hidden,
    )?;
    write_tensor(
        &args.dump_dir,
        &mut manifest,
        "second_block_hidden",
        &debug_encoded.second_block_hidden,
    )?;
    write_tensor(
        &args.dump_dir,
        &mut manifest,
        "text_condition",
        &encoded.hidden_states.to_dtype(DType::F32)?,
    )?;

    let manifest_path = args.dump_dir.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    println!("Dumped text tensors to {}", args.dump_dir.display());
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
        other => anyhow::bail!("unsupported tensor dtype for dump {name}: {other:?}"),
    };

    fs::write(dir.join(&file_name), bytes)?;
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
