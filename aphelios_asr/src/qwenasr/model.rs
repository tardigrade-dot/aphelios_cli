use crate::qwenasr::transcribe::collect_shards;
use crate::{QWEN3_ASR_MODEL_ID, qwenasr::audio::AudioConfig};
use crate::qwenasr::encoder::EncoderConfig;
use crate::qwenasr::preset::ModelPreset;
use aphelios_core::hub::load_or_download;
use candle_transformers::models::qwen3::Config as Qwen3Config;

use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
enum ModelError {
    #[error("Failed to locate weights in {0}")]
    MissingWeights(String),
    #[error("Failed to read the json index {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to parse the json index {0}")]
    CorruptIndex(#[from] serde_json::Error),
    #[error("Invalid json index: {0}")]
    InvalidIndex(String),
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub encoder: EncoderConfig,
    pub decoder: Qwen3Config,
    // TODO
    // pub tokenizer: TokenizerConfig,
    pub audio: AudioConfig,
}

#[derive(Debug)]
pub struct Model {
    // TODO
    // encoder: Encoder,
    // TODO
    // decoder: Decoder,
    // TODO
    // tokenizer: Tokenizer,
    pub config: ModelConfig,
}

// impl Model {
//     fn load_from_dir(model_dir: Option<&str>) -> Result<Self, ModelError> {
//         println!("Loading model from {:?}", model_dir);
//         // searching for a safetensors json index
//         let index = load_or_download(QWEN3_ASR_MODEL_ID, model_dir, "model.safetensors.index.json");

//         let s = collect_shards(QWEN3_ASR_MODEL_ID, model_dir);
//         if index.exists() {
//             // Reading the json
//             let content = std::fs::read_to_string(index)?;
//             let jsonv: serde_json::Value = serde_json::from_str(&content)?;
//             let weight_map = jsonv
//                 .get("weight_map")
//                 .and_then(|v| v.as_object())
//                 .ok_or_else(|| {
//                     ModelError::InvalidIndex("missing or invalid weight_map".to_string())
//                 })?;

//             // println!("{:#?}", weight_map);

//             let mut shards: Vec<String> = weight_map
//                 .values()
//                 .filter_map(|v| v.as_str().map(|s| s.to_string()))
//                 .collect();
//             // no need to use proper sort
//             shards.sort_unstable();
//             // since we deduplicate right after
//             shards.dedup();

//             // we finally have a list of shards
//             let shards: Vec<PathBuf> = shards.into_iter().map(|s| model_dir.join(s)).collect();

//             // are we sure the files are actually there?
//             for shard_path in &shards {
//                 if !shard_path.exists() {
//                     return Err(ModelError::MissingWeights(shard_path.display().to_string()));
//                 }
//             }

//             println!("{:#?}", shards);
//             // fine, we can proceed
//             // We could get everything
//             // let weights = Weights::from_files(&shards);

//             Ok(Model {
//                 config: ModelPreset::from_dir(model_dir).config(),
//             })
//         } else {
//             // no index? let's go for a single shard
//             let single_shard = model_dir.join("model.safetensors");
//             if !single_shard.exists() {
//                 return Err(ModelError::MissingWeights(format!("{:?}", model_dir)));
//             }

//             Ok(Model {
//                 config: ModelPreset::from_dir(model_dir).config(),
//             })
//         }
//     }
// }
