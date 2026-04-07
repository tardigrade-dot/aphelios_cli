use crate::longcat_audiodit::config::AudioDiTConfig;
use anyhow::{anyhow, bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{ops, Embedding, Module, VarBuilder};
use candle_transformers::models::t5::{Config as T5Config, T5EncoderModel};
use std::{
    cell::RefCell,
    fs,
    path::{Path, PathBuf},
};
use tokenizers::Tokenizer;

#[derive(Debug)]
pub struct LongCatTextEncoder {
    tokenizer: Tokenizer,
    model: RefCell<T5EncoderModel>,
    token_embedding: Embedding,
    ln_weight: Tensor,
    ln_bias: Tensor,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub text_norm_feat: bool,
    pub text_add_embed: bool,
    pub tokenizer_path: PathBuf,
}

#[derive(Debug)]
pub struct EncodedTextBatch {
    pub input_ids: Tensor,
    pub attention_mask: Tensor,
    pub hidden_states: Tensor,
    pub lengths: Tensor,
}

impl LongCatTextEncoder {
    pub fn load(
        config: &AudioDiTConfig,
        model_weights: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let tokenizer_path = tokenizer_path.as_ref().to_path_buf();
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| {
            anyhow!(
                "failed to load LongCat tokenizer from {}: {err}",
                tokenizer_path.display()
            )
        })?;

        let t5_cfg = to_t5_config(config);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_weights.as_ref()], dtype, device)
                .context("failed to open LongCat safetensors for text encoder")?
        };
        let text_vb = vb.pp("text_encoder");
        let model = T5EncoderModel::load(text_vb.clone(), &t5_cfg)
            .context("failed to load LongCat UMT5 encoder from safetensors")?;
        let token_embedding = candle_nn::embedding(
            t5_cfg.vocab_size,
            t5_cfg.d_model,
            text_vb.pp("encoder.embed_tokens"),
        )
        .context("failed to load LongCat text token embeddings")?;
        let ln_weight = Tensor::ones((t5_cfg.d_model,), DType::F32, device)?;
        let ln_bias = Tensor::zeros((t5_cfg.d_model,), DType::F32, device)?;

        Ok(Self {
            tokenizer,
            model: RefCell::new(model),
            token_embedding,
            ln_weight,
            ln_bias,
            hidden_dim: config.text_encoder_config.d_model,
            num_layers: config.text_encoder_config.num_layers,
            num_heads: config.text_encoder_config.num_heads,
            vocab_size: config.text_encoder_config.vocab_size,
            text_norm_feat: config.text_norm_feat,
            text_add_embed: config.text_add_embed,
            tokenizer_path,
        })
    }

    pub fn normalize_text(text: &str) -> String {
        let lowered = text.to_lowercase();
        let replaced: String = lowered
            .chars()
            .map(|ch| match ch {
                '"' | '“' | '”' | '‘' | '’' => ' ',
                other => other,
            })
            .collect();
        replaced.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    pub fn tokenize(&self, texts: &[String], device: &Device) -> Result<(Tensor, Tensor, Tensor)> {
        ensure!(
            !texts.is_empty(),
            "LongCat tokenize requires at least one text"
        );
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|err| anyhow!("failed to tokenize LongCat text batch: {err}"))?;
        let max_len = encodings
            .iter()
            .map(|encoding| encoding.get_ids().len())
            .max()
            .unwrap_or(0);
        if max_len == 0 {
            bail!("LongCat tokenizer produced empty encodings");
        }

        let mut ids = Vec::with_capacity(texts.len() * max_len);
        let mut mask = Vec::with_capacity(texts.len() * max_len);
        let mut lengths = Vec::with_capacity(texts.len());

        for encoding in encodings {
            let token_ids = encoding.get_ids();
            let len = token_ids.len();
            lengths.push(len as u32);
            ids.extend(token_ids.iter().copied().map(|v| v as u32));
            mask.extend(std::iter::repeat_n(1u32, len));
            let pad = max_len - len;
            ids.extend(std::iter::repeat_n(0u32, pad));
            mask.extend(std::iter::repeat_n(0u32, pad));
        }

        let input_ids = Tensor::from_vec(ids, (texts.len(), max_len), device)?;
        let attention_mask = Tensor::from_vec(mask, (texts.len(), max_len), device)?;
        let lengths = Tensor::from_vec(lengths, texts.len(), device)?;
        Ok((input_ids, attention_mask, lengths))
    }

    pub fn encode_batch(&self, texts: &[String], device: &Device) -> Result<EncodedTextBatch> {
        let (input_ids, attention_mask, lengths) = self.tokenize(texts, device)?;
        let encoder_dtype = if device.is_metal() || device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let embedding_output = self
            .token_embedding
            .forward(&input_ids)?
            .to_dtype(DType::F32)?;
        let mut hidden_states = self
            .model
            .borrow_mut()
            .forward_dt(&input_ids, Some(encoder_dtype))?
            .to_dtype(DType::F32)?;

        if self.text_norm_feat {
            hidden_states = ops::layer_norm(&hidden_states, &self.ln_weight, &self.ln_bias, 1e-6)?;
        }
        if self.text_add_embed {
            let first_hidden = if self.text_norm_feat {
                ops::layer_norm(&embedding_output, &self.ln_weight, &self.ln_bias, 1e-6)?
            } else {
                embedding_output
            };
            hidden_states = hidden_states.add(&first_hidden)?;
        }

        Ok(EncodedTextBatch {
            input_ids,
            attention_mask,
            hidden_states,
            lengths,
        })
    }
}

pub fn default_umt5_tokenizer_path() -> Option<PathBuf> {
    let direct = PathBuf::from("/Volumes/sw/pretrained_models/umt5-base/tokenizer.json");
    if direct.exists() {
        return Some(direct);
    }

    let home = std::env::var_os("HOME")?;
    let cached = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--google--umt5-base/snapshots/0de9394d54f8975e71838d309de1cb496c894ab9/tokenizer.json");
    cached.exists().then_some(cached)
}

pub fn ensure_tokenizer_path(path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = path {
        if path.exists() {
            return Ok(path.to_path_buf());
        }
        bail!("LongCat tokenizer path does not exist: {}", path.display());
    }
    default_umt5_tokenizer_path().ok_or_else(|| {
        anyhow!(
            "LongCat tokenizer.json not found. Pass LongCatInferenceConfig.tokenizer_path or install /Volumes/sw/pretrained_models/umt5-base"
        )
    })
}

fn to_t5_config(config: &AudioDiTConfig) -> T5Config {
    T5Config {
        vocab_size: config.text_encoder_config.vocab_size,
        d_model: config.text_encoder_config.d_model,
        d_kv: config.text_encoder_config.d_kv,
        d_ff: config.text_encoder_config.d_ff,
        num_layers: config.text_encoder_config.num_layers,
        num_decoder_layers: Some(config.text_encoder_config.num_layers),
        num_heads: config.text_encoder_config.num_heads,
        relative_attention_num_buckets: config.text_encoder_config.relative_attention_num_buckets,
        relative_attention_max_distance: config.text_encoder_config.relative_attention_max_distance,
        dropout_rate: config.text_encoder_config.dropout_rate,
        layer_norm_epsilon: config.text_encoder_config.layer_norm_epsilon,
        initializer_factor: 1.0,
        feed_forward_proj: candle_transformers::models::t5::ActivationWithOptionalGating {
            gated: config.text_encoder_config.dense_act_fn.contains("gated")
                || config.text_encoder_config.model_type == "umt5",
            activation: candle_nn::Activation::NewGelu,
        },
        tie_word_embeddings: config.text_encoder_config.tie_word_embeddings,
        is_decoder: false,
        is_encoder_decoder: config.text_encoder_config.is_encoder_decoder,
        use_cache: config.text_encoder_config.use_cache,
        pad_token_id: config.text_encoder_config.pad_token_id,
        eos_token_id: config.text_encoder_config.eos_token_id,
        decoder_start_token_id: Some(config.text_encoder_config.pad_token_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_normalization_matches_python_helper() {
        let text = " A“B”  C\nD ";
        assert_eq!(LongCatTextEncoder::normalize_text(text), "a b c d");
    }

    #[test]
    fn tokenizer_path_resolution_prefers_existing_override() {
        let path = PathBuf::from("/tmp/longcat-tokenizer.json");
        fs::write(&path, "{}").unwrap();
        let resolved = ensure_tokenizer_path(Some(&path)).unwrap();
        assert_eq!(resolved, path);
        fs::remove_file(resolved).unwrap();
    }
}
