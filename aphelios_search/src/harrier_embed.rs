//! Harrier OSS v1 embedding inference using candle's Qwen3 model.
//!
//! This module provides text embedding functionality for `microsoft/harrier-oss-v1-0.6b`,
//! reusing `candle_transformers::models::qwen3::Model` for the forward pass and applying
//! last-token pooling + L2 normalization to produce embeddings.

use anyhow::{Error as E, Result};
use aphelios_core::utils::common;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{kv_cache::ConcatKvCache, ops::softmax_last_dim, Activation, VarBuilder};
use candle_transformers::{
    models::{
        qwen3::Config as Qwen3Config,
        with_tracing::{linear_b, linear_no_bias, Linear, RmsNorm},
    },
    utils::repeat_kv,
};
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Load the Harrier embedding model and tokenizer from local path.
pub struct HarrierEmbedModel {
    model: Qwen3EmbedModel,
    device: Device,
    tokenizer: Tokenizer,
    pad_token_id: u32,
    #[allow(dead_code)]
    dtype: DType,
}

impl HarrierEmbedModel {
    /// Load model from local directory
    pub fn new(model_dir: &str) -> Result<Self> {
        let tokenizer_filename = std::path::Path::new(model_dir).join("tokenizer.json");
        let config_file = std::path::Path::new(model_dir).join("config.json");
        let filenames = vec![std::path::Path::new(model_dir).join("model.safetensors")];

        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // Dynamically get the pad token id, fallback to 151643 if not found
        let pad_token_id = tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .copied()
            .unwrap_or(151643);

        let (device, dtype) = common::get_device_dtype();

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        // Harrier model tensors lack the "model." prefix that candle's Qwen3 expects,
        // so we strip that prefix when looking up tensors.
        let vb = vb.rename_f(|name| {
            if name.starts_with("model.") {
                name.strip_prefix("model.").unwrap_or(name).to_string()
            } else {
                name.to_string()
            }
        });

        let config: Qwen3Config = serde_json::from_slice(&std::fs::read(config_file)?)?;
        let model = Qwen3EmbedModel::new(&config, vb)?;

        Ok(Self {
            model,
            device,
            tokenizer,
            pad_token_id,
            dtype,
        })
    }

    /// Encode a batch of texts into embeddings.
    pub fn encode(&mut self, texts: Vec<&str>) -> Result<Tensor> {
        // Tokenize with padding (right-pad) and truncation
        let encoding = self.tokenizer.encode_batch(texts, true).map_err(E::msg)?;

        let batch_size = encoding.len();
        let max_len = encoding
            .iter()
            .map(|e| e.len())
            .max()
            .unwrap_or(0)
            .min(32768); // max_position_embeddings from config

        // Build input_ids and attention_mask tensors
        let mut input_ids = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask = Vec::with_capacity(batch_size * max_len);

        for enc in &encoding {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();

            // Truncate if needed
            let ids_trunc = if ids.len() > max_len {
                &ids[..max_len]
            } else {
                ids
            };
            let mask_trunc = if mask.len() > max_len {
                &mask[..max_len]
            } else {
                mask
            };

            let pad_len = max_len - ids_trunc.len();

            // Idiomatic Rust padding: extend and resize
            // input_ids: [tokens..., pad_tokens...]
            input_ids.extend(ids_trunc.iter().map(|&id| id as u32));
            input_ids.resize(input_ids.len() + pad_len, self.pad_token_id);

            // attention_mask: [1s..., 0s...]
            attention_mask.extend(mask_trunc.iter().map(|&m| m as u32));
            attention_mask.resize(attention_mask.len() + pad_len, 0u32);
        }

        // Create tensors: [batch, seq_len]
        let input_ids = Tensor::from_slice(&input_ids, (batch_size, max_len), &self.device)?;
        let attention_mask =
            Tensor::from_slice(&attention_mask, (batch_size, max_len), &self.device)?;

        // Match the Python reference path:
        // model(**batch_dict) with use_cache disabled and an explicit attention mask.
        self.model.clear_kv_cache();
        let hidden_states = self.model.forward(&input_ids, Some(&attention_mask), 0)?;

        // Last-token pooling
        let pooled = last_token_pool(&hidden_states, &attention_mask)?;

        // L2 normalization
        let embeddings = l2_normalize(&pooled)?;

        Ok(embeddings)
    }
}

/// Extract the hidden state at the last real (non-padding) token for each sequence.
fn last_token_pool(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let batch_size = hidden_states.dim(0)?;
    let hidden_dim = hidden_states.dim(2)?;

    let last_col_sum = attention_mask
        .narrow(1, attention_mask.dim(1)? - 1, 1)?
        .sum_all()?
        .to_scalar::<u32>()?;

    if last_col_sum == batch_size as u32 {
        return Ok(hidden_states
            .narrow(1, hidden_states.dim(1)? - 1, 1)?
            .squeeze(1)?);
    }

    let mask_sum = attention_mask.sum(1)?; // [batch]
    let mask_i64 = mask_sum.to_dtype(DType::I64)?;
    let ones = Tensor::ones_like(&mask_i64)?;
    let sequence_lengths = (mask_i64 - ones)?;

    let indices = sequence_lengths
        .unsqueeze(1)?
        .expand(&[batch_size, hidden_dim])?
        .unsqueeze(1)?;

    let pooled = hidden_states
        .contiguous()?
        .gather(&indices.contiguous()?, 1)?;

    let pooled = pooled.squeeze(1)?;

    Ok(pooled)
}

/// L2 normalize the embeddings with numerical stability.
fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    // sum_sq = sum(x^2) along dim -1
    let sum_sq = (x * x)?.sum_keepdim(1)?;

    // Add epsilon for numerical stability to prevent sqrt(0) -> division by zero
    let epsilon = Tensor::new(1e-12_f32, x.device())?.to_dtype(x.dtype())?;
    let norm = sum_sq.broadcast_add(&epsilon)?.sqrt()?;

    let norm = norm.broadcast_as(x.shape())?;
    Ok(x.broadcast_div(&norm)?)
}

/// Compute cosine similarity between two sets of embeddings.
pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Convert to F32 BEFORE matmul to prevent precision loss, especially if running in BF16
    let a_f32 = a.to_dtype(DType::F32)?;
    let b_f32 = b.to_dtype(DType::F32)?;

    let similarity = a_f32.matmul(&b_f32.t()?)?;
    let similarity = (similarity * 100.0)?;

    Ok(similarity)
}

#[derive(Debug, Clone)]
struct Qwen3RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl Qwen3RotaryEmbedding {
    fn new(dtype: DType, cfg: &Qwen3Config, dev: &Device) -> candle_core::Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct Qwen3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Qwen3MLP {
    fn new(cfg: &Qwen3Config, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen3MLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<Qwen3RotaryEmbedding>,
    kv_cache: ConcatKvCache,
}

impl Qwen3Attention {
    fn new(
        cfg: &Qwen3Config,
        rotary_emb: Arc<Qwen3RotaryEmbedding>,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        if cfg.use_sliding_window {
            candle_core::bail!("sliding window is not supported")
        }

        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = linear_b(
            cfg.hidden_size,
            num_heads * head_dim,
            cfg.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;

        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        let hidden_size = head_dim * cfg.num_attention_heads;
        let kv_cache = ConcatKvCache::new(2);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb,
            kv_cache,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> candle_core::Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.add(m)?;
        }
        let probs = softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &Qwen3Config,
        rotary: Arc<Qwen3RotaryEmbedding>,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let self_attn = Qwen3Attention::new(cfg, rotary, vb.pp("self_attn"))?;
        let mlp = Qwen3MLP::new(cfg, vb.pp("mlp"))?;
        let ln1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let ln2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> candle_core::Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        x + h2
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
struct Qwen3EmbedModel {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    device: Device,
    dtype: DType,
    num_attention_heads: usize,
}

impl Qwen3EmbedModel {
    fn new(cfg: &Qwen3Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(Qwen3RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary.clone(), vb_l.pp(i))?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            num_attention_heads: cfg.num_attention_heads,
        })
    }

    fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }

    fn attention_mask(
        &self,
        attention_mask: &Tensor,
        tgt_len: usize,
        offset: usize,
    ) -> candle_core::Result<Tensor> {
        let (batch_size, src_len) = attention_mask.dims2()?;
        debug_assert_eq!(src_len, tgt_len + offset);
        let attention_mask = attention_mask.to_vec2::<u32>()?;
        let minf = f32::NEG_INFINITY;
        let mut mask =
            Vec::with_capacity(batch_size * self.num_attention_heads * tgt_len * src_len);

        for row in attention_mask {
            for _head in 0..self.num_attention_heads {
                for i in 0..tgt_len {
                    let q_pos = i + offset;
                    for (j, keep) in row.iter().enumerate() {
                        let causal_ok = j <= q_pos;
                        let key_ok = *keep != 0;
                        mask.push(if causal_ok && key_ok { 0.0 } else { minf });
                    }
                }
            }
        }

        Tensor::from_slice(
            &mask,
            (batch_size, self.num_attention_heads, tgt_len, src_len),
            &self.device,
        )?
        .to_dtype(self.dtype)
    }

    fn forward(
        &mut self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
        offset: usize,
    ) -> candle_core::Result<Tensor> {
        let (batch_size, tgt_len) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        let mask = if tgt_len == 1 && attention_mask.is_none() {
            None
        } else {
            match attention_mask {
                Some(attention_mask) => {
                    let src_attention_mask = if offset == 0 {
                        attention_mask.clone()
                    } else {
                        attention_mask.narrow(1, 0, tgt_len + offset)?
                    };
                    Some(self.attention_mask(&src_attention_mask, tgt_len, offset)?)
                }
                None => {
                    let ones =
                        Tensor::ones((batch_size, tgt_len + offset), DType::U32, &self.device)?;
                    Some(self.attention_mask(&ones, tgt_len, offset)?)
                }
            }
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, mask.as_ref(), offset)?;
        }
        self.norm.forward(&h)
    }
}
