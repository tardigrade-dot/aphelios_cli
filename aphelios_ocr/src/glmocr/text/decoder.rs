use candle_core::quantized::GgmlDType;
use candle_core::{Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, Module, RmsNorm, VarBuilder};

use super::attention::KvCache;
use super::block::TextDecoderLayer;
use super::rotary::TextRotaryEmbedding;
use crate::glmocr::config::TextConfig;
use crate::glmocr::nn_utils::rms_norm;

/// Full text decoder: embedding + transformer layers + final norm + lm_head.
///
/// Only uses num_hidden_layers (16) for inference.
/// The nextn_predict layer (layer 16) is for MTP training only and is skipped.
pub struct TextDecoder {
    embed_tokens: Embedding,
    layers: Vec<TextDecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary_emb: TextRotaryEmbedding,
}

impl TextDecoder {
    /// Create a new text decoder.
    ///
    /// `vb` points to `model.language_model` in the safetensors.
    /// `lm_head_vb` points to `lm_head` (at top level, outside `model.language_model`).
    ///
    /// Only loads num_hidden_layers (16). The MTP nextn_predict layer is skipped.
    pub fn new(
        config: &TextConfig,
        vb: VarBuilder,
        lm_head_vb: VarBuilder,
        qdtype: Option<GgmlDType>,
    ) -> Result<Self> {
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;

        // Only use base layers for inference (skip MTP nextn_predict layers)
        let num_layers = config.num_hidden_layers;
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(TextDecoderLayer::new(
                config,
                vb.pp(format!("layers.{i}")),
                qdtype,
            )?);
        }

        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;

        // lm_head is at the top level, not under model.language_model
        let lm_head = linear_no_bias(config.hidden_size, config.vocab_size, lm_head_vb)?;

        let rotary_emb = TextRotaryEmbedding::new(config.head_dim, config.rope_theta, vb.device())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
        })
    }

    /// Embed token IDs to hidden states.
    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    /// Run the transformer layers (rotary → all layers → norm), returning hidden states.
    ///
    /// Used by both `forward_with_cache` and batched generation.
    fn run_layers(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_caches: Vec<Option<KvCache>>,
    ) -> Result<(Tensor, Vec<Option<KvCache>>)> {
        let (cos, sin) = self.rotary_emb.forward(position_ids)?;

        let mut hidden_states = inputs_embeds.clone();
        let mut new_caches = Vec::with_capacity(self.layers.len());

        let mut kv_caches = kv_caches;
        while kv_caches.len() < self.layers.len() {
            kv_caches.push(None);
        }

        for (layer, cache) in self.layers.iter().zip(kv_caches.into_iter()) {
            let (h, new_cache) =
                layer.forward(&hidden_states, &cos, &sin, attention_mask, cache)?;
            hidden_states = h;
            new_caches.push(Some(new_cache));
        }

        // Final norm
        hidden_states = self.norm.forward(&hidden_states)?;
        Ok((hidden_states, new_caches))
    }

    /// Forward pass that properly consumes KV-caches.
    pub fn forward_with_cache(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_caches: Vec<Option<KvCache>>,
    ) -> Result<(Tensor, Vec<Option<KvCache>>)> {
        let (hidden_states, new_caches) =
            self.run_layers(inputs_embeds, position_ids, attention_mask, kv_caches)?;

        // LM head: logits for last token only (single-sequence path)
        let seq_len = hidden_states.dim(1)?;
        let last_hidden = hidden_states.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&last_hidden)?;

        Ok((logits, new_caches))
    }

    /// Like `forward_with_cache` but returns normalized hidden states (before lm_head).
    /// Used by batched generation where per-element position gathering is needed.
    pub fn forward_to_hidden(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_caches: Vec<Option<KvCache>>,
    ) -> Result<(Tensor, Vec<Option<KvCache>>)> {
        self.run_layers(inputs_embeds, position_ids, attention_mask, kv_caches)
    }

    /// Apply the LM head to hidden states to get logits.
    /// `hidden_states`: [batch, seq_len, hidden] → [batch, seq_len, vocab_size]
    pub fn lm_head_forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.lm_head.forward(hidden_states)
    }
}
