use anyhow::{Context, Result};
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Debug, Clone, Deserialize)]
pub struct AudioDiTConfig {
    pub dit_adaln_type: String,
    pub dit_adaln_use_text_cond: bool,
    pub dit_bias: bool,
    pub dit_cross_attn: bool,
    pub dit_cross_attn_norm: bool,
    pub dit_depth: usize,
    pub dit_dim: usize,
    pub dit_dropout: f64,
    pub dit_eps: f64,
    pub dit_ff_mult: usize,
    pub dit_heads: usize,
    pub dit_long_skip: bool,
    pub dit_qk_norm: bool,
    pub dit_text_conv: bool,
    pub dit_text_dim: usize,
    pub dit_use_latent_condition: bool,
    pub latent_dim: usize,
    pub latent_hop: usize,
    pub max_wav_duration: usize,
    pub model_type: String,
    pub repa_dit_layer: usize,
    pub sampling_rate: u32,
    pub sigma: f64,
    pub text_add_embed: bool,
    pub text_encoder_config: AudioDiTTextEncoderConfig,
    pub text_encoder_model: String,
    pub text_norm_feat: bool,
    pub transformers_version: String,
    pub vae_config: AudioDiTVaeConfig,
}

impl AudioDiTConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read AudioDiT config: {}", path.display()))?;
        serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse AudioDiT config: {}", path.display()))
    }

    pub fn samples_per_latent_frame(&self) -> usize {
        self.latent_hop
    }

    pub fn max_latent_frames(&self) -> usize {
        (self.max_wav_duration * self.sampling_rate as usize) / self.samples_per_latent_frame()
    }

    pub fn seconds_to_latent_frames(&self, seconds: f64) -> usize {
        let samples = (seconds * self.sampling_rate as f64).ceil() as usize;
        samples.div_ceil(self.samples_per_latent_frame())
    }

    pub fn component_summary(&self) -> LongCatComponentSummary {
        LongCatComponentSummary {
            text_hidden_dim: self.text_encoder_config.d_model,
            dit_hidden_dim: self.dit_dim,
            vae_latent_dim: self.vae_config.latent_dim,
            depth: self.dit_depth,
            heads: self.dit_heads,
            sampling_rate: self.sampling_rate,
            latent_hop: self.latent_hop,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioDiTTextEncoderConfig {
    pub d_ff: usize,
    pub d_kv: usize,
    pub d_model: usize,
    pub dense_act_fn: String,
    pub dropout_rate: f64,
    pub eos_token_id: usize,
    pub is_gated_act: bool,
    pub is_decoder: bool,
    pub is_encoder_decoder: bool,
    pub layer_norm_epsilon: f64,
    pub model_type: String,
    pub num_heads: usize,
    pub num_layers: usize,
    pub pad_token_id: usize,
    pub relative_attention_max_distance: usize,
    pub relative_attention_num_buckets: usize,
    pub scalable_attention: bool,
    pub tie_word_embeddings: bool,
    pub tokenizer_class: String,
    pub use_cache: bool,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioDiTVaeConfig {
    pub c_mults: Vec<usize>,
    pub channels: usize,
    pub downsample_shortcut: String,
    pub downsampling_ratio: usize,
    pub encoder_latent_dim: usize,
    pub final_tanh: bool,
    pub in_channels: usize,
    pub in_shortcut: String,
    pub latent_dim: usize,
    pub model_type: String,
    pub out_shortcut: String,
    pub sample_rate: u32,
    pub scale: f64,
    pub strides: Vec<usize>,
    pub upsample_shortcut: String,
    pub use_snake: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LongCatComponentSummary {
    pub text_hidden_dim: usize,
    pub dit_hidden_dim: usize,
    pub vae_latent_dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub sampling_rate: u32,
    pub latent_hop: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_longcat_config() {
        let raw = r#"
        {
          "dit_adaln_type": "global",
          "dit_adaln_use_text_cond": true,
          "dit_bias": true,
          "dit_cross_attn": true,
          "dit_cross_attn_norm": false,
          "dit_depth": 24,
          "dit_dim": 1536,
          "dit_dropout": 0.0,
          "dit_eps": 1e-06,
          "dit_ff_mult": 4,
          "dit_heads": 24,
          "dit_long_skip": true,
          "dit_qk_norm": true,
          "dit_text_conv": true,
          "dit_text_dim": 768,
          "dit_use_latent_condition": true,
          "latent_dim": 64,
          "latent_hop": 2048,
          "max_wav_duration": 30,
          "model_type": "audiodit",
          "repa_dit_layer": 8,
          "sampling_rate": 24000,
          "sigma": 0.0,
          "text_add_embed": true,
          "text_encoder_config": {
            "d_ff": 2048,
            "d_kv": 64,
            "d_model": 768,
            "dense_act_fn": "gelu_new",
            "dropout_rate": 0.1,
            "eos_token_id": 1,
            "is_gated_act": true,
            "is_decoder": false,
            "is_encoder_decoder": true,
            "layer_norm_epsilon": 1e-06,
            "model_type": "umt5",
            "num_heads": 12,
            "num_layers": 12,
            "pad_token_id": 0,
            "relative_attention_max_distance": 128,
            "relative_attention_num_buckets": 32,
            "scalable_attention": true,
            "tie_word_embeddings": true,
            "tokenizer_class": "T5Tokenizer",
            "use_cache": true,
            "vocab_size": 256384
          },
          "text_encoder_model": "google/umt5-base",
          "text_norm_feat": true,
          "transformers_version": "5.3.0",
          "vae_config": {
            "c_mults": [1, 2, 4, 8, 16],
            "channels": 128,
            "downsample_shortcut": "averaging",
            "downsampling_ratio": 2048,
            "encoder_latent_dim": 128,
            "final_tanh": false,
            "in_channels": 1,
            "in_shortcut": "duplicating",
            "latent_dim": 64,
            "model_type": "audiodit_vae",
            "out_shortcut": "averaging",
            "sample_rate": 24000,
            "scale": 0.71,
            "strides": [2, 4, 4, 8, 8],
            "upsample_shortcut": "duplicating",
            "use_snake": true
          }
        }"#;
        let config: AudioDiTConfig = serde_json::from_str(raw).unwrap();
        assert_eq!(config.component_summary().dit_hidden_dim, 1536);
        assert_eq!(config.max_latent_frames(), 351);
        assert_eq!(config.seconds_to_latent_frames(1.0), 12);
    }
}
