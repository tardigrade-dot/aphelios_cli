use std::path::Path;

use crate::qwenasr::audio::AudioConfig;
use crate::qwenasr::encoder::EncoderConfig;
use crate::qwenasr::model::ModelConfig;
use candle_transformers::models::qwen3::Config as Qwen3Config;

pub enum ModelPreset {
    Qwen3Asr0_6b,
    Qwen3Asr1_7b,
    Qwen3ForcedAligner0_6b,
}

impl ModelPreset {
    /// Detect model variant from the model directory.
    /// 1.7b is distributed as multiple shards with an index file; 0.6b as a single shard.
    pub fn from_dir(dir: &Path) -> Self {
        if dir.join("model.safetensors.index.json").exists() {
            ModelPreset::Qwen3Asr1_7b
        } else {
            ModelPreset::Qwen3Asr0_6b
        }
    }

    /// Detect whether a model directory contains a ForcedAligner model.
    pub fn from_dir_aligner(dir: &Path) -> Self {
        // Read config.json to check model_type
        let config_path = dir.join("config.json");
        if config_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&config_path) {
                if content.contains("qwen3_forced_aligner") {
                    return ModelPreset::Qwen3ForcedAligner0_6b;
                }
            }
        }
        Self::from_dir(dir)
    }
}

impl ModelPreset {
    fn encoder_config(&self) -> EncoderConfig {
        match self {
            ModelPreset::Qwen3Asr0_6b => EncoderConfig {
                d_model: 896,
                layers: 18,
                heads: 14,
                head_dim: 64,
                ffn_dim: 3584,
                output_dim: 1024,
                n_window: 50,
                n_window_infer: 800,
                chunk_size: 100,
            },
            ModelPreset::Qwen3Asr1_7b => EncoderConfig {
                d_model: 1024,
                layers: 24,
                heads: 16,
                head_dim: 64,
                ffn_dim: 4096,
                output_dim: 2048,
                n_window: 50,
                n_window_infer: 800,
                chunk_size: 100,
            },
            ModelPreset::Qwen3ForcedAligner0_6b => EncoderConfig {
                d_model: 1024,
                layers: 24,
                heads: 16,
                head_dim: 64,
                ffn_dim: 4096,
                output_dim: 1024,
                n_window: 50,
                n_window_infer: 800,
                chunk_size: 500,
            },
        }
    }

    fn decoder_config(&self) -> Qwen3Config {
        let (vocab_size, hidden_size, intermediate_size, num_hidden_layers, tie_word_embeddings) =
            match self {
                ModelPreset::Qwen3Asr0_6b => (151936, 1024, 3072, 28, true),
                ModelPreset::Qwen3Asr1_7b => (151936, 2048, 6144, 28, true),
                ModelPreset::Qwen3ForcedAligner0_6b => (152064, 1024, 3072, 28, false),
            };
        Qwen3Config {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            attention_bias: false,
            max_position_embeddings: 65536,
            sliding_window: None,
            max_window_layers: num_hidden_layers,
            tie_word_embeddings,
            rope_theta: 1e6,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            hidden_act: candle_nn::Activation::Silu,
        }
    }

    pub fn config(&self) -> ModelConfig {
        ModelConfig {
            encoder: self.encoder_config(),
            decoder: self.decoder_config(),
            audio: AudioConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn detects_1_7b_when_index_file_present() {
        let dir = env::temp_dir().join("qwen_preset_test_1_7b");
        std::fs::create_dir_all(&dir).unwrap();
        let index = dir.join("model.safetensors.index.json");
        std::fs::write(&index, "{}").unwrap();
        assert!(matches!(
            ModelPreset::from_dir(&dir),
            ModelPreset::Qwen3Asr1_7b
        ));
        std::fs::remove_file(&index).unwrap();
    }

    #[test]
    fn detects_0_6b_when_no_index_file() {
        let dir = env::temp_dir().join("qwen_preset_test_0_6b");
        std::fs::create_dir_all(&dir).unwrap();
        let _ = std::fs::remove_file(dir.join("model.safetensors.index.json"));
        assert!(matches!(
            ModelPreset::from_dir(&dir),
            ModelPreset::Qwen3Asr0_6b
        ));
    }
}
