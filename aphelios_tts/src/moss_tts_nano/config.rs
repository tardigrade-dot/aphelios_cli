use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub model_files: ModelFiles,
    pub generation_defaults: GenerationDefaults,
    pub tts_config: TtsConfig,
    pub prompt_templates: PromptTemplates,
    pub builtin_voices: Vec<BuiltinVoice>,
    pub text_samples: Vec<TextSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFiles {
    pub tts_meta: String,
    pub codec_meta: String,
    #[serde(default = "default_tokenizer_model")]
    pub tokenizer_model: String,
}

fn default_tokenizer_model() -> String {
    "tokenizer.model".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationDefaults {
    pub max_new_frames: i32,
    pub do_sample: bool,
    pub sample_mode: String,
    pub text_temperature: f32,
    pub text_top_p: f32,
    pub text_top_k: i32,
    pub audio_temperature: f32,
    pub audio_top_p: f32,
    pub audio_top_k: i32,
    pub audio_repetition_penalty: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsConfig {
    pub audio_assistant_slot_token_id: i32,
    pub audio_end_token_id: i32,
    pub audio_pad_token_id: i32,
    pub audio_user_slot_token_id: i32,
    pub audio_start_token_id: i32,
    pub n_vq: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplates {
    pub user_prompt_prefix_token_ids: Vec<i32>,
    pub user_prompt_after_reference_token_ids: Vec<i32>,
    pub assistant_prompt_prefix_token_ids: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltinVoice {
    pub voice: String,
    pub prompt_audio_codes: Vec<Vec<i32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSample {
    pub text: String,
    pub text_token_ids: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsMeta {
    pub files: TtsFiles,
    pub model_config: TtsModelConfig,
    pub onnx: TtsOnnxConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsFiles {
    pub prefill: String,
    pub decode_step: String,
    pub local_decoder: String,
    pub local_greedy_frame: Option<String>,
    pub local_fixed_sampled_frame: Option<String>,
    pub local_cached_step: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsModelConfig {
    pub local_layers: i32,
    pub local_heads: i32,
    pub local_head_dim: i32,
    pub audio_codebook_sizes: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsOnnxConfig {
    pub prefill_output_names: Vec<String>,
    pub decode_input_names: Vec<String>,
    pub decode_output_names: Vec<String>,
    pub local_cached_output_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecMeta {
    pub files: CodecFiles,
    pub codec_config: CodecConfig,
    pub streaming_decode: Option<StreamingDecodeConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecFiles {
    pub encode: String,
    pub decode_full: String,
    pub decode_step: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecConfig {
    pub sample_rate: i32,
    pub channels: i32,
    pub num_quantizers: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingDecodeConfig {
    pub transformer_offsets: Vec<TransformerOffset>,
    pub attention_caches: Vec<AttentionCache>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerOffset {
    pub input_name: String,
    pub output_name: String,
    pub shape: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionCache {
    pub offset_input_name: String,
    pub offset_output_name: String,
    pub offset_shape: Vec<i64>,
    pub cached_keys_input_name: String,
    pub cached_keys_output_name: String,
    pub cached_values_input_name: String,
    pub cached_values_output_name: String,
    pub cache_shape: Vec<i64>,
    pub cached_positions_input_name: String,
    pub cached_positions_output_name: String,
    pub positions_shape: Vec<i64>,
}
