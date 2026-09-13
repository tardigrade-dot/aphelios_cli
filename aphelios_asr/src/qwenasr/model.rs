use crate::qwenasr::encoder::EncoderConfig;
use crate::qwenasr::audio::AudioConfig;
use candle_transformers::models::qwen3::Config as Qwen3Config;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub encoder: EncoderConfig,
    pub decoder: Qwen3Config,
    pub audio: AudioConfig,
}

#[derive(Debug)]
pub struct Model {
    pub config: ModelConfig,
}
