use anyhow::{Context, Result};
use qwen_asr::context::QwenCtx;
use qwen_asr::transcribe;
use tracing::info;

use crate::srt::derive_srt_path;

/// Wrapper around `qwen-asr` 0.11 pure Rust CPU speech recognition engine.
pub struct QwenRsAsrEngine {
    ctx: QwenCtx,
}

impl QwenRsAsrEngine {
    /// Load Qwen-ASR model from directory or model ID.
    pub fn load(model_path_or_id: &str) -> Result<Self> {
        info!("Loading Qwen-ASR (crate 0.11) model from: {}", model_path_or_id);
        let ctx = QwenCtx::load(model_path_or_id)
            .ok_or_else(|| anyhow::anyhow!("Failed to load qwen-asr model: {}", model_path_or_id))?;
        info!("Qwen-ASR model loaded successfully");
        Ok(Self { ctx })
    }

    /// Transcribe an audio file directly.
    pub fn transcribe_file(&mut self, audio_path: &str) -> Result<String> {
        info!("Transcribing file with qwen-asr: {}", audio_path);
        let text = transcribe::transcribe(&mut self.ctx, audio_path)
            .ok_or_else(|| anyhow::anyhow!("qwen-asr transcribe failed for: {}", audio_path))?;
        Ok(text)
    }

    /// Transcribe raw 16kHz f32 PCM samples directly.
    pub fn transcribe_samples(&mut self, samples: &[f32]) -> Result<String> {
        let text = transcribe::transcribe_audio(&mut self.ctx, samples)
            .ok_or_else(|| anyhow::anyhow!("qwen-asr transcribe_audio failed"))?;
        Ok(text)
    }

    /// Transcribe file and save output text or subtitle.
    pub fn transcribe_to_srt(&mut self, audio_path: &str, output_srt: Option<&str>) -> Result<String> {
        let text = self.transcribe_file(audio_path)?;
        let save_path = output_srt
            .map(|s| s.to_string())
            .unwrap_or_else(|| derive_srt_path(audio_path));

        let srt_content = format!("1\n00:00:00,000 --> 00:59:59,999\n{}\n\n", text.trim());
        std::fs::write(&save_path, &srt_content)
            .with_context(|| format!("Failed to write SRT file to {}", save_path))?;

        info!("Saved SRT to: {}", save_path);
        Ok(save_path)
    }

    pub fn inner_ctx(&mut self) -> &mut QwenCtx {
        &mut self.ctx
    }
}

/// Convenience function to run qwen-asr 0.11 transcription.
pub fn run_qwen_rs_asr(model_path: &str, audio_path: &str) -> Result<String> {
    let mut engine = QwenRsAsrEngine::load(model_path)?;
    engine.transcribe_file(audio_path)
}
