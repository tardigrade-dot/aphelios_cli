use std::path::PathBuf;
use anyhow::Result;
use tracing::info;

use crate::{
    qwen3asr,
    qwenasr,
    qwenrsasr::QwenRsAsrEngine,
    silerovad::{VadConfig, VadProcessor},
    srt::{derive_srt_path, generate_srt},
    AsrSegment, DecodingResult,
};

/// Selectable VAD model options.
#[derive(Debug, Clone)]
pub enum VadModel {
    /// Standard Silero VAD model (dir or file path)
    SileroVad(String),
    /// SenseVoice VAD model (`vad-model.onnx` path)
    SenseVoiceVad(String),
}

/// Selectable ASR Engine implementations.
#[derive(Debug, Clone)]
pub enum AsrEngineChoice {
    /// Candle-based Qwen3-ASR native engine
    Qwen3Asr { model_dir: Option<String> },
    /// Candle-based Qwen-ASR native engine
    QwenAsr { model_dir: Option<String> },
    /// Pure Rust CPU `qwen-asr = "0.11"` crate engine
    QwenRsCrate { model_dir: String },
}

/// Abstract ASR Pipeline:
/// `Audio/Video File -> Optional VAD -> Optional ASR Engine -> Optional Forced Aligner -> SRT Output File`
pub struct AsrPipeline {
    pub input_path: String,
    pub vad: Option<VadModel>,
    pub asr: Option<AsrEngineChoice>,
    pub aligner_model_path: Option<String>,
    pub language: String,
    pub context: Option<String>,
    pub output_srt_path: Option<String>,
}

impl AsrPipeline {
    pub fn new(input_path: impl Into<String>) -> Self {
        Self {
            input_path: input_path.into(),
            vad: None,
            asr: None,
            aligner_model_path: None,
            language: "Chinese".to_string(),
            context: None,
            output_srt_path: None,
        }
    }

    pub fn with_vad(mut self, vad: VadModel) -> Self {
        self.vad = Some(vad);
        self
    }

    pub fn with_asr(mut self, asr: AsrEngineChoice) -> Self {
        self.asr = Some(asr);
        self
    }

    pub fn with_aligner(mut self, aligner_model_path: impl Into<String>) -> Self {
        self.aligner_model_path = Some(aligner_model_path.into());
        self
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    pub fn with_output_srt(mut self, output_path: impl Into<String>) -> Self {
        self.output_srt_path = Some(output_path.into());
        self
    }

    /// Execute the full ASR pipeline and return the generated SRT file path.
    pub async fn run(&self) -> Result<String> {
        info!("Starting ASR Pipeline for: {}", self.input_path);
        let srt_target_path = self
            .output_srt_path
            .clone()
            .unwrap_or_else(|| derive_srt_path(&self.input_path));

        // Case 1: QwenRsCrate engine
        if let Some(AsrEngineChoice::QwenRsCrate { ref model_dir }) = self.asr {
            let mut engine = QwenRsAsrEngine::load(model_dir)?;
            let text = engine.transcribe_file(&self.input_path)?;

            let srt_content = format!("1\n00:00:00,000 --> 00:59:59,999\n{}\n\n", text.trim());
            std::fs::write(&srt_target_path, &srt_content)?;
            info!("Generated SRT at: {}", srt_target_path);
            return Ok(srt_target_path);
        }

        // Case 2: Native Qwen3Asr or QwenAsr with optional VAD and Aligner
        match self.asr {
            Some(AsrEngineChoice::Qwen3Asr { ref model_dir }) => {
                if let Some(ref aligner_path) = self.aligner_model_path {
                    let vad_dir = match &self.vad {
                        Some(VadModel::SileroVad(p)) => Some(p.as_str()),
                        Some(VadModel::SenseVoiceVad(p)) => Some(p.as_str()),
                        None => None,
                    };
                    qwen3asr::qwen3asr_with_vad(
                        model_dir.as_deref(),
                        Some(aligner_path.as_str()),
                        vad_dir,
                        &self.input_path,
                        &self.language,
                        self.context.as_deref(),
                    )
                    .await?;
                } else {
                    let text = qwen3asr::qwen3asr_simple_with_context(
                        model_dir.as_deref(),
                        &self.input_path,
                        &self.language,
                        self.context.as_deref(),
                    )?;
                    let segment = AsrSegment {
                        start: 0.0,
                        duration: 0.0,
                        dr: DecodingResult {
                            tokens: vec![],
                            text,
                            avg_logprob: 0.0,
                            no_speech_prob: 0.0,
                            temperature: 0.0,
                            compression_ratio: 0.0,
                        },
                        sub_segments: vec![],
                    };
                    generate_srt(&[segment], &srt_target_path).await?;
                }
            }
            Some(AsrEngineChoice::QwenAsr { ref model_dir }) => {
                let aligner_path = self.aligner_model_path.as_deref();
                let vad_dir = match &self.vad {
                    Some(VadModel::SileroVad(p)) => Some(p.as_str()),
                    Some(VadModel::SenseVoiceVad(p)) => Some(p.as_str()),
                    None => None,
                };
                qwenasr::qwen3asr_with_vad(
                    model_dir.as_deref(),
                    aligner_path,
                    vad_dir,
                    &self.input_path,
                    &self.language,
                    self.context.as_deref(),
                )
                .await?;
            }
            None => {
                // VAD only export to SRT
                if let Some(ref vad_model) = self.vad {
                    let vad_path = match vad_model {
                        VadModel::SileroVad(p) => PathBuf::from(p),
                        VadModel::SenseVoiceVad(p) => PathBuf::from(p),
                    };
                    let mut vad_processor = VadProcessor::new(&vad_path, VadConfig::default())?;
                    let segments = vad_processor.process_from_file(&self.input_path)?;
                    crate::srt::generate_vad_srt(&segments, &srt_target_path).await?;
                } else {
                    anyhow::bail!("Neither VAD nor ASR engine specified in pipeline");
                }
            }
            _ => unreachable!(),
        }

        info!("Pipeline completed successfully, output: {}", srt_target_path);
        Ok(srt_target_path)
    }
}
