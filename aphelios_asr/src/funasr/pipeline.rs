use aphelios_core::audio::{MonoBuffer, ResampleQuality, Resampler};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;
use tokenizers::Tokenizer;

use super::audio_adaptor::{AudioAdaptor, CTCDecoder};
use super::audio_encoder::SenseVoiceEncoderSmall;
use super::frontend::WavFrontend;
use super::llm::{Config as LlmConfig, ModelForCausalLM};

use base64::prelude::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct CharacterEntry {
    pub char: String,
    pub start: f64, // seconds
    pub end: f64,   // seconds
    pub score: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct FunASRResult {
    pub text: String,
    pub characters: Vec<CharacterEntry>,
}

pub struct FunASR {
    pub audio_encoder: SenseVoiceEncoderSmall,
    pub audio_adaptor: AudioAdaptor,
    pub ctc_decoder: CTCDecoder,
    pub llm: ModelForCausalLM,
    pub tokenizer: Tokenizer,
    pub frontend: WavFrontend,
    pub ctc_vocab: Vec<String>,
}
impl FunASR {
    pub fn load<P: AsRef<Path>>(vb: VarBuilder, model_dir: P) -> Result<Self> {
        let device = vb.device();
        let audio_encoder = SenseVoiceEncoderSmall::load(
            vb.pp("audio_encoder"),
            560,  // input_size (stacked mel bins: 7 * 80)
            512,  // output_size
            4,    // attention_heads
            2048, // linear_units
            50,   // num_blocks
            20,   // tp_blocks
            11,   // kernel_size
            0,    // sanm_shfit
        )?;

        // audio adaptor from config
        // downsample_rate: 1, ffn_dim: 2048, llm_dim: 1024, encoder_dim: 512, n_layer: 2
        let audio_adaptor = AudioAdaptor::load(
            vb.pp("audio_adaptor"),
            2,    // num_blocks
            16,   // attention_heads (1024/64 = 16)
            512,  // encoder_dim
            1024, // llm_dim
            2048, // linear_ffn_dim
        )?;

        let ctc_decoder = CTCDecoder::load(
            vb.pp("ctc_decoder"),
            vb.pp("ctc"),
            2,     // num_blocks
            8,     // attention_heads (512/64 = 8)
            512,   // input_size
            512,   // output_size
            128,   // ffn_dim
            60515, // vocab_size
        )?;

        // llm configured for Qwen3-0.6B (from shapes)
        let llm_config = LlmConfig {
            vocab_size: 151936,
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: Some(128),
            max_position_embeddings: 8192,
            sliding_window: 8192,
            max_window_layers: 28,
            tie_word_embeddings: false,
            rope_theta: 1000000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            hidden_act: candle_nn::Activation::Silu,
        };

        let llm = ModelForCausalLM::new(&llm_config, vb.pp("llm"))?;

        // Load tokenizer from a file
        let tokenizer_path = model_dir.as_ref().join("Qwen3-0.6B/tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // Initialize frontend
        let frontend = WavFrontend::new(80, 16000, 25, 10, 7, 6, device)?;

        // Load CTC vocab
        let ctc_vocab_path = model_dir.as_ref().join("multilingual.tiktoken");
        let ctc_vocab = load_ctc_vocab(ctc_vocab_path)?;

        Ok(Self {
            audio_encoder,
            audio_adaptor,
            ctc_decoder,
            llm,
            tokenizer,
            frontend,
            ctc_vocab,
        })
    }

    // Simple helper. Given mel spectrograms [B, T, 80], it returns audio embeddings
    // and adapted LLM embeddings.
    pub fn forward_encoder(&self, speech: &Tensor) -> Result<Tensor> {
        let encoder_out = self.audio_encoder.forward(speech)?;
        let adaptor_out = self.audio_adaptor.forward(&encoder_out)?;
        Ok(adaptor_out)
    }

    /// Full inference: audio samples -> result with timestamps/scores
    pub fn inference(
        &mut self,
        pcm: &Tensor,
        sample_rate: u32,
        device: &Device,
    ) -> Result<FunASRResult> {
        // 0. Resample to 16kHz if necessary
        let pcm_16k = if sample_rate != 16000 {
            let samples = pcm.flatten_all()?.to_vec1::<f32>()?;
            let mono = MonoBuffer::new(samples, sample_rate);
            let resampler = Resampler::new().with_quality(ResampleQuality::High);
            let resampled = resampler
                .resample_mono(&mono, 16000)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let len = resampled.len();
            Tensor::from_vec(resampled.samples, (1, len), device)?
        } else {
            if pcm.rank() == 1 {
                pcm.unsqueeze(0)?
            } else {
                pcm.clone()
            }
        };

        // 1. Audio Preprocessing -> mel features -> LFR stacking
        let feats = self.frontend.forward(&pcm_16k)?;

        // 2. Encoder + Adaptor
        let encoder_out = self.audio_encoder.forward(&feats)?;
        let audio_embeds = self.audio_adaptor.forward(&encoder_out)?;

        // 2b. CTC Decoding for timestamps
        let ctc_logits = self.ctc_decoder.forward(&encoder_out)?; // [1, T_enc, V_ctc]
        let ctc_probs = candle_nn::ops::softmax(&ctc_logits.squeeze(0)?, 1)?; // [T_enc, V_ctc]
        let ctc_ids = ctc_probs.argmax(1)?.to_vec1::<u32>()?;
        let ctc_scores = ctc_probs.to_vec2::<f32>()?; // Large [T_enc, V_ctc]... maybe only take max?

        // Let's only keep non-blank (usually last token is blank in FunASR)
        // FunASR blank ID is often vocab_size - 1 or 0?
        // Based on tiktoken, 0-60k are tokens. Blank might be 60514?
        // Actually, many FunASR models use 0 as blank.
        let blank_id = 0; // Assumption for now, or check typical SenseVoice
        let mut ctc_peaks = Vec::new();
        let mut last_id = blank_id;

        for (i, &id) in ctc_ids.iter().enumerate() {
            if id != blank_id && id != last_id {
                // Each frame is 60ms (Frontend shift 10ms * LFR 6)
                let start = i as f64 * 0.06;
                let end = (i + 1) as f64 * 0.06;
                let score = ctc_scores[i][id as usize] as f64;
                let char = self.ctc_vocab.get(id as usize).cloned().unwrap_or_default();
                ctc_peaks.push(CharacterEntry {
                    char,
                    start,
                    end,
                    score,
                });
            }
            last_id = id;
        }

        // 3. LLM Prompt Assembly (ChatML)
        let system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
        let user_prompt = "<|im_start|>user\n语音转写：X<|im_end|>\n<|im_start|>assistant\n";

        let prompt = format!("{}{}", system_prompt, user_prompt);
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let token_ids = tokens.get_ids();

        // Find placeholder (X) index. ID 55 is "X" in Qwen3 tokenizer.
        let placeholder_idx = token_ids.iter().position(|&id| id == 55).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "Could not find placeholder X in prompt. Tokens: {:?}",
                token_ids
            ))
        })?;

        // 4. Input Embedding Assembly
        let text_ids_tensor = Tensor::from_slice(token_ids, (token_ids.len(),), device)?;
        let text_embeds = self.llm.base_model.embed_tokens(&text_ids_tensor)?; // [L, D]

        let part1 = text_embeds.narrow(0, 0, placeholder_idx)?;
        let part2 = text_embeds.narrow(
            0,
            placeholder_idx + 1,
            text_ids_tensor.dim(0)? - placeholder_idx - 1,
        )?;

        let input_embeds = Tensor::cat(
            &[&part1.unsqueeze(0)?, &audio_embeds, &part2.unsqueeze(0)?],
            1,
        )?;

        // 5. Greedy Decoding
        let mut generated_ids = Vec::new();
        let mut llm_scores = Vec::new();
        let mut current_input_embeds = input_embeds;
        let mut seqlen_offset = 0;

        let max_gen_len = 200;
        self.llm.clear_kv_cache();

        for _ in 0..max_gen_len {
            let seq_len = current_input_embeds.dim(1)?;
            let logits = self
                .llm
                .forward_with_embeds(&current_input_embeds, seqlen_offset)?;

            let probs = candle_nn::ops::softmax(&logits.squeeze(1)?.squeeze(0)?, 0)?;
            let token_id = probs.argmax(0)?.to_scalar::<u32>()?;
            let score = probs.to_vec1::<f32>()?[token_id as usize] as f64;

            if token_id == 151643 || token_id == 151645 {
                break;
            }
            generated_ids.push(token_id);
            llm_scores.push(score);

            seqlen_offset += seq_len;
            let next_token_id = Tensor::from_slice(&[token_id], (1, 1), device)?;
            current_input_embeds = self.llm.base_model.embed_tokens(&next_token_id)?;
        }

        let output_text = self
            .tokenizer
            .decode(&generated_ids, true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // 6. Align LLM tokens with CTC timestamps
        // For each generated token, map it to a CharacterEntry
        let mut final_chars: Vec<CharacterEntry> = Vec::new();
        let mut ctc_idx = 0;

        for (i, &id) in generated_ids.iter().enumerate() {
            let token_str = self.tokenizer.id_to_token(id).unwrap_or_default();
            // Clean token_str if it contains control chars (like Qwen's <|...|>)
            let text = self.tokenizer.decode(&[id], true).unwrap_or_default();
            if text.is_empty() {
                continue;
            }

            // Find best matching ctc_peak
            // Simplistic: just take them in order.
            let (start, end) = if ctc_idx < ctc_peaks.len() {
                let entry = &ctc_peaks[ctc_idx];
                ctc_idx += 1;
                (entry.start, entry.end)
            } else {
                // Heuristic: increment if run out
                let last_end = final_chars.last().map(|c| c.end).unwrap_or(0.0);
                (last_end, last_end + 0.06)
            };

            final_chars.push(CharacterEntry {
                char: text,
                start,
                end,
                score: llm_scores[i],
            });
        }

        Ok(FunASRResult {
            text: output_text,
            characters: final_chars,
        })
    }
}

fn load_ctc_vocab<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path)?;
    let mut vocab = vec![String::new(); 60520]; // Slightly larger buffer
    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some((b64, id_str)) = line.rsplit_once(' ') {
            let id: usize = id_str.parse().map_err(|e: std::num::ParseIntError| {
                candle_core::Error::Msg(format!("Line {}: Parse error: {}", line_num + 1, e))
            })?;

            if b64.is_empty() {
                continue;
            }

            // Some tokens might be tricky. Trim and decode.
            let decoded = BASE64_STANDARD.decode(b64.trim()).unwrap_or_else(|_| {
                // Fallback for tricky tokens like line 48475
                b64.as_bytes().to_vec()
            });

            let s = String::from_utf8_lossy(&decoded).into_owned();
            if id < vocab.len() {
                vocab[id] = s;
            }
        }
    }
    Ok(vocab)
}
