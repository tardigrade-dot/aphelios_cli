// Qwen3 ForcedAligner – NAR (non-autoregressive) forced alignment.
//
// Architecture: same encoder + Qwen3 decoder as ASR-0.6B, but with:
//   - classify_num (5000) output head instead of vocab_size
//   - lm_head NOT tied to embed_tokens
//   - Single forward pass (no KV cache, no autoregressive loop)

use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{
    embedding, linear_no_bias, ops::softmax_last_dim, rms_norm, rotary_emb::rope, Embedding,
    Linear, RmsNorm, VarBuilder,
};
use candle_transformers::models::qwen3::Config as Qwen3Config;
use thiserror::Error;
use tracing::info;

use crate::qwenasr::audio::{self, AudioConfig, AudioError};
use crate::qwenasr::encoder::Encoder;
use crate::qwenasr::preset::ModelPreset;
use crate::qwenasr::tokenizer::{
    Tokenizer, TokenizerError, TOKEN_AUDIO_END, TOKEN_AUDIO_START, TOKEN_TIMESTAMP,
};
use crate::qwenasr::transcribe::collect_shards;

// ── Constants ────────────────────────────────────────────────────────────────

const CLASSIFY_NUM: usize = 5000;
const TIMESTAMP_SEGMENT_TIME_MS: f64 = 80.0;

// ── Public types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AlignItem {
    pub text: String,
    pub start_time: f64, // seconds
    pub end_time: f64,   // seconds
}

#[derive(Debug, Error)]
pub enum AlignerError {
    #[error("audio error: {0}")]
    Audio(#[from] AudioError),
    #[error("model error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("transcribe error: {0}")]
    Transcribe(#[from] crate::qwenasr::transcribe::TranscribeError),
    #[error("missing weights: {0}")]
    MissingWeights(String),
}

// ── Text preprocessing ───────────────────────────────────────────────────────

fn is_cjk_char(ch: char) -> bool {
    let code = ch as u32;
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3400..=0x4DBF).contains(&code)
        || (0x20000..=0x2A6DF).contains(&code)
        || (0x2A700..=0x2B73F).contains(&code)
        || (0x2B740..=0x2B81F).contains(&code)
        || (0x2B820..=0x2CEAF).contains(&code)
        || (0xF900..=0xFAFF).contains(&code)
}

fn is_kept_char(ch: char) -> bool {
    ch == '\'' || ch.is_alphanumeric()
}

fn clean_token(token: &str) -> String {
    token.chars().filter(|&ch| is_kept_char(ch)).collect()
}

/// Split a segment that may contain mixed CJK and Latin characters.
/// CJK chars become individual tokens; Latin chars are grouped.
fn split_segment_with_cjk(seg: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut buf = String::new();
    for ch in seg.chars() {
        if is_cjk_char(ch) {
            if !buf.is_empty() {
                tokens.push(std::mem::take(&mut buf));
            }
            tokens.push(ch.to_string());
        } else {
            buf.push(ch);
        }
    }
    if !buf.is_empty() {
        tokens.push(buf);
    }
    tokens
}

/// Tokenize text for alignment: split by whitespace, clean punctuation,
/// and split CJK characters individually.
pub fn tokenize_for_alignment(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for seg in text.split_whitespace() {
        let cleaned = clean_token(seg);
        if !cleaned.is_empty() {
            tokens.extend(split_segment_with_cjk(&cleaned));
        }
    }
    tokens
}

// ── Timestamp fix (LIS-based monotonicity correction) ────────────────────────

fn fix_timestamps(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    // Find LIS (longest increasing subsequence) using O(n²) DP
    let mut dp = vec![1usize; n];
    let mut parent = vec![usize::MAX; n];
    for i in 1..n {
        for j in 0..i {
            if data[j] <= data[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
    }

    let max_idx = dp
        .iter()
        .enumerate()
        .max_by_key(|(_, &v)| v)
        .map(|(i, _)| i)
        .unwrap();

    let mut lis_indices = Vec::new();
    let mut idx = max_idx;
    loop {
        lis_indices.push(idx);
        if parent[idx] == usize::MAX {
            break;
        }
        idx = parent[idx];
    }
    lis_indices.reverse();

    let mut is_normal = vec![false; n];
    for &i in &lis_indices {
        is_normal[i] = true;
    }

    let mut result = data.to_vec();
    let mut i = 0;
    while i < n {
        if !is_normal[i] {
            let mut j = i;
            while j < n && !is_normal[j] {
                j += 1;
            }
            let anomaly_count = j - i;

            let left_val = (0..i).rev().find(|&k| is_normal[k]).map(|k| result[k]);
            let right_val = (j..n).find(|&k| is_normal[k]).map(|k| result[k]);

            if anomaly_count <= 2 {
                for k in i..j {
                    match (left_val, right_val) {
                        (None, Some(rv)) => result[k] = rv,
                        (Some(lv), None) => result[k] = lv,
                        (Some(lv), Some(rv)) => {
                            result[k] = if (k as f64 - (i as f64 - 1.0)) <= (j as f64 - k as f64) {
                                lv
                            } else {
                                rv
                            };
                        }
                        (None, None) => {}
                    }
                }
            } else {
                match (left_val, right_val) {
                    (Some(lv), Some(rv)) => {
                        let step = (rv - lv) / (anomaly_count + 1) as f64;
                        for k in i..j {
                            result[k] = lv + step * (k - i + 1) as f64;
                        }
                    }
                    (Some(lv), None) => {
                        for k in i..j {
                            result[k] = lv;
                        }
                    }
                    (None, Some(rv)) => {
                        for k in i..j {
                            result[k] = rv;
                        }
                    }
                    (None, None) => {}
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    result
}

// ── RoPE cache (self-contained, same as decoder.rs) ──────────────────────────

struct RopeCache {
    sin: Tensor,
    cos: Tensor,
}

impl RopeCache {
    fn new(cfg: &Qwen3Config, dev: &Device) -> candle_core::Result<Self> {
        let half = cfg.head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / cfg.rope_theta.powf(2.0 * i as f64 / cfg.head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), dev)?;
        let max_pos = cfg.max_position_embeddings;
        let t = Tensor::arange(0u32, max_pos as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_pos, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (_, _, seq, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq)?.contiguous()?;
        let sin = self.sin.narrow(0, offset, seq)?.contiguous()?;
        Ok((
            rope(&q.contiguous()?, &cos, &sin)?,
            rope(&k.contiguous()?, &cos, &sin)?,
        ))
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn repeat_kv(t: Tensor, n: usize) -> candle_core::Result<Tensor> {
    if n == 1 {
        return Ok(t);
    }
    let (b, h, l, d) = t.dims4()?;
    t.unsqueeze(2)?
        .expand((b, h, n, l, d))?
        .reshape((b, h * n, l, d))
}

fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
) -> candle_core::Result<Linear> {
    if bias {
        candle_nn::linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}

fn causal_mask(b: usize, tgt: usize, dev: &Device) -> candle_core::Result<Tensor> {
    let mask: Vec<f32> = (0..tgt)
        .flat_map(|i| (0..tgt).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
        .collect();
    Tensor::from_vec(mask, (b, 1, tgt, tgt), dev)
}

// ── Attention (no KV cache) ──────────────────────────────────────────────────

struct AlignerAttn {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    n_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rope: Arc<RopeCache>,
}

impl AlignerAttn {
    fn new(cfg: &Qwen3Config, rope: Arc<RopeCache>, vb: VarBuilder) -> candle_core::Result<Self> {
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let bias = cfg.attention_bias;
        Ok(Self {
            q_proj: linear_b(cfg.hidden_size, n_heads * head_dim, bias, vb.pp("q_proj"))?,
            k_proj: linear_b(
                cfg.hidden_size,
                n_kv_heads * head_dim,
                bias,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_b(
                cfg.hidden_size,
                n_kv_heads * head_dim,
                bias,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_b(n_heads * head_dim, cfg.hidden_size, bias, vb.pp("o_proj"))?,
            q_norm: rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            n_heads,
            n_kv_heads,
            n_kv_groups: n_heads / n_kv_heads,
            head_dim,
            hidden_size: n_heads * head_dim,
            rope,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> candle_core::Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head RMSNorm
        let q =
            self.q_norm
                .forward(&q.flatten(0, 2)?)?
                .reshape((b, self.n_heads, l, self.head_dim))?;
        let k = self.k_norm.forward(&k.flatten(0, 2)?)?.reshape((
            b,
            self.n_kv_heads,
            l,
            self.head_dim,
        ))?;

        // RoPE (offset=0, no KV cache)
        let (q, k) = self.rope.apply(&q, &k, 0)?;

        // GQA expand
        let k = repeat_kv(k, self.n_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.n_kv_groups)?.contiguous()?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let mut scores = (q.contiguous()?.matmul(&k_t)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = softmax_last_dim(&scores)?;

        probs
            .contiguous()?
            .matmul(&v.contiguous()?)?
            .transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

// ── MLP ──────────────────────────────────────────────────────────────────────

struct AlignerMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl AlignerMlp {
    fn new(cfg: &Qwen3Config, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ── Transformer layer ────────────────────────────────────────────────────────

struct AlignerLayer {
    attn: AlignerAttn,
    mlp: AlignerMlp,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl AlignerLayer {
    fn new(cfg: &Qwen3Config, rope: Arc<RopeCache>, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            attn: AlignerAttn::new(cfg, rope, vb.pp("self_attn"))?,
            mlp: AlignerMlp::new(cfg, vb.pp("mlp"))?,
            ln1: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> candle_core::Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.attn.forward(&h, mask)?;
        let x = (x + h)?;
        let h2 = self.mlp.forward(&self.ln2.forward(&x)?)?;
        x + h2
    }
}

// ── AlignerDecoder ───────────────────────────────────────────────────────────

struct AlignerDecoder {
    embed_tokens: Embedding,
    layers: Vec<AlignerLayer>,
    norm: RmsNorm,
    lm_head: Linear,
}

impl AlignerDecoder {
    fn load(
        paths: &[impl AsRef<Path>],
        cfg: &Qwen3Config,
        classify_num: usize,
        dev: &Device,
    ) -> candle_core::Result<Self> {
        let paths: Vec<&Path> = paths.iter().map(|p| p.as_ref()).collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::F32, dev)? };
        let vb = vb.pp("thinker");

        let rope = Arc::new(RopeCache::new(cfg, dev)?);

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;

        // lm_head is NOT tied to embed_tokens for the ForcedAligner
        let lm_head = linear_no_bias(cfg.hidden_size, classify_num, vb.pp("lm_head"))?;

        let vb_l = vb.pp("model.layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(AlignerLayer::new(cfg, rope.clone(), vb_l.pp(i))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    /// Look up token embeddings: `[b, seq]` → `[b, seq, hidden]`.
    fn embed(&self, token_ids: &Tensor) -> candle_core::Result<Tensor> {
        self.embed_tokens.forward(token_ids)
    }

    /// NAR forward: process entire sequence, return logits for ALL positions.
    /// Returns `[b, seq, classify_num]`.
    fn forward_all(&self, embeds: &Tensor) -> candle_core::Result<Tensor> {
        let (b, l, _) = embeds.dims3()?;
        let dev = embeds.device();

        let mask = if l > 1 {
            Some(causal_mask(b, l, dev)?)
        } else {
            None
        };

        let mut h = embeds.clone();
        for layer in &self.layers {
            h = layer.forward(&h, mask.as_ref())?;
        }

        // Apply norm and lm_head to ALL positions
        let h = self.norm.forward(&h)?;
        self.lm_head.forward(&h)
    }
}

// ── ForcedAligner (public API) ───────────────────────────────────────────────

pub struct ForcedAligner {
    encoder: Encoder,
    decoder: AlignerDecoder,
    tokenizer: Tokenizer,
    audio_cfg: AudioConfig,
    device: Device,
}

impl ForcedAligner {
    pub fn load_with_device(
        model_dir: &Path,
        dev: Device,
    ) -> std::result::Result<Self, AlignerError> {
        let preset = ModelPreset::from_dir_aligner(model_dir);
        let cfg = preset.config();
        let shards = collect_shards(model_dir)?;

        let encoder = Encoder::load(&shards, cfg.encoder, &dev)?;
        let decoder = AlignerDecoder::load(&shards, &cfg.decoder, CLASSIFY_NUM, &dev)?;
        let tokenizer = Tokenizer::load(model_dir)?;

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
            audio_cfg: cfg.audio,
            device: dev,
        })
    }

    /// Run forced alignment: given audio and its transcript, produce word-level timestamps.
    pub fn align(
        &self,
        wav_path: &Path,
        text: &str,
        language: &str,
    ) -> std::result::Result<Vec<AlignItem>, AlignerError> {
        let samples = audio::load_wav(wav_path, &self.audio_cfg)?;
        self.align_samples(&samples, text, language)
    }

    /// Run forced alignment directly on loaded waveform samples.
    pub fn align_samples(
        &self,
        samples: &[f32],
        text: &str,
        _language: &str,
    ) -> std::result::Result<Vec<AlignItem>, AlignerError> {
        let dev = &self.device;

        // 1. Extract Mel spectrogram
        let (flat, n_frames) = audio::mel_spectrogram(samples, &self.audio_cfg);
        let mel = Tensor::from_vec(flat, (self.audio_cfg.mel_bins, n_frames), dev)?;
        let enc_out = self.encoder.forward(&mel)?;
        let n_audio = enc_out.dims()[0];
        info!("aligner encoder: {n_audio} audio tokens");

        // 2. Text preprocessing
        let word_list = tokenize_for_alignment(text);
        if word_list.is_empty() {
            return Ok(vec![]);
        }
        info!("aligner: {} words/chars to align", word_list.len());

        // 3. Build suffix token IDs: <audio_end> + word1_bpe + <ts><ts> + word2_bpe + <ts><ts> ...
        let mut suffix_ids: Vec<u32> = vec![TOKEN_AUDIO_END];
        let mut ts_positions_in_suffix: Vec<usize> = Vec::new();

        for word in &word_list {
            let word_ids = self.tokenizer.encode(word)?;
            suffix_ids.extend(word_ids);
            ts_positions_in_suffix.push(suffix_ids.len());
            suffix_ids.push(TOKEN_TIMESTAMP);
            ts_positions_in_suffix.push(suffix_ids.len());
            suffix_ids.push(TOKEN_TIMESTAMP);
        }

        // Global positions: offset by (1 for audio_start + n_audio encoder tokens)
        let audio_offset = 1 + n_audio;
        let ts_global: Vec<usize> = ts_positions_in_suffix
            .iter()
            .map(|&p| audio_offset + p)
            .collect();

        // 4. Build combined embeddings
        let audio_start_ids = Tensor::from_vec(vec![TOKEN_AUDIO_START], (1, 1), dev)?;
        let audio_start_emb = self.decoder.embed(&audio_start_ids)?;
        let audio_emb = enc_out.unsqueeze(0)?; // [1, n_audio, hidden]
        let suffix_len = suffix_ids.len();
        let suffix_t = Tensor::from_vec(suffix_ids, (1, suffix_len), dev)?;
        let suffix_emb = self.decoder.embed(&suffix_t)?;

        let combined = Tensor::cat(&[&audio_start_emb, &audio_emb, &suffix_emb], 1)?;
        let total_len = combined.dims()[1];
        info!("aligner: total sequence length = {total_len}");

        // 5. Forward (NAR, single pass)
        let logits = self.decoder.forward_all(&combined)?; // [1, total_len, 5000]

        // 6. Extract timestamps at marker positions
        let logits_2d = logits.squeeze(0)?; // [total_len, 5000]
        let mut raw_class_ids: Vec<u32> = Vec::with_capacity(ts_global.len());
        for &pos in &ts_global {
            let pos_logits = logits_2d.get(pos)?;
            let class_id = pos_logits.argmax(0)?.to_scalar::<u32>()?;
            raw_class_ids.push(class_id);
        }

        // 7. Convert to milliseconds
        let timestamps_ms: Vec<f64> = raw_class_ids
            .iter()
            .map(|&id| id as f64 * TIMESTAMP_SEGMENT_TIME_MS)
            .collect();

        // 8. Fix monotonicity
        let fixed_ms = fix_timestamps(&timestamps_ms);

        // 9. Build result
        let mut items = Vec::with_capacity(word_list.len());
        for (i, word) in word_list.iter().enumerate() {
            let start_ms = fixed_ms[i * 2];
            let end_ms = fixed_ms[i * 2 + 1];
            items.push(AlignItem {
                text: word.clone(),
                start_time: (start_ms / 1000.0 * 1000.0).round() / 1000.0,
                end_time: (end_ms / 1000.0 * 1000.0).round() / 1000.0,
            });
        }

        Ok(items)
    }
}
