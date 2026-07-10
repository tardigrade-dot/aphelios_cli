// Sidon speech restoration pipeline:
//   audio → mel features → ONNX predictor (w2v-BERT) → ONNX vocoder (DAC) → 48kHz audio
//
// Two-pass architecture: predictor runs first (all chunks) then its session is
// dropped before the vocoder loads. This keeps only one model (~400 MB) in RAM
// at a time instead of both (~500 MB + CoreML compilation >> 2 GB).
//
// CoreML is disabled — CPUOnly is slower than ONNX CPU, and GPU crashes on
// dynamic-shape models. Use `--device cuda` on Linux with NVIDIA GPU for real
// acceleration.

use crate::{SIDON_MODEL_ID, voiceclear::preprocess};
use anyhow::Result;
use aphelios_core::{hub::load_file_local_or_download, utils::{bytes_to_f32_vec, common::get_available_ep}};
use ort::{inputs, session::Session, value::Tensor};
use tracing::info;
use std::{fs, path::PathBuf, time::Instant};

const PRED_IN_DIM: usize = 160;
const PRED_OUT_DIM: usize = 1024;
const SAMPLE_RATE_IN: u32 = 16_000;
const SAMPLE_RATE_OUT: u32 = 48_000;

/// Audio chunk: 96s at 16 kHz (matching Python reference).
const CHUNK_SAMPLES: usize = 96 * SAMPLE_RATE_IN as usize;
/// Pad per chunk (160 samples each side, like Python reference).
const CHUNK_PAD: usize = 160;
/// Trim from each decoded chunk to remove boundary artifacts (20 ms @ 48 kHz).
const DECODER_TRIM: usize = 960;

/// Predictor sub-chunk in frames. w2v-BERT self-attention is O(T²); splitting
/// avoids quadratic blowup on CPU. 1500 frames ≈ 30s @ 50 Hz.
const PRED_CHUNK_FRAMES: usize = 1500;
/// Vocoder sub-chunk in frames. DAC decoder benefits from L2-cache-sized batches.
const VOC_CHUNK_FRAMES: usize = 3000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Device {
    Cpu,
    Cuda,
    /// CoreML is NOT recommended — GPU crashes on dynamic shapes, CPUOnly is
    /// slower than ONNX CPU, and compilation uses massive memory.
    CoreML,
}

/// Stores the decoder-ready features from one audio chunk.
struct ChunkOutput {
    features: Vec<f32>, // [t_decode, 1024]
}

/// Full pipeline. Two-pass: predictor phase then vocoder phase, only one model
/// session alive at a time.
pub struct SidonPipeline{
    model_id: String,
    mel_filters_f64: Vec<f64>,
    window_f64: Vec<f64>
}

impl SidonPipeline {
    pub fn new(model: Option<impl Into<String>>) -> Result<Self> {

        let model_id = model
                    .map(|m| m.into())
                    .unwrap_or_else(|| SIDON_MODEL_ID.to_string());

        if !ort::init().with_name("sidon").commit() {
            anyhow::bail!("Failed to initialize ONNX Runtime");
        }

        Ok(Self {
            mel_filters_f64: bytes_to_f32_vec(&fs::read(load_file_local_or_download(&model_id, "mel_filters_f32.bin"))?).iter().map(|&v| v as f64).collect(),
            window_f64: bytes_to_f32_vec(&fs::read(load_file_local_or_download(&model_id, "mel_window_f32.bin"))?).iter().map(|&v| v as f64).collect(),
            model_id: model_id,
        })
    }

    /// Process 16 kHz mono audio → 48 kHz enhanced audio.
    pub fn process(&mut self, audio_16k: &[f32]) -> Result<Vec<f32>> {
        let t_total = Instant::now();
        let n = audio_16k.len();
        if n == 0 {
            return Ok(vec![]);
        }

        // --- preprocessing ---
        let t0 = Instant::now();
        let mut audio = audio_16k.to_vec();
        preprocess::highpass_50hz(&mut audio);
        preprocess::peak_normalize(&mut audio, 0.9);
        audio.resize(audio.len() + 24000, 0.0); // end padding
        let t_pre = t0.elapsed();
        info!("[timing] preprocess: {:.2}s", t_pre.as_secs_f32());

        // ================================================================
        // Phase 1: Predictor (w2v-BERT) — process all chunks, drop session.
        // ================================================================
        info!("--- Phase 1: predictor ---");
        let (chunk_outputs, t_mel, t_pred) = {
            let session = make_session(&load_file_local_or_download(&self.model_id, "sidon-predictor-fp16.onnx"))?;
            let mut pred_session = PredictorSession { session };
            let mut chunk_outputs: Vec<ChunkOutput> = Vec::new();
            let mut feature_cache: Option<Vec<f32>> = None;
            let mut pos = 0usize;
            let mut chunk_idx = 0usize;
            let mut t_mel = 0.0f64;
            let mut t_pred = 0.0f64;

            while pos < audio.len() {
                let chunk_end = (pos + CHUNK_SAMPLES).min(audio.len());
                let chunk = &audio[pos..chunk_end];
                let chunk_dur = chunk.len() as f32 / SAMPLE_RATE_IN as f32;

                // Pad edges
                let mut padded = Vec::with_capacity(chunk.len() + CHUNK_PAD * 2);
                padded.extend(std::iter::repeat(0.0f32).take(CHUNK_PAD));
                padded.extend_from_slice(chunk);
                padded.extend(std::iter::repeat(0.0f32).take(CHUNK_PAD));

                // Mel features
                let t0_mel = Instant::now();
                let feats = preprocess::compute_mel_features(&padded, &self.mel_filters_f64, &self.window_f64);
                let t_feat = feats.len() / PRED_IN_DIM;
                let dt_mel = t0_mel.elapsed().as_secs_f64();
                t_mel += dt_mel;

                // Predictor with sub-chunking
                let t0_pred = Instant::now();
                let mut hidden = if t_feat > 0 {
                    let mut h = Vec::new();
                    let mut s = 0usize;
                    while s < t_feat {
                        let e = (s + PRED_CHUNK_FRAMES).min(t_feat);
                        let sub = &feats[s * PRED_IN_DIM..e * PRED_IN_DIM];
                        h.extend_from_slice(&pred_session.predict(sub, e - s)?);
                        s = e;
                    }
                    h
                } else {
                    vec![]
                };
                let dt_pred = t0_pred.elapsed().as_secs_f64();
                t_pred += dt_pred;

                // Feature caching across chunks
                if let Some(ref cache) = feature_cache {
                    let mut combined = cache.clone();
                    combined.extend_from_slice(&hidden);
                    hidden = combined;
                }
                let t_hidden = hidden.len() / PRED_OUT_DIM;
                if t_hidden > 1 {
                    feature_cache = Some(hidden[(t_hidden - 1) * PRED_OUT_DIM..].to_vec());
                    let t_decode = t_hidden - 1;
                    let decode_features = hidden[..t_decode * PRED_OUT_DIM].to_vec();
                    chunk_outputs.push(ChunkOutput { features: decode_features });
                } else if t_hidden == 1 {
                    feature_cache = Some(hidden.clone());
                }

                info!(
                    "[pred chunk {:>2}] {:.0}s audio | mel={:.2}s pred={:.2}s | feats={} hidden={}",
                    chunk_idx, chunk_dur, dt_mel, dt_pred, t_feat, t_hidden
                );
                chunk_idx += 1;
                pos += CHUNK_SAMPLES;
            }
            (chunk_outputs, t_mel, t_pred)
        }; // ← pred_session DROPPED here, freeing ~400 MB

        info!("[mem] predictor session released");

        // ================================================================
        // Phase 2: Vocoder (DAC decoder) — process all features, drop session.
        // ================================================================
        info!("--- Phase 2: vocoder ({} chunks) ---", chunk_outputs.len());
        let (mut output, t_voc) = {
            let session = make_session(&load_file_local_or_download(&self.model_id, "sidon-vocoder-fp16.onnx"))?;
            let mut voc_session = VocoderSession { session };
            let mut output = Vec::new();
            let mut t_voc = 0.0f64;

            for (i, ch) in chunk_outputs.iter().enumerate() {
                let t0 = Instant::now();
                let t_feat = ch.features.len() / PRED_OUT_DIM;
                let mut decoded = Vec::new();
                let mut s = 0usize;
                while s < t_feat {
                    let e = (s + VOC_CHUNK_FRAMES).min(t_feat);
                    let sub = &ch.features[s * PRED_OUT_DIM..e * PRED_OUT_DIM];
                    decoded.extend_from_slice(&voc_session.decode(sub, e - s)?);
                    s = e;
                }
                if decoded.len() > DECODER_TRIM {
                    decoded.truncate(decoded.len() - DECODER_TRIM);
                }
                let dt = t0.elapsed().as_secs_f64();
                t_voc += dt;
                output.extend_from_slice(&decoded);
                let out_dur = decoded.len() as f32 / SAMPLE_RATE_OUT as f32;
                info!(
                    "[voc chunk {:>2}] {:.1}s audio | voc={:.2}s",
                    i, out_dur, dt
                );
            }
            (output, t_voc)
        }; // ← voc_session DROPPED here

        info!("[mem] vocoder session released");

        // --- final trim ---
        let target = (n as u64 * SAMPLE_RATE_OUT as u64 / SAMPLE_RATE_IN as u64) as usize;
        if output.len() > target {
            output.truncate(target);
        }

        let total = t_total.elapsed().as_secs_f64();
        info!(
            "[summary] mel={:.1}s pred={:.1}s voc={:.1}s pre={:.1}s | total={:.1}s ({:.1}x realtime)",
            t_mel, t_pred, t_voc, t_pre.as_secs_f64(),
            total, (n as f64 / SAMPLE_RATE_IN as f64) / total
        );

        Ok(output)
    }
}

// ── session helpers ──────────────────────────────────────────────────────

struct PredictorSession {
    session: Session,
}

impl PredictorSession {
    fn predict(&mut self, features: &[f32], t: usize) -> Result<Vec<f32>> {
        let tensor = Tensor::from_array(([1i64, t as i64, PRED_IN_DIM as i64], features.to_vec()))?;
        let out = self.session.run(inputs!["input_features" => tensor])?;
        let (_, data) = out["features"].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }
}

struct VocoderSession {
    session: Session,
}

impl VocoderSession {
    fn decode(&mut self, features: &[f32], t: usize) -> Result<Vec<f32>> {
        let tensor = Tensor::from_array(([1i64, t as i64, PRED_OUT_DIM as i64], features.to_vec()))?;
        let out = self.session.run(inputs!["features" => tensor])?;
        let (_, data) = out["audio"].try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }
}

fn make_session(model_path: &PathBuf) -> Result<Session> {
    let mut b = Session::builder()?;
    b = b.with_execution_providers(get_available_ep())
        .map_err(|e| anyhow::anyhow!("load model file faile: {}", e))?;
    Ok(b.commit_from_file(model_path)?)
}
