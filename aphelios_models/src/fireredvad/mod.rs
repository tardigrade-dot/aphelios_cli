//! FireRedVAD ONNX inference implementation using ort runtime.
//!
//! This module provides a pure Rust implementation of FireRedVAD voice activity detection,
//! using ONNX models for speech probability prediction.

mod vad_postprocessor;

pub use vad_postprocessor::*;

use anyhow::{Context, Result};
use aphelios_core::{utils::base, ScopedTimer};
use ndarray::{s, Array, Array1, Array2};
use ort::{
    ep::{CoreML, CPU},
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use serde::Deserialize;
use std::{
    path::Path,
    sync::{Arc, Mutex},
};
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const SAMPLE_RATE: usize = 16000;
pub const FRAME_LENGTH_MS: f32 = 25.0;
pub const FRAME_SHIFT_MS: f32 = 10.0;
pub const FRAME_LENGTH_S: f32 = 0.025;
pub const FRAME_SHIFT_S: f32 = 0.010;
pub const NUM_MEL_BINS: usize = 80;

// ---------------------------------------------------------------------------
// CMVN (Cepstral Mean and Variance Normalization)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct CmvnJson {
    dim: usize,
    means: Vec<f32>,
    inverse_std_variances: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Cmvn {
    means: Array1<f32>,
    inv_std: Array1<f32>,
}

impl Cmvn {
    pub fn from_json_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())
            .with_context(|| format!("Failed to open CMVN file: {:?}", path.as_ref()))?;
        let reader = std::io::BufReader::new(file);
        let cmvn: CmvnJson =
            serde_json::from_reader(reader).with_context(|| "Failed to parse CMVN JSON")?;

        if cmvn.dim != NUM_MEL_BINS {
            anyhow::bail!(
                "CMVN dimension mismatch: expected {}, got {}",
                NUM_MEL_BINS,
                cmvn.dim
            );
        }

        Ok(Self {
            means: Array1::from_vec(cmvn.means),
            inv_std: Array1::from_vec(cmvn.inverse_std_variances),
        })
    }

    pub fn apply(&self, fbank: &Array2<f32>) -> Array2<f32> {
        // (T, 80) - means (80,) * inv_std (80,)
        fbank.mapv(|v| v) - &self.means * &self.inv_std
    }
}

// ---------------------------------------------------------------------------
// Kaldi-compatible Fbank extraction
// ---------------------------------------------------------------------------

/// Build Kaldi-style mel filterbank matrix.
/// Returns shape (n_mels, n_fft // 2 + 1) — real spectrum bins.
fn build_mel_filterbank(
    n_fft: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
    sample_rate: usize,
) -> Array2<f32> {
    let n_freqs = n_fft / 2 + 1;
    let nyquist = sample_rate as f32 / 2.0;

    // Frequency bins in Hz
    let freq_bins: Vec<f32> = (0..n_freqs)
        .map(|i| (i as f32) * nyquist / (n_freqs as f32))
        .collect();

    // Mel scale points
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_min + (i as f32) * (mel_max - mel_min) / (n_mels as f32 + 1.0))
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Build filterbank matrix
    let mut filters = Array2::<f32>::zeros((n_mels, n_freqs));

    for m in 1..=n_mels {
        let left = hz_points[m - 1];
        let center = hz_points[m];
        let right = hz_points[m + 1];

        for (k, &f) in freq_bins.iter().enumerate() {
            if left <= f && f <= center {
                filters[[m - 1, k]] = (f - left) / (center - left);
            } else if center < f && f <= right {
                filters[[m - 1, k]] = (right - f) / (right - center);
            }
        }
    }

    filters
}

fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

/// Extract Kaldi-compatible log Mel filterbank features.
///
/// Input: int16 PCM array (1-D)
/// Output: (T, 80) float32 array — log fbank energies
pub fn extract_kaldi_fbank(wav_float: &[f32], sample_rate: usize) -> Array2<f32> {
    use rustfft::FftPlanner;

    let frame_len = ((sample_rate as f32 * FRAME_LENGTH_MS / 1000.0).round()) as usize; // 400
    let frame_shift = ((sample_rate as f32 * FRAME_SHIFT_MS / 1000.0).round()) as usize; // 160

    // Next power of 2 >= frame_len
    let n_fft = (frame_len as u32).next_power_of_two() as usize; // 512
    let n_freqs = n_fft / 2 + 1;

    // Hann window - precompute once
    let window: Vec<f32> = (0..frame_len)
        .map(|i| {
            (0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / frame_len as f32).cos())).sqrt()
        })
        .collect();

    // Pre-emphasis (Kaldi default: 0.97)
    let wav_preemph: Vec<f32> = std::iter::once(wav_float[0])
        .chain(
            wav_float
                .iter()
                .zip(wav_float.iter().skip(1))
                .map(|(&prev, &curr)| curr - 0.97 * prev),
        )
        .collect();

    // Framing (snip_edges=True: only frames that fit completely)
    let n_frames = if wav_preemph.len() >= frame_len {
        1 + (wav_preemph.len() - frame_len) / frame_shift
    } else {
        0
    };

    if n_frames == 0 {
        return Array2::<f32>::zeros((0, NUM_MEL_BINS));
    }

    // Pre-allocate frames as contiguous array for better cache performance
    let mut frames_data = Vec::with_capacity(n_frames * frame_len);
    for i in 0..n_frames {
        let start = i * frame_shift;
        for j in 0..frame_len {
            frames_data.push(wav_preemph[start + j] * window[j]);
        }
    }

    // FFT → power spectrum using rustfft
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    // Pre-allocate buffer once and reuse
    let mut buffer: Vec<num_complex::Complex<f32>> =
        vec![num_complex::Complex::new(0.0, 0.0); n_fft];
    let mut power_spectrum = Vec::with_capacity(n_frames * n_freqs);

    for i in 0..n_frames {
        // Copy frame data to buffer with windowing applied
        let frame_offset = i * frame_len;
        for j in 0..frame_len {
            buffer[j] = num_complex::Complex::new(frames_data[frame_offset + j], 0.0);
        }
        // Zero-pad remaining
        for j in frame_len..n_fft {
            buffer[j] = num_complex::Complex::new(0.0, 0.0);
        }

        fft.process(&mut buffer);

        // Compute power spectrum
        for k in 0..n_freqs {
            power_spectrum.push(buffer[k].norm_sqr());
        }
    }

    // Mel filterbank - precompute transposed for better cache access
    let mel_fb = build_mel_filterbank(
        n_fft,
        NUM_MEL_BINS,
        0.0,
        sample_rate as f32 / 2.0,
        sample_rate,
    );

    // Apply mel filterbank using optimized matrix multiplication
    // power_spectrum: (n_frames, n_freqs), mel_fb: (n_mels, n_freqs)
    // Result: (n_frames, n_mels)
    let mut mel_energy = Array2::<f32>::zeros((n_frames, NUM_MEL_BINS));

    // Optimized: iterate in cache-friendly order, use flat indexing
    for t in 0..n_frames {
        let power_offset = t * n_freqs;
        for m in 0..NUM_MEL_BINS {
            let mut sum = 0.0f32;
            // Manual loop unrolling for better performance
            for k in 0..n_freqs {
                sum += power_spectrum[power_offset + k] * mel_fb[[m, k]];
            }
            mel_energy[[t, m]] = sum;
        }
    }

    // Log compression — floor at 1.0
    mel_energy.mapv_inplace(|v| v.max(1.0).ln());

    mel_energy
}

// ---------------------------------------------------------------------------
// FireRedVad ONNX Inference
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct FireRedVadConfig {
    pub smooth_window_size: usize,
    pub speech_threshold: f32,
    pub min_speech_frame: usize,
    pub max_speech_frame: usize,
    pub min_silence_frame: usize,
    pub merge_silence_frame: usize,
    pub extend_speech_frame: usize,
    pub chunk_max_frame: usize,
}

impl Default for FireRedVadConfig {
    fn default() -> Self {
        Self {
            smooth_window_size: 5,
            speech_threshold: 0.4,
            min_speech_frame: 20,
            max_speech_frame: 2000,
            min_silence_frame: 20,
            merge_silence_frame: 0,
            extend_speech_frame: 0,
            chunk_max_frame: 30000,
        }
    }
}

pub struct FireRedVadOnnx {
    session: Arc<Mutex<Session>>,
    cmvn: Cmvn,
    config: FireRedVadConfig,
    postprocessor: VadPostprocessor,
}

impl FireRedVadOnnx {
    /// Create a new FireRedVad inference instance.
    ///
    /// # Arguments
    /// * `model_dir` - Directory containing model.onnx and cmvn.json
    /// * `config` - Configuration for VAD detection
    /// * `use_coreml` - Whether to use CoreML execution provider (Apple Silicon)
    pub fn new<P: AsRef<Path>>(
        model_dir: P,
        config: FireRedVadConfig,
        use_coreml: bool,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Load CMVN
        let cmvn_path = model_dir.join("cmvn.json");
        let cmvn = Cmvn::from_json_path(&cmvn_path)
            .with_context(|| format!("Failed to load CMVN from {:?}", cmvn_path))?;

        // Load ONNX model
        let model_path = model_dir.join("model.onnx");

        // Select execution provider
        let ep = match use_coreml {
            true => {
                info!("Using CoreML execution provider");
                [CoreML::default().build()]
            }
            false => {
                info!("Using CPU execution provider");
                [CPU::default().build()]
            }
        };
        let session = Session::builder()?
            .with_execution_providers(ep)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(&model_path)
            .with_context(|| format!("Failed to load ONNX model from {:?}", model_path))?;

        let postprocessor = VadPostprocessor::new(
            config.smooth_window_size,
            config.speech_threshold,
            config.min_speech_frame,
            config.max_speech_frame,
            config.min_silence_frame,
            config.merge_silence_frame,
            config.extend_speech_frame,
        );

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            cmvn,
            config,
            postprocessor,
        })
    }

    /// Load and prepare audio from a file.
    ///
    /// This method automatically handles:
    /// - Loading audio from various formats (WAV, MP3, FLAC, etc.)
    /// - Converting stereo to mono
    /// - Resampling to 16kHz if needed
    /// - Converting to int16 for feature extraction
    ///
    /// # Arguments
    /// * `audio_path` - Path to audio file (any supported format)
    ///
    /// # Returns
    /// * `MonoBuffer` - Mono audio buffer at 16kHz with f32 samples
    /// * `Vec<i16>` - int16 samples for feature extraction

    /// Extract features from audio.
    ///
    /// # Returns
    /// CMVN-normalized log-fbank features of shape (T, 80)
    fn extract_features(&self, wav_int16: &[f32]) -> Array2<f32> {
        let fbank = extract_kaldi_fbank(wav_int16, SAMPLE_RATE);
        self.cmvn.apply(&fbank)
    }

    /// Run ONNX model inference.
    ///
    /// # Arguments
    /// * `feat` - Features of shape (T, 80)
    ///
    /// # Returns
    /// Speech probabilities of shape (T,)
    fn run_model(&self, feat: &Array2<f32>) -> Result<Array1<f32>> {
        let t = feat.nrows();
        let mut all_probs = Vec::new();

        for chunk_start in (0..t).step_by(self.config.chunk_max_frame) {
            let chunk_end = (chunk_start + self.config.chunk_max_frame).min(t);
            let chunk = feat.slice(s![chunk_start..chunk_end, ..]);

            // Create input tensor: (1, t, 80)
            let input_tensor = Array::<f32, _>::from_shape_vec(
                (1, chunk.nrows(), chunk.ncols()),
                chunk.iter().copied().collect(),
            )?;
            let input_value = Value::from_array(input_tensor)?;

            // Run inference
            let mut session = self.session.lock().unwrap();
            let outputs = session.run(inputs!["input" => input_value])?;

            // Extract the output data
            let output = &outputs["output"];
            let (_, output_data) = output.try_extract_tensor::<f32>()?;

            // Collect probabilities
            all_probs.extend(output_data.iter().copied());
            // Drop outputs before releasing lock
            drop(outputs);
        }

        Ok(Array1::from_vec(all_probs))
    }

    /// Run VAD detection on an audio file.
    ///
    /// This method automatically handles audio format conversion:
    /// - Any audio format supported by AudioLoader (WAV, MP3, FLAC, OGG, etc.)
    /// - Any sample rate (automatically resampled to 16kHz)
    /// - Any number of channels (automatically converted to mono)
    ///
    /// # Arguments
    /// * `audio_path` - Path to audio file
    ///
    /// # Returns
    /// * `VadResult` - Detection results with segments and duration
    /// * `Array1<f32>` - Raw per-frame speech probabilities
    pub fn detect<P: AsRef<Path>>(&self, audio_path: P) -> Result<(VadResult, Array1<f32>)> {
        let _timer = ScopedTimer::new("FireRedVadOnnx::detect");
        let start_total = std::time::Instant::now();

        // Prepare audio (load, resample, convert to mono)
        let start_prepare = std::time::Instant::now();
        let (mono, samples_i16) = base::prepare_audio_16khz(&audio_path)?;
        let dur = mono.duration_secs();
        info!("Audio preparation took: {:.2?}", start_prepare.elapsed());

        // Extract features
        let start_features = std::time::Instant::now();
        let feat = self.extract_features(&samples_i16);
        info!(
            "Feature extraction took: {:.2?}, shape: {:?}",
            start_features.elapsed(),
            feat.dim()
        );

        // Run model
        let start_model = std::time::Instant::now();
        let probs = self.run_model(&feat)?;
        info!("Model inference took: {:.2?}", start_model.elapsed());

        // Post-process
        let start_post = std::time::Instant::now();
        let probs_vec: Vec<f32> = probs.iter().copied().collect();
        let decisions = self.postprocessor.process(&probs_vec);
        let segments = self
            .postprocessor
            .decisions_to_segments(&decisions, Some(dur as f32));
        info!("Post-processing took: {:.2?}", start_post.elapsed());

        let result = VadResult {
            dur: (dur as f32 * 1000.0).round() / 1000.0,
            timestamps: segments,
            wav_path: audio_path.as_ref().to_string_lossy().to_string(),
        };

        info!("Total VAD detection took: {:.2?}", start_total.elapsed());

        Ok((result, probs))
    }
}

/// VAD detection result
#[derive(Debug, Clone)]
pub struct VadResult {
    /// Duration in seconds
    pub dur: f32,
    /// Speech segments as (start, end) tuples in seconds
    pub timestamps: Vec<(f32, f32)>,
    /// Path to the processed audio file
    pub wav_path: String,
}

impl VadResult {
    /// Get the number of speech segments
    pub fn num_segments(&self) -> usize {
        self.timestamps.len()
    }

    /// Get total speech duration in seconds
    pub fn total_speech_duration(&self) -> f32 {
        self.timestamps.iter().map(|(s, e)| e - s).sum()
    }
}

pub fn firered_vad_test(audio_path: &str, label: &str) -> Result<()> {
    let config = FireRedVadConfig {
        speech_threshold: 0.4,
        min_speech_frame: 20,
        ..Default::default()
    };

    let vad = FireRedVadOnnx::new(
        "/Volumes/sw/pretrained_models/FireRedVAD/VAD/onnx_unified",
        config,
        false,
    )?;

    let (result, probs) = vad.detect(audio_path)?;

    // Calculate average probability for each segment
    for (start, end) in &result.timestamps {
        let start_frame = (*start / FRAME_SHIFT_S) as usize;
        let end_frame = (*end / FRAME_SHIFT_S) as usize;
        let end_frame = end_frame.min(probs.len());

        let avg_prob = if start_frame >= end_frame {
            0.0
        } else {
            let segment_probs: Vec<f32> = probs
                .slice(s![start_frame..end_frame])
                .iter()
                .copied()
                .collect();
            segment_probs.iter().sum::<f32>() / segment_probs.len() as f32
        };

        println!(
            "Speech: {:.3}s - {:.3}s (duration: {:.3}s, avg_prob: {:.3})",
            start,
            end,
            end - start,
            avg_prob
        );
    }

    println!(
        "\nSummary: {} segments, total duration: {:.3}s",
        result.timestamps.len(),
        result.dur
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hz_to_mel() {
        // Test basic conversion
        let mel = hz_to_mel(1000.0);
        assert!(mel > 0.0);
    }

    #[test]
    fn test_mel_to_hz() {
        // Test round-trip with relaxed tolerance due to floating point precision
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let hz_back = mel_to_hz(mel);
        assert!(
            (hz - hz_back).abs() < 1e-4,
            "Round-trip: {} -> {} -> {}, error: {}",
            hz,
            mel,
            hz_back,
            (hz - hz_back).abs()
        );
    }

    #[test]
    fn test_build_mel_filterbank() {
        let fb = build_mel_filterbank(512, 80, 0.0, 8000.0, 16000);
        assert_eq!(fb.shape(), &[80, 257]);
        // Check that filters have non-zero values
        let total_sum: f32 = fb.sum();
        assert!(
            total_sum > 50.0 && total_sum < 300.0,
            "Total filterbank sum: {}",
            total_sum
        );
    }
}
