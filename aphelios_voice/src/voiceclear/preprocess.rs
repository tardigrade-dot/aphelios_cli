// SeamlessM4TFeatureExtractor — mel filterbank feature extraction.
// Implements the exact pipeline from HuggingFace transformers:
//   DC removal → pre-emphasis → STFT → |·|² → mel → ln → ZMUV → frame-stack
//
// Parameters (from facebook/w2v-bert-2.0):
//   sample_rate: 16000, n_fft: 512, hop: 160, win: 400, mel_bins: 80, stride: 2

use rustfft::{num_complex::Complex, Fft, FftPlanner};

const N_FFT: usize = 512;
const N_FREQ: usize = N_FFT / 2 + 1; // 257
const HOP_LEN: usize = 160;
const WIN_LEN: usize = 400;
const N_MEL: usize = 80;
const STRIDE: usize = 2;
const FEAT_DIM: usize = N_MEL * STRIDE; // 160

const PREEMPH: f64 = 0.97;
const MEL_FLOOR: f32 = 1.192_092_9e-7; // 2^-23
const ZMUV_EPS: f32 = 1e-7;

/// Compute mel filterbank features [T_frames, 160] from 16kHz mono audio.
/// Uses f64 internally (matching Python's float64 pipeline) for accuracy.
pub fn compute_mel_features(audio_16k: &[f32], mel_filters: &Vec<f64>, window: &Vec<f64>) -> Vec<f32> {
    let n = audio_16k.len();
    if n < WIN_LEN {
        return vec![];
    }

    // Kaldi compliance: scale to 16-bit integer range
    let scaled: Vec<f64> = audio_16k.iter().map(|&x| x as f64 * 32768.0).collect();

    // Frame count: matches HF `1 + floor((len - frame_length) / hop_length)`
    let n_frames = 1 + (n.saturating_sub(WIN_LEN)) / HOP_LEN;

    // FFT planner (f64 for precision matching Python float64)
    let mut planner = FftPlanner::<f64>::new();
    let fft: std::sync::Arc<dyn Fft<f64>> = planner.plan_fft_forward(N_FFT);

    let mut mel_power = vec![0.0f64; n_frames * N_MEL];
    let mut frame_buf = vec![Complex::new(0.0f64, 0.0f64); N_FFT];

    for t in 0..n_frames {
        let start = t * HOP_LEN;

        // Copy frame
        for i in 0..WIN_LEN {
            frame_buf[i] = Complex::new(scaled[start + i], 0.0);
        }
        for i in WIN_LEN..N_FFT {
            frame_buf[i] = Complex::new(0.0, 0.0);
        }

        // Per-frame DC removal
        let mut frame_mean = 0.0f64;
        for i in 0..WIN_LEN {
            frame_mean += frame_buf[i].re;
        }
        frame_mean /= WIN_LEN as f64;
        for i in 0..WIN_LEN {
            frame_buf[i].re -= frame_mean;
        }

        // Pre-emphasis
        {
            let p = PREEMPH;
            let buf = &mut frame_buf[..WIN_LEN];
            let one_minus_p = 1.0 - p;
            for i in (1..WIN_LEN).rev() {
                buf[i].re -= p * buf[i - 1].re;
            }
            buf[0].re *= one_minus_p;
        }

        // Apply window
        for i in 0..WIN_LEN {
            frame_buf[i].re *= window[i];
        }

        // FFT
        fft.process(&mut frame_buf);

        // Power spectrum → mel filterbank
        let off = t * N_MEL;
        for m in 0..N_MEL {
            let mut sum = 0.0f64;
            for k in 0..N_FREQ {
                let pwr = frame_buf[k].re * frame_buf[k].re + frame_buf[k].im * frame_buf[k].im;
                sum += pwr * mel_filters[k * N_MEL + m];
            }
            mel_power[off + m] = sum.max(MEL_FLOOR as f64).ln();
        }
    }

    // Convert to f32 for ZMUV (matching Python's float32 output from spectrogram)
    let mel_power_f32: Vec<f32> = mel_power.iter().map(|&v| v as f32).collect();

    // ZMUV normalization per mel bin
    let eps = ZMUV_EPS;
    let mut mel_norm = mel_power_f32.clone();
    for m in 0..N_MEL {
        let mut sum = 0.0f64;
        for t in 0..n_frames {
            sum += mel_norm[t * N_MEL + m] as f64;
        }
        let mn = (sum / n_frames as f64) as f32;

        let mut ssq = 0.0f64;
        for t in 0..n_frames {
            let d = (mel_norm[t * N_MEL + m] - mn) as f64;
            ssq += d * d;
        }
        let std = ((ssq / (n_frames - 1) as f64) as f32).sqrt() + eps;

        for t in 0..n_frames {
            mel_norm[t * N_MEL + m] = (mel_norm[t * N_MEL + m] - mn) / std;
        }
    }

    // Pad to multiple of stride
    let n_padded = ((n_frames + STRIDE - 1) / STRIDE) * STRIDE;
    if n_padded > n_frames {
        mel_norm.resize(n_padded * N_MEL, 0.0);
    }

    // Frame stacking: stride=2 → [T//2, 160]
    let n_stacked = n_padded / STRIDE;
    let mut feats = vec![0.0f32; n_stacked * FEAT_DIM];
    for t in 0..n_stacked {
        let src0 = t * STRIDE * N_MEL;
        let src1 = src0 + N_MEL;
        let dst = t * FEAT_DIM;
        feats[dst..dst + N_MEL].copy_from_slice(&mel_norm[src0..src0 + N_MEL]);
        feats[dst + N_MEL..dst + FEAT_DIM].copy_from_slice(&mel_norm[src1..src1 + N_MEL]);
    }

    feats
}

/// Simple linear-interpolation resampler for mono audio.
pub fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() || from_rate == to_rate {
        return input.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = (input.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = (src_pos - src_idx as f64) as f32;
        if src_idx + 1 < input.len() {
            output.push(input[src_idx] * (1.0 - frac) + input[src_idx + 1] * frac);
        } else {
            output.push(*input.last().unwrap());
        }
    }
    output
}

/// 2nd-order biquad high-pass filter at 50 Hz for 16 kHz audio.
/// Coefficients from scipy.signal.butter(2, 50/8000, 'high').
pub fn highpass_50hz(samples: &mut [f32]) {
    const B: [f32; 3] = [0.98657227, -1.97314453, 0.98657227];
    const A: [f32; 3] = [1.0, -1.97296524, 0.97331703];
    biquad_inplace(samples, &B, &A);
}

fn biquad_inplace(x: &mut [f32], b: &[f32; 3], a: &[f32; 3]) {
    let a0 = a[0];
    let b0 = b[0] / a0;
    let b1 = b[1] / a0;
    let b2 = b[2] / a0;
    let a1 = a[1] / a0;
    let a2 = a[2] / a0;

    let (mut x1, mut x2, mut y1, mut y2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for xi in x.iter_mut() {
        let yi = b0 * *xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1;
        x1 = *xi;
        y2 = y1;
        y1 = yi;
        *xi = yi;
    }
}

/// Peak-normalize audio to target_peak (e.g. 0.9).
pub fn peak_normalize(samples: &mut [f32], target_peak: f32) {
    let max_abs = samples.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    if max_abs < 1e-10 {
        return;
    }
    let scale = target_peak / max_abs;
    for v in samples.iter_mut() {
        *v *= scale;
    }
}

/// Chunk iterator for large audio. Yields overlapping chunks of `chunk_samples`
/// with `pad` extra samples on each side.
pub struct ChunkIterator<'a> {
    audio: &'a [f32],
    chunk_samples: usize,
    pad: usize,
    pos: usize,
}

impl<'a> ChunkIterator<'a> {
    pub fn new(audio: &'a [f32], chunk_duration_s: f32, pad_samples: usize) -> Self {
        let chunk_samples = (chunk_duration_s * 16_000.0) as usize;
        Self { audio, chunk_samples, pad: pad_samples, pos: 0 }
    }
}

impl<'a> Iterator for ChunkIterator<'a> {
    type Item = Vec<f32>;

    fn next(&mut self) -> Option<Vec<f32>> {
        if self.pos >= self.audio.len() {
            return None;
        }
        let start = self.pos.saturating_sub(self.pad);
        let end = (self.pos + self.chunk_samples + self.pad).min(self.audio.len());
        let chunk = self.audio[start..end].to_vec();
        self.pos += self.chunk_samples;
        Some(chunk)
    }
}
