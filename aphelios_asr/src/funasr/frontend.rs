use candle_core::{Device, Result, Tensor};
use rustfft::{num_complex::Complex as FftComplex, FftPlanner};
use std::f32::consts::PI;

pub struct WavFrontend {
    n_mels: usize,
    frame_length: usize,
    frame_shift: usize,
    lfr_m: usize,
    lfr_n: usize,
    mel_filters: Tensor,
}

impl WavFrontend {
    pub fn new(
        n_mels: usize,
        sample_rate: usize,
        frame_ms: usize,
        shift_ms: usize,
        lfr_m: usize,
        lfr_n: usize,
        device: &Device,
    ) -> Result<Self> {
        let frame_length = (sample_rate * frame_ms) / 1000;
        let frame_shift = (sample_rate * shift_ms) / 1000;

        // For now, let's create simple mel filters or load them if possible.
        // As a "basic" version, we'll try to generate them here.
        let n_fft = frame_length.next_power_of_two();
        let mel_filters = Self::get_mel_filters(n_mels, n_fft, sample_rate, device)?;

        Ok(Self {
            n_mels,
            frame_length,
            frame_shift,
            lfr_m,
            lfr_n,
            mel_filters,
        })
    }

    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
    }

    fn get_mel_filters(
        n_mels: usize,
        n_fft: usize,
        sample_rate: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let f_min = 0.0f32;
        let f_max = (sample_rate / 2) as f32;
        let mel_min = Self::hz_to_mel(f_min);
        let mel_max = Self::hz_to_mel(f_max);

        let mels: Vec<f32> = (0..n_mels + 2)
            .map(|i| mel_min + (mel_max - mel_min) * (i as f32) / ((n_mels + 1) as f32))
            .collect();
        let hzs: Vec<f32> = mels.iter().map(|&m| Self::mel_to_hz(m)).collect();
        let bins: Vec<usize> = hzs
            .iter()
            .map(|&h| (h * (n_fft + 1) as f32 / sample_rate as f32) as usize)
            .collect();

        let mut filters = vec![0.0f32; n_mels * (n_fft / 2 + 1)];
        for i in 0..n_mels {
            for j in bins[i]..bins[i + 1] {
                filters[i * (n_fft / 2 + 1) + j] =
                    (j - bins[i]) as f32 / (bins[i + 1] - bins[i]) as f32;
            }
            for j in bins[i + 1]..bins[i + 2] {
                filters[i * (n_fft / 2 + 1) + j] =
                    (bins[i + 2] - j) as f32 / (bins[i + 2] - bins[i + 1]) as f32;
            }
        }

        Tensor::from_vec(filters, (n_mels, n_fft / 2 + 1), device)
    }

    fn hamming_window(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / (n - 1) as f32).cos())
            .collect()
    }

    pub fn forward(&self, pcm: &Tensor) -> Result<Tensor> {
        let (batch, n_samples) = pcm.dims2()?;
        let device = pcm.device();

        // 1. Setup FFT
        let n_fft = self.frame_length.next_power_of_two();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);
        let window = Self::hamming_window(self.frame_length);

        let mut all_batch_mels = Vec::new();
        for b in 0..batch {
            let pcm_vec = pcm.get(b)?.to_vec1::<f32>()?;
            let n_frames = if n_samples >= self.frame_length {
                (n_samples - self.frame_length) / self.frame_shift + 1
            } else {
                0
            };

            let mut mel_frames = Vec::with_capacity(n_frames);
            for i in 0..n_frames {
                let start = i * self.frame_shift;
                let mut buffer: Vec<FftComplex<f32>> = (0..n_fft)
                    .map(|j| {
                        if j < self.frame_length && start + j < pcm_vec.len() {
                            FftComplex::new(pcm_vec[start + j] * window[j], 0.0)
                        } else {
                            FftComplex::new(0.0, 0.0)
                        }
                    })
                    .collect();

                fft.process(&mut buffer);

                // Magnitude squared
                let power_spec: Vec<f32> = buffer
                    .iter()
                    .take(n_fft / 2 + 1)
                    .map(|c| c.norm_sqr())
                    .collect();

                let power_spec_t = Tensor::from_vec(power_spec, (1, n_fft / 2 + 1), device)?;
                // Apply mel filters: mel_filters is [n_mels, n_fft/2 + 1]
                let mel_bin = power_spec_t.matmul(&self.mel_filters.t()?)?; // [1, n_mels]

                // Log and clamping
                let mel_log = mel_bin.clamp(1e-10f32, f32::MAX)?.log()?;
                mel_frames.push(mel_log);
            }

            if mel_frames.is_empty() {
                all_batch_mels.push(Tensor::zeros(
                    (0, self.n_mels),
                    candle_core::DType::F32,
                    device,
                )?);
            } else {
                let batch_mel = Tensor::cat(&mel_frames, 0)?; // [n_frames, n_mels]
                all_batch_mels.push(batch_mel);
            }
        }

        let feats = Tensor::stack(&all_batch_mels, 0)?; // [batch, T, n_mels]
        self.apply_lfr(&feats)
    }

    pub fn apply_lfr(&self, feats: &Tensor) -> Result<Tensor> {
        // feats: [batch, T, 80]
        // lfr_m = 7, lfr_n = 6
        // padding at the beginning: (lfr_m - 1) / 2 = 3
        let (b, t, d) = feats.dims3()?;
        let left_padding = 3;

        // pad feats with repetitions of the first frame
        let first_frame = feats.narrow(1, 0, 1)?;
        let mut padded_feats = feats.clone();
        for _ in 0..left_padding {
            padded_feats = Tensor::cat(&[&first_frame, &padded_feats], 1)?;
        }

        let mut stacked = Vec::new();
        let mut t_idx = 0;
        while t_idx + self.lfr_m <= padded_feats.dim(1)? {
            let chunk = padded_feats.narrow(1, t_idx, self.lfr_m)?;
            // Flatten the chunk: [batch, 1, lfr_m * 80]
            let chunk_flat = chunk.reshape((b, 1, self.lfr_m * d))?;
            stacked.push(chunk_flat);
            t_idx += self.lfr_n;
        }

        if stacked.is_empty() {
            return Ok(Tensor::zeros(
                (b, 0, self.lfr_m * d),
                feats.dtype(),
                feats.device(),
            )?);
        }

        Tensor::cat(&stacked, 1)
    }
}
