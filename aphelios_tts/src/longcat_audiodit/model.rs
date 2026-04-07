use crate::longcat_audiodit::{
    config::AudioDiTConfig,
    device::{default_dtype, select_device},
    loader::{ModelPaths, WeightIndex, WeightSummary},
    scheduler::DiffusionScheduler,
    text_encoder::{ensure_tokenizer_path, EncodedTextBatch, LongCatTextEncoder},
    transformer::{AudioDiTTransformer, TransformerForwardInput},
    vae::AudioVae,
};
use crate::audio::resample::{ResampleQuality, Resampler};
use anyhow::{ensure, Context, Result};
use aphelios_core::audio::loader::AudioLoader;
use candle_core::{DType, Device, Tensor, D};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuidanceMethod {
    Cfg,
    Apg,
}

impl GuidanceMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cfg => "cfg",
            Self::Apg => "apg",
        }
    }

    pub fn from_cli(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "cfg" => Ok(Self::Cfg),
            "apg" => Ok(Self::Apg),
            _ => anyhow::bail!("invalid guidance method: {s}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LongCatInferenceConfig {
    pub cpu: bool,
    pub tokenizer_path: Option<PathBuf>,
    pub dtype: Option<DType>,
}

impl Default for LongCatInferenceConfig {
    fn default() -> Self {
        Self {
            cpu: false,
            tokenizer_path: None,
            dtype: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LongCatSynthesisRequest {
    pub text: String,
    pub prompt_text: Option<String>,
    pub prompt_audio: Option<PathBuf>,
    pub duration: Option<usize>,
    pub steps: usize,
    pub cfg_strength: f64,
    pub guidance_method: GuidanceMethod,
    pub seed: u64,
}

impl LongCatSynthesisRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            prompt_text: None,
            prompt_audio: None,
            duration: None,
            steps: 16,
            cfg_strength: 4.0,
            guidance_method: GuidanceMethod::Cfg,
            seed: 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferencePlan {
    pub sample_rate: u32,
    pub latent_hop: usize,
    pub prompt_audio_samples: usize,
    pub prompt_audio_frames: usize,
    pub target_frames: usize,
    pub total_frames: usize,
    pub scheduler: DiffusionScheduler,
}

#[derive(Debug)]
pub struct LongCatAudioDiT {
    pub paths: ModelPaths,
    pub config: AudioDiTConfig,
    pub device: Device,
    pub dtype: DType,
    pub weights: WeightSummary,
    pub text_encoder: LongCatTextEncoder,
    pub transformer: AudioDiTTransformer,
    pub vae: AudioVae,
}

impl LongCatAudioDiT {
    pub fn from_pretrained(
        model_dir: impl AsRef<Path>,
        options: LongCatInferenceConfig,
    ) -> Result<Self> {
        let device = select_device(options.cpu)?;
        let dtype = options.dtype.unwrap_or_else(|| default_dtype(&device));
        let paths = ModelPaths::discover(model_dir, options.tokenizer_path.as_ref())?;
        let config = paths.load_config()?;
        let index = WeightIndex::from_file(&paths.weights)?;
        let tokenizer_path = ensure_tokenizer_path(paths.tokenizer.as_deref())?;
        let text_encoder = LongCatTextEncoder::load(
            &config,
            &paths.weights,
            &tokenizer_path,
            DType::F32,
            &device,
        )?;
        let transformer = AudioDiTTransformer::load(&config, &paths.weights, dtype, &device)?;
        let vae = AudioVae::load(&config.vae_config, &paths.weights, &device)?;

        Ok(Self {
            paths,
            config,
            device,
            dtype,
            weights: index.summary(),
            text_encoder,
            transformer,
            vae,
        })
    }

    pub fn tokenizer_path(&self) -> Option<&Path> {
        self.paths.tokenizer.as_deref()
    }

    pub fn prepare_text_batch(
        &self,
        request: &LongCatSynthesisRequest,
    ) -> Result<EncodedTextBatch> {
        let text = LongCatTextEncoder::normalize_text(&request.text);
        let full_text = match request.prompt_text.as_deref() {
            Some(prompt_text) => {
                let prompt = LongCatTextEncoder::normalize_text(prompt_text);
                format!("{prompt} {text}")
            }
            None => text,
        };
        let mut encoded = self
            .text_encoder
            .encode_batch(&[full_text], &self.device)
            .context("failed to encode LongCat text condition")?;
        // Ensure hidden states match transformer's expected dtype
        encoded.hidden_states = encoded.hidden_states.to_dtype(self.dtype)?;
        Ok(encoded)
    }

    pub fn validate_request(&self, request: &LongCatSynthesisRequest) -> Result<()> {
        validate_request_payload(request)
    }

    pub fn plan_inference(&self, request: &LongCatSynthesisRequest) -> Result<InferencePlan> {
        ensure!(
            self.validate_request(request).is_ok(),
            "request validation should run before planning"
        );
        self.validate_request(request)?;

        let prompt_audio_samples = match request.prompt_audio.as_ref() {
            Some(path) => load_prompt_audio_sample_count(path, self.config.sampling_rate)?,
            None => 0,
        };
        let prompt_audio_frames = prompt_audio_samples.div_ceil(self.config.latent_hop);
        let total_frames = request.duration.unwrap_or_else(|| {
            // Match Python duration estimation logic from approx_duration_from_text
            let sr = self.config.sampling_rate as f64;
            let full_hop = self.config.latent_hop as f64;
            let max_duration = self.config.max_wav_duration as f64;

            // Calculate prompt time from actual prompt audio frames
            let prompt_time = prompt_audio_frames as f64 * full_hop / sr;

            // Estimate text duration (matching Python's approx_duration_from_text)
            let text_dur = approx_duration_from_text(&request.text, max_duration - prompt_time);

            // Apply ratio adjustment if there's a prompt
            let duration = if request.prompt_audio.is_some() {
                let prompt_text = request.prompt_text.as_deref().unwrap_or_default();
                let approx_pd = approx_duration_from_text(prompt_text, max_duration);
                let ratio = (prompt_time / approx_pd).clamp(1.0, 1.5);
                text_dur * ratio
            } else {
                text_dur
            };

            // Convert duration to frames - match Python: int(dur_sec * sr // full_hop)
            let gen_frames = (duration * sr / full_hop).floor() as usize;
            prompt_audio_frames + gen_frames
        });
        ensure!(
            total_frames <= self.config.max_latent_frames(),
            "requested duration exceeds model max_wav_duration={}s ({} latent frames)",
            self.config.max_wav_duration,
            self.config.max_latent_frames()
        );
        ensure!(
            total_frames >= prompt_audio_frames,
            "requested duration {} is shorter than prompt audio frames {}",
            total_frames,
            prompt_audio_frames
        );

        let target_frames = total_frames.saturating_sub(prompt_audio_frames);
        let scheduler = DiffusionScheduler::new(
            request.steps,
            request.cfg_strength,
            request.guidance_method,
            self.config.sigma,
        );

        Ok(InferencePlan {
            sample_rate: self.config.sampling_rate,
            latent_hop: self.config.latent_hop,
            prompt_audio_samples,
            prompt_audio_frames,
            target_frames,
            total_frames,
            scheduler,
        })
    }

    pub fn synthesize(&self, request: &LongCatSynthesisRequest) -> Result<Vec<f32>> {
        let mut plan = self.plan_inference(request)?;
        let encoded_text = self.prepare_text_batch(request)?;

        // Load prompt audio and VAE encode if present
        let (_prompt_audio_samples, prompt_latent) = if let Some(audio_path) = &request.prompt_audio
        {
            let audio = AudioLoader::new().load(audio_path).with_context(|| {
                format!(
                    "failed to load LongCat prompt audio: {}",
                    audio_path.display()
                )
            })?;
            let mut mono = audio.into_mono();

            // Resample to target sample rate if needed
            if mono.sample_rate != self.config.sampling_rate {
                let audio_buf = crate::audio::AudioBuffer::new(mono.samples.clone(), mono.sample_rate);
                let resampler = Resampler::new(ResampleQuality::High);
                let resampled = resampler.resample(&audio_buf, self.config.sampling_rate)?;
                mono.samples = resampled.samples;
                mono.sample_rate = resampled.sample_rate;
            }

            // Match Python padding: pad to multiple of latent_hop, then add 3*latent_hop zeros
            let hop = self.config.latent_hop;
            let off = 3;
            if mono.samples.len() % hop != 0 {
                let pad_len = hop - (mono.samples.len() % hop);
                mono.samples.extend(std::iter::repeat_n(0.0, pad_len));
            }
            mono.samples.extend(std::iter::repeat_n(0.0, hop * off));

            let samples_len = mono.samples.len();
            // Convert to [1, 1, T] tensor for VAE
            let audio_tensor = Tensor::from_vec(mono.samples.clone(), (1, 1, samples_len), &self.device)?;

            // VAE encode: [1, 1, T] -> [1, latent_dim, frames]
            let latent_2d = self.vae.encode(&audio_tensor)?;
            // Remove padding frames (off=3, like Python code)
            let latent_no_pad = if latent_2d.dim(D::Minus1)? > off {
                latent_2d.narrow(D::Minus1, 0, latent_2d.dim(D::Minus1)? - off)?
            } else {
                latent_2d
            };
            // Permute to [1, frames, latent_dim] for transformer
            let prompt_latent_t = latent_no_pad.permute([0, 2, 1])?.to_dtype(self.dtype)?;

            (samples_len, prompt_latent_t)
        } else {
            (
                0,
                Tensor::zeros((1, 0, self.config.latent_dim), self.dtype, &self.device)?,
            )
        };

        // Re-sync plan with actual encoded prompt frames
        let prompt_frames = prompt_latent.dim(1)?;
        if prompt_frames != plan.prompt_audio_frames {
            // Re-calculate target frames based on actual prompt frames
            // Use the same duration estimation logic
            let sr = self.config.sampling_rate as f64;
            let full_hop = self.config.latent_hop as f64;
            let max_duration = self.config.max_wav_duration as f64;
            let prompt_time = prompt_frames as f64 * full_hop / sr;
            let text_dur = approx_duration_from_text(&request.text, max_duration - prompt_time);
            let duration = if request.prompt_audio.is_some() {
                let prompt_text = request.prompt_text.as_deref().unwrap_or_default();
                let approx_pd = approx_duration_from_text(prompt_text, max_duration);
                let ratio = (prompt_time / approx_pd).clamp(1.0, 1.5);
                text_dur * ratio
            } else {
                text_dur
            };
            let gen_frames = (duration * sr / full_hop).floor() as usize;
            let new_total = prompt_frames + gen_frames;

            plan.prompt_audio_frames = prompt_frames;
            plan.total_frames = new_total.max(plan.total_frames);
            plan.target_frames = plan.total_frames.saturating_sub(prompt_frames);
        }

        // Prepare masks
        let total_frames = plan.total_frames;
        let gen_frames = plan.target_frames;
        let prompt_frames = plan.prompt_audio_frames;

        let latent_mask = Tensor::ones((1, total_frames), DType::U32, &self.device)?;
        let text_mask = &encoded_text.attention_mask;
        let text_len = &encoded_text.lengths;

        // Negative text (for CFG)
        let neg_text = Tensor::zeros_like(&encoded_text.hidden_states)?;
        let neg_text_len = text_len.clone();

        // Initial noise (y0) - use Candle's built-in randn
        let y0 = Tensor::randn(
            0.0f32,
            1.0f32,
            (1, total_frames, self.config.latent_dim),
            &self.device,
        )?
        .to_dtype(self.dtype)?;

        // Extract prompt noise for conditioning
        let prompt_noise = if prompt_frames > 0 {
            y0.narrow(D::Minus2, 0, prompt_frames)?
        } else {
            Tensor::zeros((1, 0, self.config.latent_dim), self.dtype, &self.device)?
        };

        // Latent conditioning
        let latent_cond = if prompt_frames > 0 {
            // Pad prompt latent to total duration
            let pad = Tensor::zeros(
                (1, gen_frames, self.config.latent_dim),
                self.dtype,
                &self.device,
            )?;
            Tensor::cat(&[&prompt_latent, &pad], D::Minus2)?
        } else {
            Tensor::zeros(
                (1, total_frames, self.config.latent_dim),
                self.dtype,
                &self.device,
            )?
        };
        let empty_latent_cond = Tensor::zeros_like(&latent_cond)?;

        // APG momentum buffer
        let mut apg_momentum = ApgMomentumBuffer::new(-0.3);

        // Euler ODE integration
        let timesteps = plan.scheduler.timesteps();
        let mut x = y0.clone();

        for i in 0..timesteps.len() - 1 {
            let t_current = timesteps[i];
            let t_next = timesteps[i + 1];
            let dt = t_next - t_current;

            // Evaluate velocity function
            let velocity = self.evaluate_velocity(
                &x,
                &encoded_text.hidden_states,
                text_len,
                t_current,
                &latent_mask,
                text_mask,
                &latent_cond,
                &empty_latent_cond,
                &prompt_noise,
                prompt_frames,
                gen_frames,
                &neg_text,
                &neg_text_len,
                plan.scheduler.cfg_strength,
                plan.scheduler.guidance_method,
                &mut apg_momentum,
            )?;

            // Euler step: x_{t+1} = x_t + velocity * dt
            let dt_tensor = Tensor::new(&[dt], &self.device)?.to_dtype(self.dtype)?;
            x = x.add(&velocity.broadcast_mul(&dt_tensor)?)?;
        }

        let sampled = x;

        // Extract generated portion (skip prompt)
        let pred_latent = if prompt_frames > 0 {
            sampled.narrow(D::Minus2, prompt_frames, gen_frames)?
        } else {
            sampled
        };

        // Permute to [B, latent_dim, frames] for VAE
        let pred_latent = pred_latent.permute([0, 2, 1])?.to_dtype(DType::F32)?;

        // VAE decode to waveform: [1, latent_dim, frames] -> [1, 1, T]
        let waveform_2d = self.vae.decode(&pred_latent)?;

        // Flatten to mono audio samples
        let waveform_vec: Vec<f32> = waveform_2d.flatten_all()?.to_vec1()?;
        Ok(waveform_vec)
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_velocity(
        &self,
        x: &Tensor,
        text: &Tensor,
        text_len: &Tensor,
        t: f32,
        mask: &Tensor,
        text_mask: &Tensor,
        latent_cond: &Tensor,
        empty_latent_cond: &Tensor,
        prompt_noise: &Tensor,
        prompt_frames: usize,
        gen_frames: usize,
        neg_text: &Tensor,
        neg_text_len: &Tensor,
        cfg_strength: f64,
        guidance_method: GuidanceMethod,
        apg_momentum: &mut ApgMomentumBuffer,
    ) -> Result<Tensor> {
        let t_tensor = Tensor::new(&[t], &self.device)?.to_dtype(self.dtype)?;

        // Condition the latent with prompt: path y(t) = prompt_noise * (1-t) + prompt_latent * t
        let mut x_cond = x.clone();
        if prompt_frames > 0 {
            let one_minus_t = Tensor::new(&[1.0 - t], &self.device)?.to_dtype(self.dtype)?;
            let t_scalar = Tensor::new(&[t], &self.device)?.to_dtype(self.dtype)?;
            let prompt_part = prompt_noise
                .broadcast_mul(&one_minus_t)?
                .add(
                    &latent_cond
                        .narrow(D::Minus2, 0, prompt_frames)?
                        .broadcast_mul(&t_scalar)?,
                )?;
            let remaining = x_cond.narrow(
                D::Minus2,
                prompt_frames,
                x_cond.dim(D::Minus2)? - prompt_frames,
            )?;
            x_cond = Tensor::cat(&[&prompt_part, &remaining], D::Minus2)?;
        }

        // Forward pass (conditional)
        let output = self.transformer.forward(TransformerForwardInput {
            x: &x_cond,
            text: &text.to_dtype(self.dtype)?,
            text_len,
            time: &t_tensor,
            mask: Some(mask),
            cond_mask: Some(text_mask),
            latent_cond: Some(latent_cond),
        })?;
        let pred = output.last_hidden_state;

        if cfg_strength < 1e-5 {
            return Ok(pred);
        }

        // Unconditional pass: zero out prompt portion
        let mut x_uncond = x.clone();
        if prompt_frames > 0 {
            let zeros = Tensor::zeros(
                (1, prompt_frames, x_uncond.dim(D::Minus1)?),
                self.dtype,
                &self.device,
            )?;
            let remaining = x_uncond.narrow(
                D::Minus2,
                prompt_frames,
                x_uncond.dim(D::Minus2)? - prompt_frames,
            )?;
            x_uncond = Tensor::cat(&[&zeros, &remaining], D::Minus2)?;
        }

        let null_output = self.transformer.forward(TransformerForwardInput {
            x: &x_uncond,
            text: &neg_text.to_dtype(self.dtype)?,
            text_len: neg_text_len,
            time: &t_tensor,
            mask: Some(mask),
            cond_mask: Some(text_mask),
            latent_cond: Some(empty_latent_cond),
        })?;
        let null_pred = null_output.last_hidden_state;

        match guidance_method {
            GuidanceMethod::Cfg => {
                let cfg_scale = Tensor::new(&[cfg_strength as f32], &self.device)?.to_dtype(self.dtype)?;
                Ok(pred.add(&pred.sub(&null_pred)?.broadcast_mul(&cfg_scale)?)?)
            }
            GuidanceMethod::Apg => {
                let gen_start = prompt_frames;
                let x_s = x.narrow(D::Minus2, gen_start, x.dim(D::Minus2)? - gen_start)?;
                let pred_s = pred.narrow(D::Minus2, gen_start, pred.dim(D::Minus2)? - gen_start)?;
                let null_s = null_pred.narrow(
                    D::Minus2,
                    gen_start,
                    null_pred.dim(D::Minus2)? - gen_start,
                )?;

                let one_minus_t = Tensor::new(&[1.0 - t], &self.device)?.to_dtype(self.dtype)?;
                let pred_sample = x_s.add(&pred_s.broadcast_mul(&one_minus_t)?)?;
                let null_sample = x_s.add(&null_s.broadcast_mul(&one_minus_t)?)?;

                let out = apg_forward(&pred_sample, &null_sample, cfg_strength, apg_momentum)?;
                let velocity = out.sub(&x_s)?.broadcast_div(&one_minus_t)?;

                if prompt_frames > 0 {
                    let prompt_vel = latent_cond
                        .narrow(D::Minus2, 0, prompt_frames)?
                        .sub(prompt_noise)?;
                    Ok(Tensor::cat(&[&prompt_vel, &velocity], D::Minus2)?)
                } else {
                    Ok(velocity)
                }
            }
        }
    }
}

/// APG Momentum Buffer
struct ApgMomentumBuffer {
    momentum: f32,
    running_average: Option<Tensor>,
}

impl ApgMomentumBuffer {
    fn new(momentum: f32) -> Self {
        Self {
            momentum,
            running_average: None,
        }
    }

    fn update(&mut self, update_value: &Tensor) -> Result<()> {
        let new_avg = if let Some(ref running) = self.running_average {
            let momentum_tensor = Tensor::new(&[self.momentum], update_value.device())?.to_dtype(update_value.dtype())?;
            update_value.add(&running.broadcast_mul(&momentum_tensor)?)?
        } else {
            update_value.clone()
        };
        self.running_average = Some(new_avg);
        Ok(())
    }

    fn get(&self) -> Option<&Tensor> {
        self.running_average.as_ref()
    }
}

/// APG forward pass
fn apg_forward(
    pred_cond: &Tensor,
    pred_uncond: &Tensor,
    guidance_scale: f64,
    momentum_buffer: &mut ApgMomentumBuffer,
) -> Result<Tensor> {
    let diff = pred_cond.sub(pred_uncond)?;
    momentum_buffer.update(&diff)?;

    let diff = momentum_buffer.get().cloned().unwrap_or(diff);

    // Project diff - match Python dims=[-1, -2] (feature and time dimensions)
    // For tensor shape [batch, time, feature], this is dims=[1, 2]
    let (diff_parallel, diff_orthogonal) = project(&diff, pred_cond, &[1, 2])?;

    // eta = 0.5 (from Python code)
    let eta = 0.5;
    let eta_tensor = Tensor::new(&[eta as f32], pred_cond.device())?.to_dtype(pred_cond.dtype())?;
    let normalized_update = diff_orthogonal
        .add(&diff_parallel.broadcast_mul(&eta_tensor)?)?;

    let guidance_scale_tensor = Tensor::new(&[guidance_scale as f32], pred_cond.device())?.to_dtype(pred_cond.dtype())?;
    Ok(pred_cond.add(&normalized_update.broadcast_mul(&guidance_scale_tensor)?)?)
}

/// Project vector onto parallel and orthogonal components
fn project(v0: &Tensor, v1: &Tensor, dims: &[usize]) -> Result<(Tensor, Tensor)> {
    // Normalize v1
    let v1_norm = normalize_dims(v1, dims)?;

    // Parallel component: (v0 · v1_norm) * v1_norm
    let dot_product = (v0.broadcast_mul(&v1_norm)?).sum_keepdim(dims)?;
    let v0_parallel = v1_norm.broadcast_mul(&dot_product)?;

    // Orthogonal component: v0 - v0_parallel
    let v0_orthogonal = v0.sub(&v0_parallel)?;

    Ok((v0_parallel, v0_orthogonal))
}

fn normalize_dims(x: &Tensor, dims: &[usize]) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(dims)?.sqrt()?;
    let norm_safe =
        norm.add(&Tensor::full(1e-12f32, norm.shape(), norm.device())?.to_dtype(norm.dtype())?)?;
    Ok(x.broadcast_div(&norm_safe)?)
}

fn load_prompt_audio_sample_count(path: &Path, target_sample_rate: u32) -> Result<usize> {
    let audio = AudioLoader::new()
        .load(path)
        .with_context(|| format!("failed to load LongCat prompt audio: {}", path.display()))?;
    let mono = audio.into_mono();
    let sample_count = if mono.sample_rate == target_sample_rate {
        mono.samples.len()
    } else {
        let ratio = target_sample_rate as f64 / mono.sample_rate as f64;
        (mono.samples.len() as f64 * ratio).ceil() as usize
    };
    Ok(sample_count)
}

/// Match Python's approx_duration_from_text from utils.py
fn approx_duration_from_text(text: &str, max_duration: f64) -> f64 {
    const EN_DUR_PER_CHAR: f64 = 0.082;
    const ZH_DUR_PER_CHAR: f64 = 0.21;

    // Remove whitespace (matching Python's re.sub(r"\s+", "", text))
    let text: String = text.chars().filter(|c| !c.is_whitespace()).collect();

    let mut num_zh = 0usize;
    let mut num_en = 0usize;
    let mut num_other = 0usize;

    for c in text.chars() {
        if ('\u{4e00}'..='\u{9fff}').contains(&c) {
            num_zh += 1;
        } else if c.is_ascii_alphabetic() {
            num_en += 1;
        } else {
            num_other += 1;
        }
    }

    // Match Python: if num_zh > num_en, num_zh += num_other, else num_en += num_other
    if num_zh > num_en {
        num_zh += num_other;
    } else {
        num_en += num_other;
    }

    let duration = num_zh as f64 * ZH_DUR_PER_CHAR + num_en as f64 * EN_DUR_PER_CHAR;
    duration.min(max_duration)
}

fn validate_request_payload(request: &LongCatSynthesisRequest) -> Result<()> {
    ensure!(
        !request.text.trim().is_empty(),
        "LongCat synthesis text is empty"
    );
    ensure!(request.steps > 0, "LongCat diffusion steps must be > 0");
    ensure!(
        request.cfg_strength >= 0.0,
        "LongCat cfg/apg strength must be >= 0"
    );
    if request.prompt_audio.is_some() {
        ensure!(
            request
                .prompt_text
                .as_deref()
                .is_some_and(|text| !text.trim().is_empty()),
            "voice cloning requires both prompt_audio and prompt_text"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_validation_requires_prompt_pair() {
        let mut request = LongCatSynthesisRequest::new("hello");
        request.prompt_audio = Some(PathBuf::from("prompt.wav"));
        assert!(validate_request_payload(&request).is_err());
    }
}
