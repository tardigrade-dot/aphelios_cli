use crate::longcat_audiodit::{
    config::{AudioDiTConfig, AudioDiTVaeConfig},
    loader::WeightIndex,
    rng::LongCatRng,
};
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};

// ---------------------------------------------------------------------------
// Weight-norm helpers
// ---------------------------------------------------------------------------

fn reparam_wn(weight_g: &Tensor, weight_v: &Tensor) -> Result<Tensor> {
    let dims: Vec<usize> = (1..weight_v.rank()).collect();
    let norm = weight_v.sqr()?.sum_keepdim(dims.as_slice())?.sqrt()?;
    // Add epsilon to avoid division by zero - match dtype
    let norm_safe =
        norm.add(&Tensor::full(1e-8f32, norm.shape(), norm.device())?.to_dtype(norm.dtype())?)?;
    Ok(weight_g.broadcast_mul(&weight_v.broadcast_div(&norm_safe)?)?)
}

fn load_wn_conv1d(
    vb: &VarBuilder,
    out_ch: usize,
    in_ch: usize,
    k: usize,
) -> Result<(Tensor, Option<Tensor>)> {
    let wg = vb.get((out_ch, 1, 1), "weight_g")?;
    let wv = vb.get((out_ch, in_ch, k), "weight_v")?;
    let bias = vb.get(out_ch, "bias").ok();
    Ok((reparam_wn(&wg, &wv)?, bias))
}

fn load_wn_conv_transpose1d(
    vb: &VarBuilder,
    out_ch: usize,
    in_ch: usize,
    k: usize,
) -> Result<(Tensor, Option<Tensor>)> {
    // Python PyTorch ConvTranspose1d weight_norm stores:
    //   weight_g: [in_ch, 1, 1]  -- grouped by input channel
    //   weight_v: [in_ch, out_ch, k]
    // Candle ConvTranspose1d expects weight: [in_ch, out_ch, k] -- same format!
    // So we can load directly without transpose.
    let wg = vb.get((in_ch, 1, 1), "weight_g")?;
    let wv = vb.get((in_ch, out_ch, k), "weight_v")?;
    let bias = vb.get(out_ch, "bias").ok();
    let weight = reparam_wn(&wg, &wv)?;
    Ok((weight, bias))
}

// ---------------------------------------------------------------------------
// SnakeBeta: f(x) = x + (1/beta) * sin(alpha * x)^2
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Snake {
    alpha: Tensor,
    beta: Tensor,
    alpha_logscale: bool,
}

impl Snake {
    fn load(ch: usize, vb: VarBuilder) -> Result<Self> {
        // Check if alpha_logscale is needed (default True in Python)
        Ok(Self {
            alpha: vb.get(ch, "alpha")?,
            beta: vb.get(ch, "beta")?,
            alpha_logscale: true,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut alpha = self.alpha.reshape((1, self.alpha.dim(0)?, 1))?;
        let mut beta = self.beta.reshape((1, self.beta.dim(0)?, 1))?;
        // Apply exp if logscale (matching Python AudioDiTSnakeBeta)
        if self.alpha_logscale {
            alpha = alpha.exp()?;
            beta = beta.exp()?;
        }
        Ok(x.add(
            &x.broadcast_mul(&alpha)?
                .sin()?
                .sqr()?
                .broadcast_mul(&beta.recip()?)?,
        )?)
    }
}

// ---------------------------------------------------------------------------
// Manual repeat_interleave (Candle doesn't have it)
// ---------------------------------------------------------------------------

fn repeat_interleave(x: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    // For 3D [B, C, W] repeating along dim=1 (C):
    if dim == 1 && x.rank() == 3 {
        let (b, c, w) = (x.dim(0)?, x.dim(1)?, x.dim(2)?);
        let x = x.reshape((b, c, 1, w))?;
        let ones = Tensor::ones((1, 1, repeats, 1), x.dtype(), x.device())?;
        let x_tiled = x.broadcast_mul(&ones)?;
        Ok(x_tiled.reshape((b, c * repeats, w))?)
    } else {
        anyhow::bail!("repeat_interleave only supports dim=1 for 3D tensors")
    }
}

// ---------------------------------------------------------------------------
// Residual unit
// ---------------------------------------------------------------------------

struct ResUnit {
    act1: Snake,
    conv1: Conv1d,
    act2: Snake,
    conv2: Conv1d,
}

impl ResUnit {
    fn load(ch: usize, dil: usize, vb: VarBuilder) -> Result<Self> {
        let pad = dil * 3;
        let (w1, b1) = load_wn_conv1d(&vb.pp("layers").pp(1), ch, ch, 7)?;
        let (w2, b2) = load_wn_conv1d(&vb.pp("layers").pp(3), ch, ch, 1)?;
        Ok(Self {
            act1: Snake::load(ch, vb.pp("layers").pp(0))?,
            conv1: Conv1d::new(
                w1,
                b1,
                Conv1dConfig {
                    padding: pad,
                    stride: 1,
                    dilation: dil,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
            ),
            act2: Snake::load(ch, vb.pp("layers").pp(2))?,
            conv2: Conv1d::new(
                w2,
                b2,
                Conv1dConfig {
                    padding: 0,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
            ),
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(&self.act1.forward(x)?)?;
        Ok(x.add(&self.conv2.forward(&self.act2.forward(&h)?)?)?)
    }
}

// ---------------------------------------------------------------------------
// Encoder block
// ---------------------------------------------------------------------------

struct DownShort {
    factor: usize,
    out_ch: usize,
    grp: usize,
}

impl DownShort {
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, mut w) = x.dims3()?;
        let mut x = x.clone();
        if self.factor > 1 {
            if w % self.factor != 0 {
                let pad = self.factor - (w % self.factor);
                x = x.pad_with_zeros(D::Minus1, 0, pad)?;
                w += pad;
            }
            let w_div = w / self.factor;
            let x = x
                .reshape((b, c, w_div, self.factor))?
                .transpose(2, 3)?
                .reshape((b, c * self.factor, w_div))?
                .reshape((b, self.out_ch, self.grp, w_div))?;
            Ok(x.mean(D::Minus2)?)
        } else {
            // factor=1: channel-only pooling (used in Encoder shortcut)
            let grp = c / self.out_ch;
            let x = x.reshape((b, self.out_ch, grp, w))?;
            Ok(x.mean(D::Minus2)?)
        }
    }
}

struct EncBlock {
    res: Vec<ResUnit>,
    down_act: Snake,
    down_conv: Conv1d,
    shortcut: Option<DownShort>,
}

impl EncBlock {
    fn load(in_ch: usize, out_ch: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let mut res = Vec::new();
        for (i, d) in [1, 3, 9].iter().enumerate() {
            res.push(ResUnit::load(in_ch, *d, vb.pp("layers").pp(i))?);
        }
        let down_act = Snake::load(in_ch, vb.pp("layers").pp(3))?;
        let (dw, db) = load_wn_conv1d(&vb.pp("layers").pp(4), out_ch, in_ch, 2 * stride)?;
        let down_conv = Conv1d::new(
            dw,
            db,
            Conv1dConfig {
                padding: stride.div_ceil(2),
                stride,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        );
        let shortcut = Some(DownShort {
            factor: stride,
            out_ch,
            grp: in_ch * stride / out_ch,
        });
        Ok(Self {
            res,
            down_act,
            down_conv,
            shortcut,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for u in &self.res {
            h = u.forward(&h)?;
        }
        h = self.down_conv.forward(&self.down_act.forward(&h)?)?;
        if let Some(s) = &self.shortcut {
            h = h.add(&s.fwd(x)?)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// Decoder block
// ---------------------------------------------------------------------------

struct UpShort {
    factor: usize,
    repeats: usize,
}

impl UpShort {
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c_in, w) = x.dims3()?;
        // repeat_interleave matches Python: x.repeat_interleave(self.repeats, dim=1)
        let x = repeat_interleave(x, self.repeats, 1)?;
        // After repeat_interleave: [b, c_in * repeats, w]
        let c_total = c_in * self.repeats;
        let c_out = c_total / self.factor;
        // Pixel shuffle: [b, c_total, w] -> [b, c_out, factor, w] -> [b, c_out, w, factor] -> [b, c_out, w*factor]
        // Matches Python: x.view(b, c, factor, w).permute(0, 1, 3, 2).contiguous().view(b, c, w * factor)
        Ok(x.reshape((b, c_out, self.factor, w))?
            .transpose(2, 3)?
            .reshape((b, c_out, w * self.factor))?)
    }
}

struct DecBlock {
    up_act: Snake,
    up_conv: ConvTranspose1d,
    res: Vec<ResUnit>,
    shortcut: Option<UpShort>,
}

impl DecBlock {
    fn load(in_ch: usize, out_ch: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let up_act = Snake::load(in_ch, vb.pp("layers").pp(0))?;
        let (uw, ub) = load_wn_conv_transpose1d(&vb.pp("layers").pp(1), out_ch, in_ch, 2 * stride)?;
        let up_conv = ConvTranspose1d::new(
            uw,
            ub,
            ConvTranspose1dConfig {
                padding: stride.div_ceil(2),
                stride,
                dilation: 1,
                groups: 1,
                output_padding: 0,
            },
        );
        let mut res = Vec::new();
        for (i, d) in [1, 3, 9].iter().enumerate() {
            res.push(ResUnit::load(out_ch, *d, vb.pp("layers").pp(i + 2))?);
        }
        let shortcut = Some(UpShort {
            factor: stride,
            repeats: out_ch * stride / in_ch,
        });
        Ok(Self {
            up_act,
            up_conv,
            res,
            shortcut,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.up_conv.forward(&self.up_act.forward(x)?)?;
        for u in &self.res {
            h = u.forward(&h)?;
        }
        if let Some(s) = &self.shortcut {
            h = h.add(&s.fwd(x)?)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

struct Encoder {
    init: Conv1d,
    blocks: Vec<EncBlock>,
    final_conv: Conv1d,
    has_shortcut: bool,
    sc_in: usize,
    sc_out: usize,
}

impl Encoder {
    fn load(cfg: &AudioDiTVaeConfig, vb: VarBuilder) -> Result<Self> {
        let cm = [&[1], &cfg.c_mults[..]].concat();
        let ch = cfg.channels;
        let (iw, ib) = load_wn_conv1d(&vb.pp("layers").pp(0), cm[0] * ch, cfg.in_channels, 7)?;
        let init = Conv1d::new(
            iw,
            ib,
            Conv1dConfig {
                padding: 3,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        );
        let mut blocks = Vec::new();
        for i in 0..cm.len() - 1 {
            blocks.push(EncBlock::load(
                cm[i] * ch,
                cm[i + 1] * ch,
                cfg.strides[i],
                vb.pp("layers").pp(i + 1),
            )?);
        }
        let lc = cm[cm.len() - 1] * ch;
        let (fw, fb) =
            load_wn_conv1d(&vb.pp("layers").pp(cm.len()), cfg.encoder_latent_dim, lc, 3)?;
        let final_conv = Conv1d::new(
            fw,
            fb,
            Conv1dConfig {
                padding: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        );
        Ok(Self {
            init,
            blocks,
            final_conv,
            has_shortcut: cfg.out_shortcut == "averaging",
            sc_in: lc,
            sc_out: cfg.encoder_latent_dim,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.init.forward(x)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        if !self.has_shortcut {
            return Ok(self.final_conv.forward(&h)?);
        }
        let before = h.clone();
        h = self.final_conv.forward(&h)?;
        let sc = DownShort {
            factor: 1,
            out_ch: self.sc_out,
            grp: self.sc_in,
        };
        Ok(h.add(&sc.fwd(&before)?)?)
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

struct Decoder {
    init: Conv1d,
    blocks: Vec<DecBlock>,
    final_act: Snake,
    final_conv: Conv1d,
    has_shortcut: bool,
    sc_repeats: usize,
}

impl Decoder {
    fn load(cfg: &AudioDiTVaeConfig, vb: VarBuilder) -> Result<Self> {
        let cm = [&[1], &cfg.c_mults[..]].concat();
        let ch = cfg.channels;
        let nb = cm.len() - 1;
        let lc = cm[nb] * ch;
        let (iw, ib) = load_wn_conv1d(&vb.pp("layers").pp(0), lc, cfg.latent_dim, 7)?;
        let init = Conv1d::new(
            iw,
            ib,
            Conv1dConfig {
                padding: 3,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        );
        let mut blocks = Vec::new();
        for i in 0..nb {
            let ii = nb - i;
            blocks.push(DecBlock::load(
                cm[ii] * ch,
                cm[ii - 1] * ch,
                cfg.strides[ii - 1],
                vb.pp("layers").pp(i + 1),
            )?);
        }
        let fa = Snake::load(cm[0] * ch, vb.pp("layers").pp(nb + 1))?;
        let (fw, fb) = load_wn_conv1d(&vb.pp("layers").pp(nb + 2), cfg.in_channels, cm[0] * ch, 7)?;
        let final_conv = Conv1d::new(
            fw,
            fb,
            Conv1dConfig {
                padding: 3,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        );
        Ok(Self {
            init,
            blocks,
            final_act: fa,
            final_conv,
            has_shortcut: cfg.in_shortcut == "duplicating",
            sc_repeats: lc / cfg.latent_dim,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if !self.has_shortcut {
            let mut h = self.init.forward(x)?;
            for b in self.blocks.iter() {
                h = b.forward(&h)?;
            }
            return Ok(self.final_conv.forward(&self.final_act.forward(&h)?)?);
        }
        let sc = UpShort {
            factor: 1,
            repeats: self.sc_repeats,
        };
        let sc_out = sc.fwd(x)?;
        let mut h = self.init.forward(x)?;
        h = h.add(&sc_out)?;
        for (i, b) in self.blocks.iter().enumerate() {
            h = b.forward(&h)?;
        }
        let final_act_out = self.final_act.forward(&h)?;
        Ok(self.final_conv.forward(&final_act_out)?)
    }
}

// ---------------------------------------------------------------------------
// Full VAE
// ---------------------------------------------------------------------------

pub struct AudioVae {
    enc: Encoder,
    dec: Decoder,
    scale: f64,
    dtype: DType,
}

impl AudioVae {
    pub fn load(
        cfg: &AudioDiTVaeConfig,
        wp: impl AsRef<std::path::Path>,
        dev: &Device,
    ) -> Result<Self> {
        // Use F32 for VAE to avoid numerical issues with F16/BF16
        let dtype = DType::F32;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[wp.as_ref()], DType::F32, dev)
                .context("failed to open LongCat safetensors for VAE")?
        };
        let enc = Encoder::load(cfg, vb.pp("vae.encoder")).context("failed to load VAE encoder")?;
        let dec = Decoder::load(cfg, vb.pp("vae.decoder")).context("failed to load VAE decoder")?;
        Ok(Self {
            enc,
            dec,
            scale: cfg.scale,
            dtype,
        })
    }

    pub fn encode(&self, audio: &Tensor, rng: &mut LongCatRng) -> Result<Tensor> {
        let inp = audio.to_dtype(self.dtype)?;
        let lat = self.enc.forward(&inp)?;
        let chunks = lat.chunk(2, D::Minus2)?;
        let mean = &chunks[0];
        let scale_p = &chunks[1];
        // Numerically stable softplus: softplus(x) = ln(1 + exp(x))
        // For large x: softplus(x) ≈ x, for small x: softplus(x) ≈ ln(2) + x/2
        // Use: softplus(x) = max(x, 0) + ln(1 + exp(-|x|)) to avoid overflow
        let abs_x = scale_p.abs()?;
        let neg_abs_x = abs_x.neg()?;
        let stable_softplus = scale_p.relu()?.add(&neg_abs_x.exp()?.add(&Tensor::ones(scale_p.shape(), scale_p.dtype(), scale_p.device())?)?.log()?)?;
        let stdev = stable_softplus.add(&Tensor::full(1e-4f32, scale_p.shape(), scale_p.device())?.to_dtype(scale_p.dtype())?)?;
        let noise = rng.standard_normal_tensor(mean.shape().dims(), stdev.dtype(), mean.device())?;
        let lat = noise.broadcast_mul(&stdev)?.add(mean)?;
        let scale_tensor =
            Tensor::full(self.scale as f32, lat.shape(), lat.device())?.to_dtype(lat.dtype())?;
        Ok(lat.broadcast_div(&scale_tensor)?)
    }

    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let scale_tensor = Tensor::full(self.scale as f32, latents.shape(), latents.device())?
            .to_dtype(latents.dtype())?;
        let z = latents.broadcast_mul(&scale_tensor)?;
        let z = z.to_dtype(self.dtype)?;
        let dec = self.dec.forward(&z)?;
        Ok(dec.to_dtype(DType::F32)?)
    }
}

impl std::fmt::Debug for AudioVae {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioVae")
            .field("scale", &self.scale)
            .field("dtype", &self.dtype)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct AudioVaeSpec {
    pub latent_dim: usize,
    pub encoder_latent_dim: usize,
    pub sample_rate: u32,
    pub downsampling_ratio: usize,
    pub uses_snake: bool,
}

impl AudioVaeSpec {
    pub fn from_index(_cfg: &AudioDiTConfig, index: &WeightIndex) -> Result<Self> {
        index.require_prefix("vae.")?;
        ensure!(
            index.prefix_count("vae.encoder.") > 0,
            "VAE missing encoder weights"
        );
        ensure!(
            index.prefix_count("vae.decoder.") > 0,
            "VAE missing decoder weights"
        );
        Ok(Self {
            latent_dim: 64,
            encoder_latent_dim: 128,
            sample_rate: 24000,
            downsampling_ratio: 2048,
            uses_snake: true,
        })
    }
}
