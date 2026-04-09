use crate::longcat_audiodit::{
    config::AudioDiTConfig,
    loader::WeightIndex,
    rope::apply_qwen_rope,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{
    conv1d, layer_norm, layer_norm_no_bias, linear_b, ops, Conv1d, Conv1dConfig,
    LayerNorm, Linear, Module, RmsNorm, VarBuilder,
};
use std::path::Path;

#[derive(Debug)]
pub struct AudioDiTTransformer {
    dim: usize,
    latent_dim: usize,
    long_skip: bool,
    repa_dit_layer: usize,
    adaln_type: AdaLnType,
    adaln_use_text_cond: bool,
    time_embed: TimestepEmbedding,
    input_embed: AudioDiTEmbedder,
    text_embed: AudioDiTEmbedder,
    rotary_embed: RotaryEmbedding,
    blocks: Vec<AudioDiTBlock>,
    norm_out: AdaLayerNormZeroFinal,
    proj_out: Linear,
    adaln_global_mlp: Option<AdaLnMlp>,
    text_conv_layer: Vec<ConvNeXtV2Block>,
    latent_embed: Option<AudioDiTEmbedder>,
    latent_cond_embedder: Option<AudioDiTEmbedder>,
}

#[derive(Debug, Clone)]
pub struct TransformerForwardInput<'a> {
    pub x: &'a Tensor,
    pub text: &'a Tensor,
    pub text_len: &'a Tensor,
    pub time: &'a Tensor,
    pub mask: Option<&'a Tensor>,
    pub cond_mask: Option<&'a Tensor>,
    pub latent_cond: Option<&'a Tensor>,
}

#[derive(Debug)]
pub struct TransformerForwardOutput {
    pub last_hidden_state: Tensor,
}

impl AudioDiTTransformer {
    pub fn load(
        config: &AudioDiTConfig,
        model_weights: impl AsRef<Path>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_weights.as_ref()], DType::F32, device)
                .context("failed to open LongCat safetensors for transformer")?
        };
        let vb = vb.pp("transformer");
        let dim = config.dit_dim;
        let latent_dim = config.latent_dim;
        let dim_head = dim / config.dit_heads;
        let blocks_vb = vb.pp("blocks");

        let mut blocks = Vec::with_capacity(config.dit_depth);
        for idx in 0..config.dit_depth {
            blocks.push(AudioDiTBlock::load(config, blocks_vb.pp(idx))?);
        }

        let text_conv_layer = if config.dit_text_conv {
            let mut layers = Vec::with_capacity(4);
            let text_conv_vb = vb.pp("text_conv_layer");
            for idx in 0..4 {
                layers.push(ConvNeXtV2Block::load(
                    dim,
                    dim * 2,
                    config.dit_bias,
                    config.dit_eps,
                    text_conv_vb.pp(idx),
                )?);
            }
            layers
        } else {
            Vec::new()
        };

        Ok(Self {
            dim,
            latent_dim,
            long_skip: config.dit_long_skip,
            repa_dit_layer: config.repa_dit_layer,
            adaln_type: AdaLnType::from_str(&config.dit_adaln_type),
            adaln_use_text_cond: config.dit_adaln_use_text_cond,
            time_embed: TimestepEmbedding::load(dim, vb.pp("time_embed"))?,
            input_embed: AudioDiTEmbedder::load(latent_dim, dim, vb.pp("input_embed"))?,
            text_embed: AudioDiTEmbedder::load(config.dit_text_dim, dim, vb.pp("text_embed"))?,
            rotary_embed: RotaryEmbedding::new(dim_head, 2048, 100000.0, dtype, device)?,
            blocks,
            norm_out: AdaLayerNormZeroFinal::load(dim, config.dit_eps, vb.pp("norm_out"))?,
            proj_out: linear_b(dim, latent_dim, config.dit_bias, vb.pp("proj_out"))?,
            adaln_global_mlp: matches!(
                AdaLnType::from_str(&config.dit_adaln_type),
                AdaLnType::Global
            )
            .then(|| AdaLnMlp::load(dim, dim * 6, true, vb.pp("adaln_global_mlp")))
            .transpose()?,
            text_conv_layer,
            latent_embed: config
                .dit_use_latent_condition
                .then(|| AudioDiTEmbedder::load(latent_dim, dim, vb.pp("latent_embed")))
                .transpose()?,
            latent_cond_embedder: config
                .dit_use_latent_condition
                .then(|| AudioDiTEmbedder::load(dim * 2, dim, vb.pp("latent_cond_embedder")))
                .transpose()?,
        })
    }

    pub fn forward(&self, input: TransformerForwardInput<'_>) -> Result<TransformerForwardOutput> {
        let batch = input.x.dim(0)?;
        let text_seq_len = input.text.dim(1)?;

        let mut t = input.time.clone();
        if t.rank() == 0 {
            t = t.reshape((1,))?.expand((batch,))?;
        }

        let t = self.time_embed.forward(&t)?;
        let mut text = self.text_embed.forward(input.text, input.cond_mask)?;
        if !self.text_conv_layer.is_empty() {
            for block in &self.text_conv_layer {
                text = block.forward(&text)?;
            }
            if let Some(cond_mask) = input.cond_mask {
                text = apply_sequence_mask(&text, cond_mask)?;
            }
        }

        let mut x = self.input_embed.forward(input.x, input.mask)?;
        if let (Some(latent_embed), Some(latent_cond_embedder), Some(latent_cond)) = (
            self.latent_embed.as_ref(),
            self.latent_cond_embedder.as_ref(),
            input.latent_cond,
        ) {
            let latent_cond = latent_embed.forward(latent_cond, input.mask)?;
            let fused = Tensor::cat(&[&x, &latent_cond], D::Minus1)?;
            x = latent_cond_embedder.forward(&fused, input.mask)?;
        }

        let x_skip = if self.long_skip {
            Some(x.clone())
        } else {
            None
        };
        let rope = self.rotary_embed.slice(x.dim(1)?)?;
        let cond_rope = self.rotary_embed.slice(text_seq_len)?;
        let norm_cond = match self.adaln_type {
            AdaLnType::Global if self.adaln_use_text_cond => {
                let text_mean = masked_mean(&text, input.text_len)?;
                Some((&t + &text_mean)?)
            }
            AdaLnType::Global => Some(t.clone()),
            AdaLnType::Local => None,
        };
        let adaln_global_out = match (self.adaln_global_mlp.as_ref(), norm_cond.as_ref()) {
            (Some(mlp), Some(norm_cond)) => Some(mlp.forward(norm_cond)?),
            _ => None,
        };

        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(
                &x,
                &t,
                &text,
                input.mask,
                input.cond_mask,
                Some(&rope),
                Some(&cond_rope),
                adaln_global_out.as_ref(),
            )?;
            // Match Python: apply long-skip after repa_dit_layer block (mid-forward)
            // Python: if return_ith_layer == i + 1: x = x + x_clone
            // With repa_dit_layer=8, this triggers after block index 7
            if self.long_skip && i + 1 == self.repa_dit_layer {
                if let Some(ref x_skip) = x_skip {
                    x = (&x + x_skip)?;
                }
            }
        }

        if let Some(x_skip) = x_skip.as_ref() {
            x = (&x + x_skip)?;
        }

        let norm_emb = norm_cond.as_ref().unwrap_or(&t);
        x = self.norm_out.forward(&x, norm_emb)?;
        let last_hidden_state = x.apply(&self.proj_out)?;
        Ok(TransformerForwardOutput { last_hidden_state })
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    pub fn hidden_dim(&self) -> usize {
        self.dim
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum AdaLnType {
    Local,
    Global,
}

impl AdaLnType {
    fn from_str(value: &str) -> Self {
        match value {
            "global" => Self::Global,
            _ => Self::Local,
        }
    }
}

#[derive(Debug)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dim: usize,
        max_seq_len: usize,
        base: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|idx| 1f32 / (base as f32).powf(idx as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(DType::F32)?;
        let positions = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = positions.matmul(&inv_freq)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        println!("Rs cos mean: {:?}", cos.mean_all()?.to_scalar::<f32>()?);
        println!("Rs sin mean: {:?}", sin.mean_all()?.to_scalar::<f32>()?);
        Ok(Self {
            cos,
            sin,
        })
    }

    fn slice(&self, seq_len: usize) -> Result<RotarySlice> {
        Ok(RotarySlice {
            cos: self.cos.narrow(0, 0, seq_len)?,
            sin: self.sin.narrow(0, 0, seq_len)?,
        })
    }
}

#[derive(Debug)]
struct RotarySlice {
    cos: Tensor,
    sin: Tensor,
}

#[derive(Debug)]
struct TimestepEmbedding {
    dim: usize,
    time_mlp_0: Linear,
    time_mlp_2: Linear,
}

impl TimestepEmbedding {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            dim,
            time_mlp_0: linear_b(256, dim, true, vb.pp("time_mlp").pp(0))?,
            time_mlp_2: linear_b(dim, dim, true, vb.pp("time_mlp").pp(2))?,
        })
    }

    fn forward(&self, timestep: &Tensor) -> Result<Tensor> {
        let sinus = sinus_position_embedding(timestep, 256)?;
        let hidden = sinus.apply(&self.time_mlp_0)?.silu()?;
        Ok(hidden.apply(&self.time_mlp_2)?)
    }
}

#[derive(Debug)]
struct AudioDiTEmbedder {
    proj_0: Linear,
    proj_2: Linear,
}

impl AudioDiTEmbedder {
    fn load(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj_0: linear_b(in_dim, out_dim, true, vb.pp("proj").pp(0))?,
            proj_2: linear_b(out_dim, out_dim, true, vb.pp("proj").pp(2))?,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let masked = match mask {
            Some(mask) => apply_sequence_mask(x, mask)?,
            None => x.clone(),
        };
        let out = masked.apply(&self.proj_0)?.silu()?.apply(&self.proj_2)?;
        match mask {
            Some(mask) => apply_sequence_mask(&out, mask),
            None => Ok(out),
        }
    }
}

#[derive(Debug)]
struct AdaLnMlp {
    linear: Linear,
}

impl AdaLnMlp {
    fn load(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: linear_b(in_dim, out_dim, bias, vb.pp("mlp").pp(1))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.silu()?.apply(&self.linear)?)
    }
}

#[derive(Debug)]
struct AdaLayerNormZeroFinal {
    linear: Linear,
    norm: LayerNorm,
}

impl AdaLayerNormZeroFinal {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = Tensor::ones((dim,), vb.dtype(), vb.device())?;
        Ok(Self {
            linear: linear_b(dim, dim * 2, true, vb.pp("linear"))?,
            norm: LayerNorm::new_no_bias(weight, eps),
        })
    }

    fn forward(&self, x: &Tensor, emb: &Tensor) -> Result<Tensor> {
        let emb = emb.silu()?.apply(&self.linear)?;
        let chunks = split_last_dim(&emb, 2)?;
        modulate(&self.norm.forward(x)?, &chunks[0], &chunks[1])
    }
}

#[derive(Debug)]
struct AudioDiTSelfAttention {
    heads: usize,
    head_dim: usize,
    inner_dim: usize,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    to_out: Linear,
}

impl AudioDiTSelfAttention {
    fn load(
        dim: usize,
        heads: usize,
        bias: bool,
        qk_norm: bool,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = dim;
        Ok(Self {
            heads,
            head_dim: dim / heads,
            inner_dim,
            to_q: linear_b(dim, inner_dim, bias, vb.pp("to_q"))?,
            to_k: linear_b(dim, inner_dim, bias, vb.pp("to_k"))?,
            to_v: linear_b(dim, inner_dim, bias, vb.pp("to_v"))?,
            q_norm: qk_norm
                .then(|| candle_nn::rms_norm(inner_dim, eps, vb.pp("q_norm")))
                .transpose()?,
            k_norm: qk_norm
                .then(|| candle_nn::rms_norm(inner_dim, eps, vb.pp("k_norm")))
                .transpose()?,
            to_out: linear_b(inner_dim, dim, bias, vb.pp("to_out").pp(0))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rope: Option<&RotarySlice>,
    ) -> Result<Tensor> {
        let batch = x.dim(0)?;
        let seq_len = x.dim(1)?;
        let mut q = x.apply(&self.to_q)?;
        let mut k = x.apply(&self.to_k)?;
        let v = x.apply(&self.to_v)?;
        if let Some(norm) = self.q_norm.as_ref() {
            q = norm.forward(&q)?;
        }
        if let Some(norm) = self.k_norm.as_ref() {
            k = norm.forward(&k)?;
        }
        let q = reshape_heads(&q, batch, seq_len, self.heads, self.head_dim)?;
        let k = reshape_heads(&k, batch, seq_len, self.heads, self.head_dim)?;
        let v = reshape_heads(&v, batch, seq_len, self.heads, self.head_dim)?;
        let (q, k) = match rope {
            Some(rope) => (
                apply_qwen_rope(&q.contiguous()?, &rope.cos, &rope.sin)?,
                apply_qwen_rope(&k.contiguous()?, &rope.cos, &rope.sin)?,
            ),
            None => (q, k),
        };
        let context = attention(&q, &k, &v, mask, mask, self.head_dim)?;
        Ok(context
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.inner_dim))?
            .apply(&self.to_out)?)
    }
}

#[derive(Debug)]
struct AudioDiTCrossAttention {
    heads: usize,
    head_dim: usize,
    inner_dim: usize,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    to_out: Linear,
}

impl AudioDiTCrossAttention {
    fn load(
        q_dim: usize,
        kv_dim: usize,
        heads: usize,
        bias: bool,
        qk_norm: bool,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = q_dim;
        Ok(Self {
            heads,
            head_dim: q_dim / heads,
            inner_dim,
            to_q: linear_b(q_dim, inner_dim, bias, vb.pp("to_q"))?,
            to_k: linear_b(kv_dim, inner_dim, bias, vb.pp("to_k"))?,
            to_v: linear_b(kv_dim, inner_dim, bias, vb.pp("to_v"))?,
            q_norm: qk_norm
                .then(|| candle_nn::rms_norm(inner_dim, eps, vb.pp("q_norm")))
                .transpose()?,
            k_norm: qk_norm
                .then(|| candle_nn::rms_norm(inner_dim, eps, vb.pp("k_norm")))
                .transpose()?,
            to_out: linear_b(inner_dim, q_dim, bias, vb.pp("to_out").pp(0))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cond: &Tensor,
        mask: Option<&Tensor>,
        cond_mask: Option<&Tensor>,
        rope: Option<&RotarySlice>,
        cond_rope: Option<&RotarySlice>,
    ) -> Result<Tensor> {
        let batch = x.dim(0)?;
        let seq_len = x.dim(1)?;
        let cond_len = cond.dim(1)?;
        let mut q = x.apply(&self.to_q)?;
        let mut k = cond.apply(&self.to_k)?;
        let v = cond.apply(&self.to_v)?;
        if let Some(norm) = self.q_norm.as_ref() {
            q = norm.forward(&q)?;
        }
        if let Some(norm) = self.k_norm.as_ref() {
            k = norm.forward(&k)?;
        }
        let q = reshape_heads(&q, batch, seq_len, self.heads, self.head_dim)?;
        let k = reshape_heads(&k, batch, cond_len, self.heads, self.head_dim)?;
        let v = reshape_heads(&v, batch, cond_len, self.heads, self.head_dim)?;
        let q = match rope {
            Some(rope) => apply_qwen_rope(&q.contiguous()?, &rope.cos, &rope.sin)?,
            None => q,
        };
        let k = match cond_rope {
            Some(rope) => apply_qwen_rope(&k.contiguous()?, &rope.cos, &rope.sin)?,
            None => k,
        };
        let context = attention(&q, &k, &v, mask, cond_mask, self.head_dim)?;
        Ok(context
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.inner_dim))?
            .apply(&self.to_out)?)
    }
}

#[derive(Debug)]
struct AudioDiTFeedForward {
    fc1: Linear,
    fc2: Linear,
}

impl AudioDiTFeedForward {
    fn load(dim: usize, mult: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear_b(dim, dim * mult, bias, vb.pp("ff").pp(0))?,
            fc2: linear_b(dim * mult, dim, bias, vb.pp("ff").pp(3))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.apply(&self.fc1)?.gelu()?.apply(&self.fc2)?)
    }
}

#[derive(Debug)]
struct AudioDiTBlock {
    adaln_type: AdaLnType,
    adaln_use_text_cond: bool,
    adaln_mlp: Option<AdaLnMlp>,
    adaln_scale_shift: Option<Tensor>,
    self_attn: AudioDiTSelfAttention,
    cross_attn: Option<AudioDiTCrossAttention>,
    cross_attn_norm: Option<LayerNorm>,
    cross_attn_norm_c: Option<LayerNorm>,
    ffn: AudioDiTFeedForward,
    eps: f64,
}

impl AudioDiTBlock {
    fn load(config: &AudioDiTConfig, vb: VarBuilder) -> Result<Self> {
        let dim = config.dit_dim;
        Ok(Self {
            adaln_type: AdaLnType::from_str(&config.dit_adaln_type),
            adaln_use_text_cond: config.dit_adaln_use_text_cond,
            adaln_mlp: if config.dit_adaln_type == "local" {
                Some(AdaLnMlp::load(dim, dim * 6, true, vb.pp("adaln_mlp"))?)
            } else {
                None
            },
            adaln_scale_shift: if config.dit_adaln_type == "global" {
                Some(vb.get(dim * 6, "adaln_scale_shift")?)
            } else {
                None
            },
            self_attn: AudioDiTSelfAttention::load(
                dim,
                config.dit_heads,
                config.dit_bias,
                config.dit_qk_norm,
                config.dit_eps,
                vb.pp("self_attn"),
            )?,
            cross_attn: config
                .dit_cross_attn
                .then(|| {
                    AudioDiTCrossAttention::load(
                        dim,
                        dim,
                        config.dit_heads,
                        config.dit_bias,
                        config.dit_qk_norm,
                        config.dit_eps,
                        vb.pp("cross_attn"),
                    )
                })
                .transpose()?,
            cross_attn_norm: config
                .dit_cross_attn
                .then(|| {
                    if config.dit_cross_attn_norm {
                        layer_norm(dim, config.dit_eps, vb.pp("cross_attn_norm"))
                    } else {
                        let weight = Tensor::ones((dim,), vb.dtype(), vb.device())?;
                        let bias = Tensor::zeros((dim,), vb.dtype(), vb.device())?;
                        Ok(LayerNorm::new(weight, bias, config.dit_eps))
                    }
                })
                .transpose()?,
            cross_attn_norm_c: config
                .dit_cross_attn
                .then(|| {
                    if config.dit_cross_attn_norm {
                        layer_norm(dim, config.dit_eps, vb.pp("cross_attn_norm_c"))
                    } else {
                        let weight = Tensor::ones((dim,), vb.dtype(), vb.device())?;
                        let bias = Tensor::zeros((dim,), vb.dtype(), vb.device())?;
                        Ok(LayerNorm::new(weight, bias, config.dit_eps))
                    }
                })
                .transpose()?,
            ffn: AudioDiTFeedForward::load(dim, config.dit_ff_mult, config.dit_bias, vb.pp("ffn"))?,
            eps: config.dit_eps,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        cond: &Tensor,
        mask: Option<&Tensor>,
        cond_mask: Option<&Tensor>,
        rope: Option<&RotarySlice>,
        cond_rope: Option<&RotarySlice>,
        adaln_global_out: Option<&Tensor>,
    ) -> Result<Tensor> {
        let adaln_out = match self.adaln_type {
            AdaLnType::Local if adaln_global_out.is_none() => {
                let norm_cond = if self.adaln_use_text_cond {
                    let cond_mean = cond.mean(1)?;
                    (t + cond_mean)?
                } else {
                    t.clone()
                };
                self.adaln_mlp
                    .as_ref()
                    .expect("local adaln requires mlp")
                    .forward(&norm_cond)?
            }
            AdaLnType::Global => {
                let scale_shift = self
                    .adaln_scale_shift
                    .as_ref()
                    .expect("global adaln requires scale shift")
                    .reshape((1, self.adaln_scale_shift.as_ref().unwrap().dim(0)?))?;
                (adaln_global_out.expect("global adaln requires global mlp output") + &scale_shift)?
            }
            AdaLnType::Local => unreachable!(),
        };
        let chunks = split_last_dim(&adaln_out, 6)?;
        let gate_sa = &chunks[0];
        let scale_sa = &chunks[1];
        let shift_sa = &chunks[2];
        let gate_ffn = &chunks[3];
        let scale_ffn = &chunks[4];
        let shift_ffn = &chunks[5];

        let norm = modulate(&layer_norm_without_affine(x, self.eps)?, scale_sa, shift_sa)?;
        let attn_output = self.self_attn.forward(&norm, mask, rope)?;
        let x = (x + &broadcast_time_feature(gate_sa)?.broadcast_mul(&attn_output)?)?;

        let x = match (
            self.cross_attn.as_ref(),
            self.cross_attn_norm.as_ref(),
            self.cross_attn_norm_c.as_ref(),
        ) {
            (Some(cross_attn), Some(cross_attn_norm), Some(cross_attn_norm_c)) => {
                let cross_out = cross_attn.forward(
                    &cross_attn_norm.forward(&x)?,
                    &cross_attn_norm_c.forward(cond)?,
                    mask,
                    cond_mask,
                    rope,
                    cond_rope,
                )?;
                (&x + &cross_out)?
            }
            _ => x,
        };

        let norm = modulate(
            &layer_norm_without_affine(&x, self.eps)?,
            scale_ffn,
            shift_ffn,
        )?;
        let ff_output = self.ffn.forward(&norm)?;
        Ok((x + &broadcast_time_feature(gate_ffn)?.broadcast_mul(&ff_output)?)?)
    }
}

#[derive(Debug)]
struct Grn {
    gamma: Tensor,
    beta: Tensor,
}

impl Grn {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gamma: vb.get((1, 1, dim), "gamma")?,
            beta: vb.get((1, 1, dim), "beta")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gx = x.sqr()?.sum_keepdim(1)?.sqrt()?;
        let nx = gx.broadcast_div(&(gx.mean_keepdim(D::Minus1)? + 1e-6)?)?;
        let scaled = x.broadcast_mul(&nx)?;
        let modulated = self
            .gamma
            .broadcast_mul(&scaled)?
            .broadcast_add(&self.beta)?;
        Ok((modulated + x)?)
    }
}

#[derive(Debug)]
struct ConvNeXtV2Block {
    dwconv: Conv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    grn: Grn,
    pwconv2: Linear,
}

impl ConvNeXtV2Block {
    fn load(
        dim: usize,
        intermediate_dim: usize,
        bias: bool,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: dim,
            cudnn_fwd_algo: None,
        };
        Ok(Self {
            dwconv: conv1d(dim, dim, 7, cfg, vb.pp("dwconv"))?,
            norm: layer_norm(dim, eps, vb.pp("norm"))?,
            pwconv1: linear_b(dim, intermediate_dim, bias, vb.pp("pwconv1"))?,
            grn: Grn::load(intermediate_dim, vb.pp("grn"))?,
            pwconv2: linear_b(intermediate_dim, dim, bias, vb.pp("pwconv2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let hidden = x.transpose(1, 2)?.apply(&self.dwconv)?.transpose(1, 2)?;
        let hidden = hidden.apply(&self.norm)?.apply(&self.pwconv1)?.silu()?;
        let hidden = self.grn.forward(&hidden)?.apply(&self.pwconv2)?;
        Ok((residual + hidden)?)
    }
}

fn sinus_position_embedding(x: &Tensor, dim: usize) -> Result<Tensor> {
    let half_dim = dim / 2;
    let device = x.device();
    let input_dtype = x.dtype();
    let emb: Vec<f32> = (0..half_dim)
        .map(|idx| (-(10000f32).ln() * idx as f32 / (half_dim.saturating_sub(1) as f32)).exp())
        .collect();
    let emb = Tensor::from_vec(emb, (1, half_dim), device)?.to_dtype(input_dtype)?;
    let x = x.reshape((x.dim(0)?, 1))?;
    let thousand = Tensor::new(&[1000.0f32], device)?.to_dtype(input_dtype)?;
    let emb = x.broadcast_mul(&emb)?.broadcast_mul(&thousand)?.to_dtype(input_dtype)?;
    let result = Tensor::cat(&[&emb.sin()?, &emb.cos()?], D::Minus1)?;
    Ok(result.to_dtype(input_dtype)?)
}

fn reshape_heads(
    x: &Tensor,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    Ok(x.reshape((batch, seq, heads, head_dim))?.transpose(1, 2)?)
}

fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_mask: Option<&Tensor>,
    kv_mask: Option<&Tensor>,
    head_dim: usize,
) -> Result<Tensor> {
    let scale = 1.0 / (head_dim as f64).sqrt();
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let scores = (q.matmul(&k_t)? * scale)?;
    let mut scores = if let Some(mask) = build_attention_mask(q_mask, kv_mask, q.dtype(), q.device())? {
        scores.broadcast_add(&mask)?
    } else {
        scores
    };
    let softmax_out = ops::softmax_last_dim(&scores)?;
    Ok(softmax_out.matmul(&v.contiguous()?)?)
}

fn build_attention_mask(
    q_mask: Option<&Tensor>,
    kv_mask: Option<&Tensor>,
    dtype: DType,
    _device: &Device,
) -> Result<Option<Tensor>> {
    let (q_mask, kv_mask) = match (q_mask, kv_mask) {
        (Some(q_mask), Some(kv_mask)) => (q_mask, kv_mask),
        _ => return Ok(None),
    };
    let bsz = q_mask.dim(0)?;
    let q_len = q_mask.dim(1)?;
    let kv_len = kv_mask.dim(1)?;
    let q_mask_f = q_mask.to_dtype(dtype)?.reshape((bsz, 1, q_len, 1))?;
    let kv_mask_f = kv_mask.to_dtype(dtype)?.reshape((bsz, 1, 1, kv_len))?;
    let keep = q_mask_f.broadcast_mul(&kv_mask_f)?;
    let is_zero = keep.eq(&Tensor::zeros(keep.shape(), keep.dtype(), keep.device())?)?;
    let neg_inf = Tensor::full(f32::NEG_INFINITY, keep.shape(), keep.device())?.to_dtype(dtype)?;
    let zeros = Tensor::zeros(keep.shape(), keep.dtype(), keep.device())?;
    let mask = is_zero.where_cond(&neg_inf, &zeros)?;
    Ok(Some(mask))
}

fn apply_sequence_mask(x: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let mask = mask.to_dtype(x.dtype())?.unsqueeze(D::Minus1)?;
    Ok(x.broadcast_mul(&mask)?)
}

fn layer_norm_without_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let hidden = x.dim(D::Minus1)?;
    let weight = Tensor::ones((hidden,), x.dtype(), x.device())?;
    let bias = Tensor::zeros((hidden,), x.dtype(), x.device())?;
    Ok(ops::layer_norm(x, &weight, &bias, eps as f32)?)
}

fn broadcast_time_feature(x: &Tensor) -> Result<Tensor> {
    if x.rank() == 2 {
        Ok(x.unsqueeze(1)?)
    } else {
        Ok(x.clone())
    }
}

fn modulate(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let scale = broadcast_time_feature(scale)?;
    let shift = broadcast_time_feature(shift)?;
    let scaled = x.broadcast_mul(&(scale + 1.0)?)?;
    Ok(scaled.broadcast_add(&shift)?)
}

fn masked_mean(x: &Tensor, lengths: &Tensor) -> Result<Tensor> {
    let sum = x.sum(1)?;
    let denom = lengths
        .to_dtype(sum.dtype())?
        .reshape((lengths.dim(0)?, 1))?;
    Ok(sum.broadcast_div(&denom)?)
}

fn split_last_dim(x: &Tensor, chunks: usize) -> Result<Vec<Tensor>> {
    let hidden = x.dim(D::Minus1)?;
    let chunk = hidden / chunks;
    (0..chunks)
        .map(|idx| Ok(x.narrow(D::Minus1, idx * chunk, chunk)?))
        .collect()
}

pub fn lens_to_mask(lengths: &Tensor) -> Result<Tensor> {
    let bsz = lengths.dim(0)?;
    let max_len = lengths.max(0)?.to_scalar::<u32>()? as usize;
    let device = lengths.device();
    let seq = Tensor::arange(0u32, max_len as u32, device)?
        .reshape((1, max_len))?
        .expand((bsz, max_len))?;
    let lens = lengths
        .to_dtype(DType::U32)?
        .reshape((bsz, 1))?
        .expand((bsz, max_len))?;
    Ok(seq.lt(&lens)?)
}
