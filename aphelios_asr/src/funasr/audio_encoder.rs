use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

// FSMN Block structure
#[derive(Debug)]
pub struct FsmnBlock {
    weight: Tensor,
    kernel_size: usize,
    sanm_shfit: usize,
    d_model: usize,
}

impl FsmnBlock {
    pub fn load(
        vb: VarBuilder,
        d_model: usize,
        kernel_size: usize,
        sanm_shfit: usize,
    ) -> Result<Self> {
        let weight = vb.get((d_model, 1, kernel_size), "weight")?;
        Ok(Self {
            weight,
            kernel_size,
            sanm_shfit,
            d_model,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x shape: [batch_size, seq_len, d_model]
        let (b, t, d) = x.dims3()?;
        // Implement 1D convolution or padding based on FSMN logic
        // For simplicity, a standard FSMN applies the depthwise FSMN filters
        // For FunASR SANM, the FSMN is applied over the time dimension
        // x transpose: [batch_size, d_model, seq_len]
        let x_t = x.transpose(1, 2)?.contiguous()?;

        // FSMN is simply conv1d with FSMN weights
        // we use conv1d with self.weight
        let pad_left = (self.kernel_size - 1) / 2 + self.sanm_shfit;
        let pad_right = self.kernel_size - 1 - pad_left;

        // Padded x
        let x_pad = x_t.pad_with_zeros(2, pad_left, pad_right)?;

        let out = x_pad.conv1d(&self.weight, 0, 1, 1, self.d_model)?;
        let out = out.transpose(1, 2)?.contiguous()?;
        // x += inputs handled outside
        Ok(out)
    }
}

// MultiHeadedAttentionSANM
#[derive(Debug)]
pub struct MultiHeadedAttentionSANM {
    linear_q_k_v: Linear,
    linear_out: Linear,
    fsmn_block: FsmnBlock,
    d_k: usize,
    h: usize,
    kernel_size: usize,
    sanm_shfit: usize,
}

impl MultiHeadedAttentionSANM {
    pub fn load(
        vb: VarBuilder,
        attention_heads: usize,
        input_size: usize,
        output_size: usize,
        kernel_size: usize,
        sanm_shfit: usize,
    ) -> Result<Self> {
        let linear_q_k_v = candle_nn::linear(input_size, 3 * output_size, vb.pp("linear_q_k_v"))?;
        let linear_out = candle_nn::linear(output_size, output_size, vb.pp("linear_out"))?;
        let fsmn_block =
            FsmnBlock::load(vb.pp("fsmn_block"), output_size, kernel_size, sanm_shfit)?;

        Ok(Self {
            linear_q_k_v,
            linear_out,
            fsmn_block,
            d_k: output_size / attention_heads,
            h: attention_heads,
            kernel_size,
            sanm_shfit,
        })
    }

    // Simplification for the forward pass
    // without cache/masking for now (batch processing)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let q_k_v = self.linear_q_k_v.forward(x)?;
        // Split q_k_v
        let qkv = q_k_v.chunk(3, 2)?;
        let (q, k, v) = (&qkv[0], &qkv[1], &qkv[2]);

        let q_h = q.reshape(vec![b, t, self.h, self.d_k])?.transpose(1, 2)?;
        let k_h = k.reshape(vec![b, t, self.h, self.d_k])?.transpose(1, 2)?;
        let v_h = v.reshape(vec![b, t, self.h, self.d_k])?.transpose(1, 2)?;

        let fsmn_memory = self.fsmn_block.forward(v)?;
        let fsmn_memory = (fsmn_memory + v)?;

        let q_h = (q_h * (1.0 / (self.d_k as f64).sqrt()))?;
        let scores = q_h
            .contiguous()?
            .matmul(&k_h.transpose(2, 3)?.contiguous()?)?;

        let attn = candle_nn::ops::softmax(&scores, 3)?;
        let out = attn.matmul(&v_h.contiguous()?)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.h * self.d_k))?;
        let out = self.linear_out.forward(&out)?;

        Ok((out + fsmn_memory)?)
    }
}

// PositionwiseFeedForward
#[derive(Debug)]
pub struct PositionwiseFeedForward {
    w_1: Linear,
    w_2: Linear,
}

impl PositionwiseFeedForward {
    pub fn load(vb: VarBuilder, input_size: usize, hidden_size: usize) -> Result<Self> {
        let w_1 = candle_nn::linear(input_size, hidden_size, vb.pp("w_1"))?;
        let w_2 = candle_nn::linear(hidden_size, input_size, vb.pp("w_2"))?;
        Ok(Self { w_1, w_2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.w_1.forward(x)?;
        let x = x.relu()?;
        self.w_2.forward(&x)
    }
}

// EncoderLayerSANM
#[derive(Debug)]
pub struct EncoderLayerSANM {
    self_attn: MultiHeadedAttentionSANM,
    feed_forward: PositionwiseFeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    concat_linear: Option<Linear>,
    in_size: usize,
    output_size: usize,
}

impl EncoderLayerSANM {
    pub fn load(
        vb: VarBuilder,
        attention_heads: usize,
        input_size: usize,
        output_size: usize,
        linear_units: usize,
        kernel_size: usize,
        sanm_shfit: usize,
        concat_after: bool,
    ) -> Result<Self> {
        let self_attn = MultiHeadedAttentionSANM::load(
            vb.pp("self_attn"),
            attention_heads,
            input_size,
            output_size,
            kernel_size,
            sanm_shfit,
        )?;
        let feed_forward =
            PositionwiseFeedForward::load(vb.pp("feed_forward"), output_size, linear_units)?;
        let norm1 = candle_nn::layer_norm(input_size, 1e-5, vb.pp("norm1"))?;
        let norm2 = candle_nn::layer_norm(output_size, 1e-5, vb.pp("norm2"))?;

        let concat_linear = if concat_after {
            Some(candle_nn::linear(
                output_size + output_size,
                output_size,
                vb.pp("concat_linear"),
            )?)
        } else {
            None
        };

        Ok(Self {
            self_attn,
            feed_forward,
            norm1,
            norm2,
            concat_linear,
            in_size: input_size,
            output_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let nx = self.norm1.forward(x)?;

        let mut out = self.self_attn.forward(&nx)?;

        if let Some(cl) = &self.concat_linear {
            let cat = Tensor::cat(&[&nx, &out], 2)?;
            out = cl.forward(&cat)?;
        }

        // the python code actually does `x = residual + self.self_attn(...)` if not concat
        let x = if self.in_size == self.output_size {
            (residual + out)?
        } else {
            out
        };

        let residual = x.clone();
        let nx = self.norm2.forward(&x)?;
        let out = self.feed_forward.forward(&nx)?;
        let x = (residual + out)?;

        Ok(x)
    }
}

#[derive(Debug)]
pub struct SinusoidalPositionEncoder {
    d_model: usize,
}

impl SinusoidalPositionEncoder {
    pub fn new(d_model: usize) -> Self {
        Self { d_model }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, timesteps, input_dim) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();

        // positions: [1, timesteps] containing 1..=timesteps
        let positions =
            Tensor::arange(1f32, (timesteps + 1) as f32, device)?.reshape((1, timesteps))?;

        // log_timescale_increment = log(10000) / (depth / 2 - 1)
        let depth = input_dim as f32;
        let log_timescale_increment = (10000f32.ln()) / (depth / 2.0 - 1.0);

        // inv_timescales = exp(arange(depth / 2) * -log_timescale_increment)
        let inv_timescales = Tensor::arange(0f32, depth / 2.0, device)?
            .affine(-log_timescale_increment as f64, 0.0)?
            .exp()?;

        // inv_timescales: [1, depth/2]
        let inv_timescales = inv_timescales.reshape((1, (depth / 2.0) as usize))?;

        // scaled_time: [1, timesteps, depth/2]
        // positions: [1, timesteps, 1], inv_timescales: [1, 1, depth/2]
        let scaled_time = positions
            .unsqueeze(2)?
            .broadcast_mul(&inv_timescales.unsqueeze(1)?)?;

        // encoding: cat([sin, cos], dim=2)
        let sin = scaled_time.sin()?;
        let cos = scaled_time.cos()?;
        let pe = Tensor::cat(&[sin, cos], 2)?;

        x.add(&pe.to_dtype(dtype)?)
    }
}

pub struct SenseVoiceEncoderSmall {
    encoders0: Vec<EncoderLayerSANM>,
    encoders: Vec<EncoderLayerSANM>,
    tp_encoders: Vec<EncoderLayerSANM>,
    after_norm: LayerNorm,
    tp_norm: LayerNorm,
    output_size: usize,
    pos_enc: SinusoidalPositionEncoder,
}

impl SenseVoiceEncoderSmall {
    pub fn load(
        vb: VarBuilder,
        input_size: usize,
        output_size: usize,
        attention_heads: usize,
        linear_units: usize,
        num_blocks: usize,
        tp_blocks: usize,
        kernel_size: usize,
        sanm_shfit: usize,
    ) -> Result<Self> {
        let mut encoders0 = Vec::new();
        encoders0.push(EncoderLayerSANM::load(
            vb.pp("encoders0.0"),
            attention_heads,
            input_size,
            output_size,
            linear_units,
            kernel_size,
            sanm_shfit,
            false,
        )?);

        let mut encoders = Vec::new();
        for i in 0..(num_blocks - 1) {
            encoders.push(EncoderLayerSANM::load(
                vb.pp(format!("encoders.{}", i)),
                attention_heads,
                output_size,
                output_size,
                linear_units,
                kernel_size,
                sanm_shfit,
                false,
            )?);
        }

        let mut tp_encoders = Vec::new();
        for i in 0..tp_blocks {
            tp_encoders.push(EncoderLayerSANM::load(
                vb.pp(format!("tp_encoders.{}", i)),
                attention_heads,
                output_size,
                output_size,
                linear_units,
                kernel_size,
                sanm_shfit,
                false,
            )?);
        }

        let after_norm = candle_nn::layer_norm(output_size, 1e-12, vb.pp("after_norm"))?;
        let tp_norm = candle_nn::layer_norm(output_size, 1e-12, vb.pp("tp_norm"))?;

        Ok(Self {
            encoders0,
            encoders,
            tp_encoders,
            after_norm,
            tp_norm,
            output_size,
            pos_enc: SinusoidalPositionEncoder::new(output_size),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();

        // scale x
        x = (&x * (self.output_size as f64).sqrt())?;

        // Apply positional encoding
        x = self.pos_enc.forward(&x)?;

        for layer in &self.encoders0 {
            x = layer.forward(&x)?;
        }

        for layer in &self.encoders {
            x = layer.forward(&x)?;
        }

        x = self.after_norm.forward(&x)?;

        for layer in &self.tp_encoders {
            x = layer.forward(&x)?;
        }

        x = self.tp_norm.forward(&x)?;
        Ok(x)
    }
}
