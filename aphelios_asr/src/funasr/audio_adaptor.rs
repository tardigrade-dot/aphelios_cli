use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};

// Simple Multi-Head Attention with separate Q, K, V
#[derive(Debug)]
pub struct MultiHeadedAttention {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    h: usize,
    d_k: usize,
}

impl MultiHeadedAttention {
    pub fn load(
        vb: VarBuilder,
        input_size: usize,
        output_size: usize,
        attention_heads: usize,
    ) -> Result<Self> {
        let linear_q = candle_nn::linear(input_size, output_size, vb.pp("linear_q"))?;
        let linear_k = candle_nn::linear(input_size, output_size, vb.pp("linear_k"))?;
        let linear_v = candle_nn::linear(input_size, output_size, vb.pp("linear_v"))?;
        let linear_out = candle_nn::linear(output_size, output_size, vb.pp("linear_out"))?;

        Ok(Self {
            linear_q,
            linear_k,
            linear_v,
            linear_out,
            h: attention_heads,
            d_k: output_size / attention_heads,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;

        let q = self.linear_q.forward(x)?;
        let k = self.linear_k.forward(x)?;
        let v = self.linear_v.forward(x)?;

        let q_h = q.reshape(vec![b, t, self.h, self.d_k])?.transpose(1, 2)?;
        let k_h = k.reshape(vec![b, t, self.h, self.d_k])?.transpose(1, 2)?;
        let v_h = v.reshape(vec![b, t, self.h, self.d_k])?.transpose(1, 2)?;

        let q_h = (q_h * (1.0 / (self.d_k as f64).sqrt()))?;
        let scores = q_h
            .contiguous()?
            .matmul(&k_h.transpose(2, 3)?.contiguous()?)?;

        let attn = candle_nn::ops::softmax(&scores, 3)?;
        let out = attn.matmul(&v_h.contiguous()?)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape(vec![b, t, self.h * self.d_k])?;
        self.linear_out.forward(&out)
    }
}

// PositionwiseFeedForward
#[derive(Debug)]
pub struct PositionwiseFeedForward {
    w_1: Linear,
    w_2: Linear,
}

impl PositionwiseFeedForward {
    pub fn load(
        vb: VarBuilder,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) -> Result<Self> {
        let w_1 = candle_nn::linear(input_size, hidden_size, vb.pp("w_1"))?;
        let w_2 = candle_nn::linear(hidden_size, output_size, vb.pp("w_2"))?;
        Ok(Self { w_1, w_2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.w_2.forward(&self.w_1.forward(x)?.relu()?)
    }
}

// EncoderLayer (Standard Transformer Layer)
#[derive(Debug)]
pub struct EncoderLayer {
    self_attn: MultiHeadedAttention,
    feed_forward: PositionwiseFeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl EncoderLayer {
    pub fn load(
        vb: VarBuilder,
        attention_heads: usize,
        input_size: usize,
        output_size: usize,
        ffn_dim: usize,
    ) -> Result<Self> {
        let self_attn = MultiHeadedAttention::load(
            vb.pp("self_attn"),
            input_size,
            output_size,
            attention_heads,
        )?;
        let feed_forward = PositionwiseFeedForward::load(
            vb.pp("feed_forward"),
            output_size,
            ffn_dim,
            output_size,
        )?;

        let norm1 = candle_nn::layer_norm(input_size, 1e-12, vb.pp("norm1"))?;
        let norm2 = candle_nn::layer_norm(output_size, 1e-12, vb.pp("norm2"))?;

        Ok(Self {
            self_attn,
            feed_forward,
            norm1,
            norm2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let nx = self.norm1.forward(x)?;
        let attn = self.self_attn.forward(&nx)?;
        let x = (x + attn)?;

        let nx = self.norm2.forward(&x)?;
        let ff = self.feed_forward.forward(&nx)?;
        let x = (x + ff)?;

        Ok(x)
    }
}

#[derive(Debug)]
pub struct AudioAdaptor {
    linear1: Linear,
    linear2: Linear,
    blocks: Vec<EncoderLayer>,
}

impl AudioAdaptor {
    pub fn load(
        vb: VarBuilder,
        num_blocks: usize,
        attention_heads: usize,
        encoder_dim: usize,
        llm_dim: usize,
        linear_ffn_dim: usize,
    ) -> Result<Self> {
        let linear1 = candle_nn::linear(encoder_dim, linear_ffn_dim, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(linear_ffn_dim, llm_dim, vb.pp("linear2"))?;

        // Inner FFN dim for the blocks is llm_dim // 4 by default in FunASR
        let block_ffn_dim = llm_dim / 4;

        let mut blocks = Vec::new();
        for i in 0..num_blocks {
            blocks.push(EncoderLayer::load(
                vb.pp(format!("blocks.{}", i)),
                attention_heads,
                llm_dim,
                llm_dim,
                block_ffn_dim,
            )?);
        }
        Ok(Self {
            linear1,
            linear2,
            blocks,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear2.forward(&self.linear1.forward(x)?.relu()?)?;
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

#[derive(Debug)]
pub struct CTCDecoder {
    blocks: Vec<EncoderLayer>,
    pub ctc_lo: Linear,
}

impl CTCDecoder {
    pub fn load(
        vb: VarBuilder,     // ctc_decoder prefix
        vb_ctc: VarBuilder, // ctc prefix
        num_blocks: usize,
        attention_heads: usize,
        input_size: usize,
        output_size: usize,
        ffn_dim: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..num_blocks {
            blocks.push(EncoderLayer::load(
                vb.pp(format!("blocks.{}", i)),
                attention_heads,
                output_size,
                output_size,
                ffn_dim,
            )?);
        }
        let ctc_lo = candle_nn::linear(output_size, vocab_size, vb_ctc.pp("ctc_lo"))?;
        Ok(Self { blocks, ctc_lo })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        self.ctc_lo.forward(&x)
    }
}
