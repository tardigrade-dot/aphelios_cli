use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{bart, donut::DonutConfig, swin};

/// Donut model combining Swin encoder and BART decoder.
pub struct CusDonutModel {
    encoder: swin::SwinEncoder,
    decoder: bart::BartForCausalLM,
}

impl CusDonutModel {
    pub fn clean_kv(&mut self) {
        self.decoder.reset_kv_cache();
    }
    pub fn load(config: &DonutConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = swin::SwinEncoder::new(&config.encoder, vb.clone())?;
        let decoder = bart::BartForCausalLM::new(&config.decoder, vb)?;

        Ok(Self { encoder, decoder })
    }

    pub fn encode(&self, pixel_values: &Tensor) -> Result<Tensor> {
        self.encoder.forward(pixel_values)
    }

    pub fn decode(
        &mut self,
        decoder_input_ids: &Tensor,
        encoder_output: &Tensor,
        past_kv_len: usize,
    ) -> Result<Tensor> {
        let seq_len = decoder_input_ids.dim(1)?;
        let device = decoder_input_ids.device();

        // Create causal mask for decoder
        let attn_mask = if past_kv_len > 0 && seq_len == 1 {
            None
        } else {
            Some(bart::BartForCausalLM::create_causal_mask(seq_len, device)?)
        };

        self.decoder.forward(
            decoder_input_ids,
            Some(encoder_output),
            past_kv_len,
            attn_mask.as_ref(),
        )
    }
}
