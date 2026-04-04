use candle_core::{DType, Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{bart, donut::DonutConfig, swin};

/// Donut model combining Swin encoder and BART decoder.
pub struct CusDonutModel {
    encoder: swin::SwinEncoder,
    decoder: bart::BartForCausalLM,
    dtype: candle_core::DType,
}

impl CusDonutModel {
    pub fn clean_kv(&mut self) {
        self.decoder.reset_kv_cache();
    }
    pub fn load(
        config: &DonutConfig,
        tensors: std::collections::HashMap<String, Tensor>,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let mut encoder_tensors = std::collections::HashMap::new();
        let mut decoder_tensors = std::collections::HashMap::new();

        for (name, tensor) in tensors {
            if name.starts_with("encoder.") {
                encoder_tensors.insert(name.clone(), tensor.to_dtype(DType::F32)?);
            } else if name.starts_with("decoder.") {
                decoder_tensors.insert(name.clone(), tensor.to_dtype(dtype)?);
            } else {
                // Shared or other tensors
                encoder_tensors.insert(name.clone(), tensor.to_dtype(DType::F32)?);
                decoder_tensors.insert(name.clone(), tensor.to_dtype(dtype)?);
            }
        }

        // swin::SwinEncoder::new expects VarBuilder to be at the level where it can .pp("encoder")
        let encoder_vb = VarBuilder::from_tensors(encoder_tensors, DType::F32, device);
        let encoder = swin::SwinEncoder::new(&config.encoder, encoder_vb)?;

        // bart::BartForCausalLM::new expects VarBuilder to be at the level where it can .pp("decoder")
        let decoder_vb = VarBuilder::from_tensors(decoder_tensors, dtype, device);
        let decoder = bart::BartForCausalLM::new(&config.decoder, decoder_vb)?;

        Ok(Self {
            encoder,
            decoder,
            dtype,
        })
    }

    pub fn encode(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // Encoder always expects F32 on Metal due to candle-transformers bugs
        let pixel_values = pixel_values.to_dtype(DType::F32)?;
        let encoder_output = self.encoder.forward(&pixel_values)?;

        // Convert output to model dtype for decoder
        encoder_output.to_dtype(self.dtype)
    }

    pub fn decode(
        &mut self,
        decoder_input_ids: &Tensor,
        encoder_output: &Tensor,
        past_kv_len: usize,
    ) -> Result<Tensor> {
        let seq_len = decoder_input_ids.dim(1)?;
        let device = decoder_input_ids.device();
        let dtype = self.dtype; // Use model's stored dtype

        // Create causal mask for decoder
        let attn_mask = if past_kv_len > 0 && seq_len == 1 {
            None
        } else {
            let mask = bart::BartForCausalLM::create_causal_mask(seq_len, device)?;
            Some(mask.to_dtype(dtype)?)
        };

        // Ensure inputs are correct dtype
        let encoder_output = if encoder_output.dtype() != dtype {
            encoder_output.to_dtype(dtype)?
        } else {
            encoder_output.clone()
        };

        self.decoder.forward(
            decoder_input_ids,
            Some(&encoder_output),
            past_kv_len,
            attn_mask.as_ref(),
        )
    }
}
