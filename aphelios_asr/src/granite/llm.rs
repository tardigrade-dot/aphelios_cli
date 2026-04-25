use anyhow::{Error as E, Result};
use aphelios_core::utils::common;
use aphelios_core::utils::token_output_stream::TokenOutputStream;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::granitemoehybrid::{
    self as model, GraniteMoeHybridInternalConfig,
};
use model::{GraniteMoeHybrid, GraniteMoeHybridCache, GraniteMoeHybridConfig};
use std::io::Write;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct GraniteModel {
    model: GraniteMoeHybrid,
    tokenizer: Tokenizer,
    config: GraniteMoeHybridInternalConfig,
    device: Device,
    pub(crate) dtype: DType,
    cache: GraniteMoeHybridCache,
}

impl std::ops::Deref for GraniteModel {
    type Target = DType;

    fn deref(&self) -> &Self::Target {
        &self.dtype
    }
}

impl GraniteModel {
    /// 初始化模型：支持本地路径或 HuggingFace ID
    pub fn new(model_id: &str) -> Result<Self> {
        let (device, dtype) = common::get_device_dtype();

        let path = Path::new(model_id);
        let (tokenizer_filename, config_filename, safetensors) = (
            path.join("tokenizer.json"),
            path.join("config.json"),
            path.join("model.safetensors"),
        );

        let vb_config: GraniteMoeHybridConfig =
            serde_json::from_slice(&std::fs::read(config_filename)?)?;

        let config = vb_config.clone().into_config(false);

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[safetensors], dtype, &device)? };

        let model = GraniteMoeHybrid::load(vb, &config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let cache = GraniteMoeHybridCache::new(true, dtype, &config, &device)?;
        Ok(Self {
            model,
            tokenizer,
            config,
            device,
            dtype,
            cache,
        })
    }

    /// 执行推理生成
    pub fn generate(&mut self, prompt: &str, sample_len: usize, temp: f64) -> Result<()> {
        let chat_prompt = format!(
            "<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
        );

        let mut tokens = self
            .tokenizer
            .encode(chat_prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let mut tos = TokenOutputStream::new(self.tokenizer.clone());

        let mut logits_processor =
            LogitsProcessor::from_sampling(299792458, Sampling::All { temperature: temp });

        print!("Assistant: ");
        let mut index_pos = 0;

        for index in 0..sample_len {
            let (context_size, context_index) = if index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };
            let context = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(context, &self.device)?.unsqueeze(0)?;
            let logits = self
                .model
                .forward(&input, context_index, &mut self.cache)?
                .squeeze(0)?;

            index_pos += context.len();
            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);

            if next_token == self.config.eos_token_id.unwrap_or(100257) {
                break;
            }
            if let Some(token) = tos.next_token(next_token)? {
                print!("{token}");
                std::io::stdout().flush()?;
            }
        }
        println!();
        Ok(())
    }
}
