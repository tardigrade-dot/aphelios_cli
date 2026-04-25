use std::path::Path;

use anyhow::{Error as E, Result};

use aphelios_core::utils::common::get_device_dtype;
use aphelios_core::utils::token_output_stream::TokenOutputStream;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen3::{Config as Config3, ModelForCausalLM as Model3};
use tokenizers::Tokenizer;

enum Model {
    Base3(Model3),
}

impl Model {
    fn forward(&mut self, xs: &Tensor, s: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Base3(ref mut m) => m.forward(xs, s),
        }
    }
}

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };
        let eos_token2 = match self.tokenizer.get_token("<|im_end|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|im_end|> token"),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || next_token == eos_token2 {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

fn format_prompt(prompt: &str, use_chat_template: bool, thinking: bool) -> String {
    if !use_chat_template {
        return prompt.to_string();
    }
    let think_tag = if thinking { " /think" } else { " /no_think" };
    format!("<|im_start|>user\n{prompt}{think_tag}<|im_end|>\n<|im_start|>assistant\n")
}

pub fn qwen3_llm(prompt: &str, model_dir: &str) -> Result<()> {
    let use_chat_template = true;
    let thinking = false;

    let model_path = Path::new(model_dir);
    assert!(model_path.exists(), "model_dir not found: {}", model_dir);
    let tokenizer_filename = model_path.join("tokenizer.json");
    let config_file = model_path.join("config.json");

    let filenames = vec![model_path.join("model.safetensors")];
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let (device, dtype) = get_device_dtype();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

    let config: Config3 = serde_json::from_slice(&std::fs::read(config_file)?)?;
    let model = Model::Base3(Model3::new(&config, vb)?);

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline =
        TextGeneration::new(model, tokenizer, 299792458, None, None, 1.1, 64, &device);
    let prompt = format_prompt(prompt, use_chat_template, thinking);
    pipeline.run(&prompt, 10000)?;
    Ok(())
}
