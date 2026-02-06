use anyhow::{Error as E, Ok, Result};
use candle_core::backend::BackendDevice;
use candle_core::{DType, Device, MetalDevice, Tensor};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen3::{Config as Config3, ModelForCausalLM as Model3};
use tokenizers::Tokenizer;
use std::path::Path;

struct TextGeneration {
    model: Model3,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    fn new(
        model_dir: &Path,
        device: Device,
        seed: u64,
        temperature: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<Self> {
        let config_file = model_dir.join("config.json");
        let model_file = model_dir.join("model.safetensors");
        let tokenizer_file = model_dir.join("tokenizer.json");

        let config: Config3 = serde_json::from_slice(&std::fs::read(config_file)?)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::BF16, &device)? };
        let model = Model3::new(&config, vb)?;

        let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;
        let logits_processor = LogitsProcessor::new(seed, temperature, top_p);

        Ok(Self {
            model,
            device,
            tokenizer: tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
        })
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<String> {
        use std::io::Write;

        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let eos_token = self
            .tokenizer
            // .token_to_id(token)
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| anyhow::anyhow!("cannot find <|endoftext|> token"))?;

        let eos_token2 = self
            .tokenizer
            .token_to_id("<|im_end|>")
            .ok_or_else(|| anyhow::anyhow!("cannot find <|im_end|> token"))?;

        let mut generated_tokens = 0usize;

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
                candle_transformers::utils::apply_repeat_penalty(&logits, self.repeat_penalty, &tokens[start_at..])?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;

            if next_token == eos_token || next_token == eos_token2{
                break;
            }
        }

        let result = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;

        println!("generated result: {result}");
        let dt = start_gen.elapsed();
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );

        Ok(result)
    }
}

fn get_device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        // Ok(Device::Cpu)
        let metal: MetalDevice = MetalDevice::new(0)?;

        println!("run model on metal");
        Ok(Device::Metal(metal))
    }
}

/// 方便调用的本地推理函数
pub fn qwen_infer(
    model_dir: &str,
    prompt: &str,
    is_cpu: bool,
    sample_len: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: Option<u64>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
) -> Result<String> {

    let repeat_penalty = repeat_penalty.unwrap_or(1.1);
    let repeat_last_n = repeat_last_n.unwrap_or(64);
    let sample_len = sample_len.unwrap_or(10000);
    let device = get_device(is_cpu)?;
    let seed = seed.unwrap_or(299792458);

    println!(
        "Qwen Inference on device: {:?}, model dir: {}",
        device,
        model_dir
    );
    let mut pipeline = TextGeneration::new(
        Path::new(model_dir),
        device,
        seed,
        temperature,
        top_p,
        repeat_penalty,
        repeat_last_n,
    )?;

    pipeline.run(prompt, sample_len)
}
