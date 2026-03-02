use anyhow::{Context, Result};
use aphelios_core::utils::core_utils::{
    TEXIFY2_MODEL_DECODER_PATH, TEXIFY2_MODEL_ENCODER_PATH, TEXIFY2_TOKENIZER_PATH,
};
use ndarray::Array4;
use ort::ep::CPU;
use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use tokenizers::Tokenizer;

/// Texify2 image dimensions (fixed input size for the model)
/// Based on the error message: encoder outputs {1, 50, 170, 128}
/// The patch size is likely 4x4, so input should be 50*4=200 and 170*4=680
const IMAGE_HEIGHT: usize = 200;
const IMAGE_WIDTH: usize = 680;

/// Load and preprocess image for texify2 model
/// Converts to RGB, resizes to fixed dimensions, and normalizes
fn preprocess_image(image_path: &Path) -> Result<Array4<f32>> {
    let img = image::open(image_path)
        .with_context(|| format!("Failed to open image: {:?}", image_path))?;

    // Convert to RGB (model expects 3 channels)
    let rgb = img.to_rgb8();
    let (_width, _height) = rgb.dimensions();

    // Resize to model input dimensions using bilinear interpolation
    let resized = image::imageops::resize(
        &rgb,
        IMAGE_WIDTH as u32,
        IMAGE_HEIGHT as u32,
        image::imageops::FilterType::Triangle,
    );

    // Normalize: convert u8 [0, 255] to f32 [0, 1] and rearrange to CHW format
    let mut input_data = vec![0.0f32; 3 * IMAGE_HEIGHT * IMAGE_WIDTH];

    for (y, row) in resized.rows().enumerate() {
        for (x, pixel) in row.enumerate() {
            let base_idx = y * IMAGE_WIDTH + x;
            // CHW format: [C, H, W]
            input_data[base_idx] = pixel[0] as f32 / 255.0; // R channel
            input_data[IMAGE_HEIGHT * IMAGE_WIDTH + base_idx] = pixel[1] as f32 / 255.0; // G channel
            input_data[2 * IMAGE_HEIGHT * IMAGE_WIDTH + base_idx] = pixel[2] as f32 / 255.0; // B channel
        }
    }

    // Create tensor with shape (1, 3, H, W) - NCHW format
    let array = Array4::from_shape_vec((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), input_data)
        .context("Failed to create input tensor")?;

    Ok(array)
}

/// Load ONNX model with specified execution providers
fn load_model(
    model_path: &Path,
    execution_providers: Vec<ExecutionProviderDispatch>,
) -> Result<Session> {
    let session = Session::builder()?
        .with_execution_providers(execution_providers)?
        .commit_from_file(model_path)
        .with_context(|| format!("Failed to load model: {:?}", model_path))?;

    Ok(session)
}

/// Run encoder model to get encoder hidden states
fn run_encoder(encoder_session: &mut Session, pixel_values: &Array4<f32>) -> Result<Value> {
    let input_tensor = Value::from_array(pixel_values.clone())?;

    // Get input name from model metadata
    let input_name = encoder_session.inputs()[0].name().to_string();

    // Run inference - convert to String to avoid &str issues
    let outputs = encoder_session.run(vec![(input_name, input_tensor)])?;

    // Get the first output and extract tensor data
    let (_name, value_ref) = outputs.iter().next().context("No output from encoder")?;

    // Extract the tensor data to create a new owned Value
    let tensor_data = value_ref.try_extract_tensor::<f32>()?;
    let (shape_info, data) = tensor_data;
    let data_vec: Vec<f32> = data.iter().copied().collect();
    let shape: Vec<usize> = shape_info.iter().map(|&x| x as usize).collect();

    let output_value: Value =
        Value::from_array(ndarray::ArrayD::<f32>::from_shape_vec(shape, data_vec)?)?.into_dyn();

    Ok(output_value)
}

/// Run decoder model to generate token predictions
fn run_decoder(
    decoder_session: &mut Session,
    encoder_hidden_states: &Value,
    decoder_input_ids: &[i64],
) -> Result<(Vec<f32>, Vec<f32>)> {
    // Create decoder input IDs tensor - 2D shape (batch, seq_len)
    let batch_size = 1;
    let seq_len = decoder_input_ids.len();
    let input_ids_array =
        ndarray::Array2::from_shape_vec((batch_size, seq_len), decoder_input_ids.to_vec())?;
    let input_ids_tensor: Value = Value::from_array(input_ids_array)?.into_dyn();

    // Get encoder hidden states data - shape should be (batch, seq, hidden)
    let encoder_hs_array = encoder_hidden_states.try_extract_tensor::<f32>()?;
    let (shape_info, data) = encoder_hs_array;
    let data_vec: Vec<f32> = data.iter().copied().collect();
    let shape: Vec<usize> = shape_info.iter().map(|&x| x as usize).collect();

    let encoder_hs_tensor = Value::from_array(ndarray::ArrayD::<f32>::from_shape_vec(
        shape.clone(),
        data_vec,
    )?)?
    .into_dyn();

    // Get input names
    let input_names: Vec<String> = decoder_session
        .inputs()
        .iter()
        .map(|input| input.name().to_string())
        .collect();

    // Build inputs vec - provide all required inputs
    let mut inputs_vec: Vec<(String, Value)> = Vec::new();

    // Add input_ids
    if input_names.len() > 0 {
        inputs_vec.push((input_names[0].clone(), input_ids_tensor));
    }

    // Add encoder_hidden_states
    if input_names.len() > 1 {
        inputs_vec.push((input_names[1].clone(), encoder_hs_tensor));
    }

    // Add past_key_values as zeros (for first iteration, no cache)
    // Each past_key_values entry has shape (batch, num_heads, seq_len, head_dim)
    // Model uses 16 heads with 64 dimension each
    for i in 0..8 {
        // decoder key/value
        if input_names.len() > 2 + i * 4 {
            let zeros = vec![0.0f32; 1 * 16 * 1 * 64]; // placeholder
            let pk_tensor = Value::from_array(ndarray::Array4::<f32>::from_shape_vec(
                (1, 16, 1, 64),
                zeros,
            )?)?
            .into_dyn();
            inputs_vec.push((input_names[2 + i * 4].clone(), pk_tensor));
        }
        if input_names.len() > 3 + i * 4 {
            let zeros = vec![0.0f32; 1 * 16 * 1 * 64];
            let pv_tensor = Value::from_array(ndarray::Array4::<f32>::from_shape_vec(
                (1, 16, 1, 64),
                zeros,
            )?)?
            .into_dyn();
            inputs_vec.push((input_names[3 + i * 4].clone(), pv_tensor));
        }
        // encoder key/value
        if input_names.len() > 4 + i * 4 {
            let encoder_seq_len = shape.get(1).copied().unwrap_or(154);
            let zeros = vec![0.0f32; 1 * 16 * encoder_seq_len * 64];
            let ek_tensor = Value::from_array(ndarray::Array4::<f32>::from_shape_vec(
                (1, 16, encoder_seq_len, 64),
                zeros,
            )?)?
            .into_dyn();
            inputs_vec.push((input_names[4 + i * 4].clone(), ek_tensor));
        }
        if input_names.len() > 5 + i * 4 {
            let encoder_seq_len = shape.get(1).copied().unwrap_or(154);
            let zeros = vec![0.0f32; 1 * 16 * encoder_seq_len * 64];
            let ev_tensor = Value::from_array(ndarray::Array4::<f32>::from_shape_vec(
                (1, 16, encoder_seq_len, 64),
                zeros,
            )?)?
            .into_dyn();
            inputs_vec.push((input_names[5 + i * 4].clone(), ev_tensor));
        }
    }

    // Add use_cache_branch (boolean, false for first iteration)
    if input_names.iter().any(|n| n == "use_cache_branch") {
        let use_cache = Value::from_array(ndarray::arr1(&[false]))?.into_dyn();
        inputs_vec.push(("use_cache_branch".to_string(), use_cache));
    }

    // Run inference
    let outputs = decoder_session.run(inputs_vec)?;

    // Extract logits from outputs
    let mut logits: Option<Vec<f32>> = None;

    for (name, value) in &outputs {
        if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
            let data_vec: Vec<f32> = data.iter().copied().collect();

            // Get logits output
            if name == "logits" {
                logits = Some(data_vec);
                break;
            }
        }
    }

    Ok((logits.unwrap_or_default(), vec![]))
}

/// Greedy decoding to generate token sequence
fn greedy_decode(
    decoder_session: &mut Session,
    encoder_hidden_states: &Value,
    vocab_size: usize,
    max_tokens: usize,
    bos_token_id: i64,
    eos_token_id: i64,
    pad_token_id: i64,
) -> Result<Vec<i64>> {
    let mut tokens = vec![bos_token_id];

    for _ in 0..max_tokens {
        let (logits, _) = run_decoder(decoder_session, encoder_hidden_states, &tokens)?;

        if logits.is_empty() {
            break;
        }

        // Logits shape is (batch, seq, vocab) flattened to [batch * seq * vocab]
        // For our case: batch=1, seq=current_seq_len, vocab=vocab_size
        // We only need the logits for the last position
        let current_seq_len = tokens.len();

        // The logits for the last token position starts at:
        // (batch-1) * seq * vocab + (seq-1) * vocab = (seq-1) * vocab when batch=1
        let last_token_start = (current_seq_len - 1) * vocab_size;

        let last_token_logits = if last_token_start + vocab_size <= logits.len() {
            &logits[last_token_start..last_token_start + vocab_size]
        } else {
            eprintln!(
                "Logits out of bounds: start={}, vocab_size={}, logits_len={}",
                last_token_start,
                vocab_size,
                logits.len()
            );
            break;
        };

        // Greedy: select token with highest logit
        let max_idx = last_token_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as i64)
            .unwrap_or(pad_token_id);

        tokens.push(max_idx);

        // Stop on EOS token
        if max_idx == eos_token_id {
            break;
        }
    }

    Ok(tokens)
}

/// Decode tokens to text using tokenizer
fn decode_tokens(tokenizer: &Tokenizer, tokens: &[i64]) -> String {
    let token_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    tokenizer.decode(&token_ids, true).unwrap_or_default()
}

fn main() -> Result<()> {
    // File paths
    let encoder_file = Path::new(TEXIFY2_MODEL_ENCODER_PATH);
    let decoder_file = Path::new(TEXIFY2_MODEL_DECODER_PATH);
    let test_image = Path::new("/Users/larry/coderesp/aphelios_cli/test_data/texify2_test.png");
    let tokenizer_file = Path::new(TEXIFY2_TOKENIZER_PATH);

    // Verify files exist
    for (name, path) in [
        ("encoder model", encoder_file),
        ("decoder model", decoder_file),
        ("test image", test_image),
        ("tokenizer", tokenizer_file),
    ] {
        if !path.exists() {
            anyhow::bail!("{} not found: {:?}", name, path);
        }
    }

    println!("=== Texify2 ONNX Inference ===");
    println!("Encoder: {:?}", encoder_file);
    println!("Decoder: {:?}", decoder_file);
    println!("Test image: {:?}", test_image);

    // Load tokenizer
    println!("\nLoading tokenizer...");
    let tokenizer = Tokenizer::from_file(tokenizer_file)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    println!("Tokenizer loaded successfully");

    // Set up execution providers (CPU fallback)
    let execution_providers = vec![CPU::default().build().into()];

    // Load models
    println!("\nLoading encoder model...");
    let mut encoder_session = load_model(encoder_file, execution_providers.clone())?;
    println!("Encoder loaded successfully");
    println!(
        "  Inputs: {:?}",
        encoder_session
            .inputs()
            .iter()
            .map(|i| i.name())
            .collect::<Vec<_>>()
    );
    println!(
        "  Outputs: {:?}",
        encoder_session
            .outputs()
            .iter()
            .map(|o| o.name())
            .collect::<Vec<_>>()
    );

    println!("\nLoading decoder model...");
    let mut decoder_session = load_model(decoder_file, execution_providers)?;
    println!("Decoder loaded successfully");
    println!(
        "  Inputs: {:?}",
        decoder_session
            .inputs()
            .iter()
            .map(|i| i.name())
            .collect::<Vec<_>>()
    );
    println!(
        "  Outputs: {:?}",
        decoder_session
            .outputs()
            .iter()
            .map(|o| o.name())
            .collect::<Vec<_>>()
    );

    // Preprocess image
    println!("\nPreprocessing image...");
    let start = std::time::Instant::now();
    let pixel_values = preprocess_image(test_image)?;
    println!("Image preprocessed in {:?}", start.elapsed());
    println!("  Input shape: {:?}", pixel_values.shape());

    // Run encoder
    println!("\nRunning encoder...");
    let start = std::time::Instant::now();
    let encoder_hidden_states = run_encoder(&mut encoder_session, &pixel_values)?;
    println!("Encoder inference completed in {:?}", start.elapsed());

    // Get encoder output shape
    if let Ok((shape, _)) = encoder_hidden_states.try_extract_tensor::<f32>() {
        let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        println!("  Encoder output shape: {:?}", shape_vec);
    }

    // Run decoder with greedy decoding
    // Token IDs from generation_config.json and tokenizer_config.json
    let vocab_size = 50000; // Texify2 vocab size (from logits output)
    let max_tokens = 256;
    let eos_token_id = 2i64; // </s>
    let pad_token_id = 1i64; // <pad>
    let decoder_start_token_id = 0i64; // <s>

    println!("\nRunning decoder with greedy decoding...");
    let start = std::time::Instant::now();
    let tokens = greedy_decode(
        &mut decoder_session,
        &encoder_hidden_states,
        vocab_size,
        max_tokens,
        decoder_start_token_id,
        eos_token_id,
        pad_token_id,
    )?;
    println!("Decoder inference completed in {:?}", start.elapsed());
    println!("  Generated {} tokens", tokens.len());

    // Decode tokens to text
    let decoded_text = decode_tokens(&tokenizer, &tokens);
    println!("\n=== Decoded Text ===");
    println!("{}", decoded_text);
    println!("\n=== Inference Complete ===");

    Ok(())
}
