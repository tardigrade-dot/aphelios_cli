//! Dolphin ONNX Inference Example - Layout Recognition Stage
//!
//! This example demonstrates ONNX runtime inference for Dolphin OCR model.
//! Reference implementation: /Users/larry/coderesp/aphelios_cli/aphelios_ocr/src/bin/dolphin_ocr.rs

use anyhow::{Context, Result};
use aphelios_core::utils::core_utils;
use aphelios_ocr::dolphin::dolphin_utils::{self, transform_to_pixel_dynamic};
use image::{DynamicImage, GenericImageView, RgbImage, Rgba};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use ndarray::Array4;
use ort::session::Session;
use ort::value::Value;
use regex::Regex;
use std::path::Path;
use tokenizers::Tokenizer;

/// Dolphin model image dimensions (from preprocessor_config.json)
const IMAGE_HEIGHT: usize = 896;
const IMAGE_WIDTH: usize = 896;

/// Load and preprocess image for Dolphin model
/// 修改后的预处理函数：采用居中对齐 (Center Padding)
fn preprocess_image(image_path: &Path) -> Result<Array4<f32>> {
    let img = image::open(image_path)
        .with_context(|| format!("Failed to open image: {:?}", image_path))?;

    // 1. 按照 target 尺寸进行等比例缩放
    let resized = img.resize(
        IMAGE_WIDTH as u32,
        IMAGE_HEIGHT as u32,
        image::imageops::FilterType::Triangle,
    );

    // 2. 创建黑色画布
    let mut canvas = RgbImage::from_pixel(
        IMAGE_WIDTH as u32,
        IMAGE_HEIGHT as u32,
        image::Rgb([0, 0, 0]),
    );

    // 3. 计算居中偏移量 (关键：与 get_tensor_from_image 一致)
    let x_offset = (IMAGE_WIDTH as u32 - resized.width()) / 2;
    let y_offset = (IMAGE_HEIGHT as u32 - resized.height()) / 2;

    // 4. 将缩放后的图像居中放置在画布上
    image::imageops::overlay(
        &mut canvas,
        &resized.to_rgb8(),
        x_offset as i64,
        y_offset as i64,
    );

    let (width, height) = (canvas.width() as usize, canvas.height() as usize);

    // 5. 标准化处理 (Donut/Dolphin 均使用 0.5 均值和标准差)
    let image_mean = [0.5f32, 0.5, 0.5];
    let image_std = [0.5f32, 0.5, 0.5];

    let mut normalized = vec![0f32; 3 * height * width];

    for (c, (&mean, &std)) in image_mean.iter().zip(image_std.iter()).enumerate() {
        for y in 0..height {
            for x in 0..width {
                let pixel = canvas.get_pixel(x as u32, y as u32);
                let idx = c * height * width + y * width + x;
                // 将 0-255 映射到 [-1.0, 1.0] 空间
                normalized[idx] = (pixel[c] as f32 / 255.0 - mean) / std;
            }
        }
    }

    // 6. 转换为 ndarray 格式输出
    let array = Array4::from_shape_vec((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), normalized)
        .context("Failed to create input tensor")?;

    // 可选：调试保存，检查图片是否居中
    // canvas.save("debug_centered_input.png")?;

    Ok(array)
}

/// Load ONNX model
fn load_model(model_path: &Path) -> Result<Session> {
    let execution_providers = core_utils::get_available_ep();

    let session = Session::builder()?
        .with_execution_providers(execution_providers)?
        .commit_from_file(model_path)
        .with_context(|| format!("Failed to load model: {:?}", model_path))?;

    Ok(session)
}

/// Run encoder-decoder model for layout recognition
fn run_layout_recognition(
    encoder_session: &mut Session,
    decoder_session: &mut Session,
    pixel_values: &Array4<f32>,
    tokenizer: &Tokenizer,
    task_prompt: &str,
    max_tokens: usize,
) -> Result<String> {
    let eos_token_id = tokenizer.token_to_id("</s>").unwrap_or(2) as i64;

    // Run encoder
    let encoder_input = Value::from_array(pixel_values.clone())?;
    let encoder_input_name = encoder_session.inputs()[0].name().to_string();
    let encoder_outputs: ort::session::SessionOutputs<'_> =
        encoder_session.run(vec![(encoder_input_name, encoder_input)])?;

    // Get encoder hidden states
    let encoder_hidden_states = encoder_outputs
        .iter()
        .find(|(name, _)| *name == "last_hidden_state")
        .and_then(|(_, value_ref)| {
            value_ref
                .try_extract_tensor::<f32>()
                .ok()
                .map(|(shape_info, data)| {
                    let data_vec: Vec<f32> = data.iter().copied().collect();
                    let shape: Vec<usize> = shape_info.iter().map(|&x| x as usize).collect();
                    Value::from_array(
                        ndarray::ArrayD::<f32>::from_shape_vec(shape, data_vec).unwrap(),
                    )
                    .unwrap()
                    .into_dyn()
                })
        })
        .context("No last_hidden_state output from encoder")?;

    // Tokenize prompt
    let tokens = tokenizer
        .encode(task_prompt, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let mut token_ids: Vec<i64> = tokens.get_ids().iter().map(|&x| x as i64).collect();

    let decoder_input_names: Vec<String> = decoder_session
        .inputs()
        .iter()
        .map(|input| input.name().to_string())
        .collect();

    // Greedy decoding loop
    for _step in 0..max_tokens {
        // Pass full sequence (decoder doesn't use KV cache)
        let input_ids_array =
            ndarray::Array2::from_shape_vec((1, token_ids.len()), token_ids.clone())?;
        let input_ids_tensor = Value::from_array(input_ids_array)?.into_dyn();

        // Build decoder inputs
        let mut inputs_vec: Vec<(String, Value)> = Vec::new();
        if decoder_input_names.len() > 0 {
            inputs_vec.push((decoder_input_names[0].clone(), input_ids_tensor));
        }
        if decoder_input_names.len() > 1 {
            let encoder_hs_array = encoder_hidden_states.try_extract_tensor::<f32>()?;
            let (shape_info, data) = encoder_hs_array;
            let data_vec: Vec<f32> = data.iter().copied().collect();
            let shape: Vec<usize> = shape_info.iter().map(|&x| x as usize).collect();
            let encoder_hs_tensor = Value::from_array(ndarray::ArrayD::<f32>::from_shape_vec(
                shape.clone(),
                data_vec,
            )?)?
            .into_dyn();
            inputs_vec.push((decoder_input_names[1].clone(), encoder_hs_tensor));
        }

        let outputs = decoder_session.run(inputs_vec)?;

        // Extract logits
        let logits_result =
            outputs
                .iter()
                .find(|(name, _)| *name == "logits")
                .and_then(|(_, value)| {
                    value.try_extract_tensor::<f32>().ok().map(|(shape, data)| {
                        let data_vec: Vec<f32> = data.iter().copied().collect();
                        let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
                        (data_vec, shape_vec)
                    })
                });

        if let Some((logits_data, shape_vec)) = logits_result {
            let vocab_size = shape_vec.last().copied().unwrap_or(73921);
            let seq_len = shape_vec.get(1).copied().unwrap_or(1);
            let position = seq_len - 1;
            let token_start = position * vocab_size;

            let next_token_logits = if token_start + vocab_size <= logits_data.len() {
                &logits_data[token_start..token_start + vocab_size]
            } else {
                break;
            };

            // Apply temperature
            let temperature = 0.8f32;
            let scaled_logits: Vec<f32> =
                next_token_logits.iter().map(|&x| x / temperature).collect();

            let next_token = scaled_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx as i64)
                .unwrap_or(1);

            token_ids.push(next_token);

            if next_token == eos_token_id {
                break;
            }

            // Stop on repetition
            if token_ids.len() > 5 {
                let last_five: Vec<&i64> = token_ids.iter().rev().take(5).collect();
                if last_five.iter().all(|&x| x == &next_token) {
                    break;
                }
            }
        } else {
            break;
        }
    }

    let decoded = tokenizer
        .decode(
            &token_ids.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            false,
        )
        .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;

    let result = decoded
        .replace(task_prompt, "")
        .replace("</s>", "")
        .replace("\n", "");
    Ok(result)
}

/// Parse layout string to extract bounding boxes
/// Format: [x1,y1,x2,y2][label][PAIR_SEP][x1,y1,x2,y2][label]...
fn parse_layout_bboxes(layout_str: &str) -> Result<Vec<([i32; 4], String)>> {
    let re = Regex::new(r"\[(\d+),(\d+),(\d+),(\d+)\]\[([^\]]+)\]")?;
    let mut bboxes = Vec::new();

    for cap in re.captures_iter(layout_str) {
        let x1: u32 = cap[1].parse()?;
        let y1: u32 = cap[2].parse()?;
        let x2: u32 = cap[3].parse()?;
        let y2: u32 = cap[4].parse()?;
        let label = cap[5].to_string();

        // Skip PAIR_SEP markers
        if label == "PAIR_SEP" {
            continue;
        }

        bboxes.push(([x1 as i32, y1 as i32, x2 as i32, y2 as i32], label));
    }

    Ok(bboxes)
}

/// Draw bounding boxes on image and save
fn draw_bboxes_and_save(
    img: &DynamicImage,
    bboxes: &[([u32; 4], String)],
    save_path: &Path,
) -> Result<()> {
    let (orig_w, orig_h) = img.dimensions();

    // 1. 模拟与 preprocess_image 完全一致的缩放逻辑
    // 获取缩放后的实际像素尺寸，这比手动 round() 更可靠
    // let resized = img.resize(896, 896, image::imageops::FilterType::Triangle);
    // let r_w = resized.width() as f32;
    // let r_h = resized.height() as f32;

    // // 2. 重新计算比例和偏移（基于实际 resized 尺寸）
    // let ratio_x = r_w / orig_w as f32;
    // let ratio_y = r_h / orig_h as f32;

    // // 这里非常重要：由于 resize 保持比例，ratio_x 和 ratio_y 理论上相等（取最小值那个）
    // let ratio = ratio_x.min(ratio_y);

    // // 计算居中偏移量
    // let offset_x = (896.0 - r_w) / 2.0;
    // let offset_y = (896.0 - r_h) / 2.0;

    let mut img_rgba = img.to_rgba8();

    for (bbox, label) in bboxes {
        // 模型输出通常是 0-1000 归一化坐标 -> 映射到 896 画布像素
        // let x1_896 = (bbox[0] as f32 / 1000.0) * 896.0;
        // let y1_896 = (bbox[1] as f32 / 1000.0) * 896.0;
        // let x2_896 = (bbox[2] as f32 / 1000.0) * 896.0;
        // let y2_896 = (bbox[3] as f32 / 1000.0) * 896.0;

        // // 3. 核心还原公式：先减偏移，再除比例，并使用 round() 减少舍入误差
        // let rx1 = ((x1_896 - offset_x) / ratio).round() as i32;
        // let ry1 = ((y1_896 - offset_y) / ratio).round() as i32;
        // let rx2 = ((x2_896 - offset_x) / ratio).round() as i32;
        // let ry2 = ((y2_896 - offset_y) / ratio).round() as i32;

        // // 4. 边界处理：防止坐标溢出原图范围
        // let x1 = rx1.clamp(0, orig_w as i32 - 1);
        // let y1 = ry1.clamp(0, orig_h as i32 - 1);
        // let x2 = rx2.clamp(0, orig_w as i32);
        // let y2 = ry2.clamp(0, orig_h as i32);

        let [x1, y1, x2, y2] = bbox;
        let w = (x2 - x1).max(1) as u32;
        let h = (y2 - y1).max(1) as u32;

        // 绘制逻辑
        let rect = Rect::at(*x1 as i32, *y1 as i32).of_size(w, h);

        // 可选：根据标签区分颜色
        let color = if label.contains("para") {
            Rgba([255, 0, 0, 255]) // 段落用红色
        } else {
            Rgba([0, 255, 0, 255]) // 其他用绿色
        };

        draw_hollow_rect_mut(&mut img_rgba, rect, color);
    }

    img_rgba.save(save_path)?;
    Ok(())
}

//  [78,59,514,239][half_para][PAIR_SEP][78,244,515,425][para][PAIR_SEP][164,456,398,485][sec_1][PAIR_SEP][81,507,517,687][para][PAIR_SEP][82,692,517,769][para][PAIR_SEP][103,780,149,792][foot][page_num]
fn main() -> Result<()> {
    let encoder_file =
        Path::new("/Volumes/sw/onnx_models/Dolphin-1.5-onnx/onnx/encoder_model.onnx");
    let decoder_file =
        Path::new("/Volumes/sw/onnx_models/Dolphin-1.5-onnx/onnx/decoder_model.onnx");
    let test_image = Path::new("/Users/larry/coderesp/aphelios_cli/test_data/page_32.png");
    let tokenizer_file = Path::new("/Volumes/sw/onnx_models/Dolphin-1.5-onnx/tokenizer.json");

    // Expected output format:
    // [226,60,664,239][half_para][PAIR_SEP][227,244,664,425][para][PAIR_SEP]...

    println!("=== Dolphin ONNX Layout Recognition ===");
    println!("Encoder: {:?}", encoder_file);
    println!("Decoder: {:?}", decoder_file);
    println!("Test image: {:?}", test_image);

    if !encoder_file.exists()
        || !decoder_file.exists()
        || !test_image.exists()
        || !tokenizer_file.exists()
    {
        println!("\nNote: Model files not found. Please update paths.");
        return Ok(());
    }

    println!("\nLoading tokenizer...");
    let tokenizer = Tokenizer::from_file(tokenizer_file)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    println!("Loading encoder model...");
    let mut encoder_session = load_model(encoder_file)?;
    println!(
        "  Inputs: {:?}",
        encoder_session
            .inputs()
            .iter()
            .map(|i| i.name())
            .collect::<Vec<_>>()
    );

    println!("Loading decoder model...");
    let mut decoder_session = load_model(decoder_file)?;
    println!(
        "  Inputs: {:?}",
        decoder_session
            .inputs()
            .iter()
            .map(|i| i.name())
            .collect::<Vec<_>>()
    );

    println!("\nPreprocessing image...");
    let start = std::time::Instant::now();
    let pixel_values = preprocess_image(test_image)?;

    let device = core_utils::get_default_device(false)?;
    let img = image::ImageReader::open(test_image)
        .with_context(|| format!("Failed to open image file {}", test_image.to_str().unwrap()))?
        .decode()
        .with_context(|| format!("Failed to decode image {}", test_image.to_str().unwrap()))?;

    let img_tensor = dolphin_utils::get_tensor_from_image(&img, 896, 896, &device);

    println!("Image preprocessed in {:?}", start.elapsed());

    let task_prompt = "<s>Parse the reading order of this document. <Answer/>";
    println!("\nRunning layout recognition...");

    let start = std::time::Instant::now();
    let layout_str = run_layout_recognition(
        &mut encoder_session,
        &mut decoder_session,
        &pixel_values,
        &tokenizer,
        task_prompt,
        512,
    )?;
    println!("Completed in {:?}", start.elapsed());

    println!("\n=== Layout Recognition Result ===");
    println!("{}", layout_str);

    // Parse and draw bounding boxes
    println!("\n=== Parsing Bounding Boxes ===");
    let bboxes = parse_layout_bboxes(&layout_str)?;
    println!("Found {} bounding boxes:", bboxes.len());

    let (img_w, img_h) = img.dimensions();
    let mut draw_bbox: Vec<([u32; 4], String)> = Vec::new();
    for (bbox, label) in &bboxes {
        println!(
            "  [{}] {},{},{},{}",
            label, bbox[0], bbox[1], bbox[2], bbox[3]
        );

        let n_bbox = transform_to_pixel_dynamic(bbox, img_w, img_h, 896, 895);
        draw_bbox.push((n_bbox, label.to_string()));
    }

    // Draw bboxes on image
    let output_dir = Path::new("/Users/larry/coderesp/aphelios_cli/output");
    let _ = std::fs::create_dir_all(output_dir);
    let output_path = output_dir.join("dolphin_layout.png");

    let original_img = image::open(test_image)?;
    draw_bboxes_and_save(&original_img, &draw_bbox, &output_path)?;

    println!("\nLayout visualization saved to: {:?}", output_path);

    println!("\n=== Inference Complete ===");

    Ok(())
}
