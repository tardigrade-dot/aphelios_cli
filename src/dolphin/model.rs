use crate::commands::donut2::CusDonutModel;
use crate::dolphin::utils;
use anyhow::Ok;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor, safetensors};
use candle_nn::VarBuilder;
use candle_transformers::models::donut::DonutConfig;
use futures_util::{StreamExt, pin_mut};
use image::{DynamicImage, GenericImageView, RgbImage};
use regex::Regex;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use tokenizers::AddedToken;
use tokenizers::Tokenizer;
use tracing::info;

fn load_model(model_id: &String) -> Result<(CusDonutModel, DonutConfig, Tokenizer, Device)> {
    let model_path = PathBuf::from(model_id);

    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    info!("Using device: {:?}", device);

    let config_str = std::fs::read_to_string(model_path.join("config.json"))?;
    let config: DonutConfig = serde_json::from_str(&config_str)?;

    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(E::msg)?;

    let mut tensors = { safetensors::load(model_path.join("model.safetensors"), &device)? };

    let lm_head_weight_key = "decoder.lm_head.weight";
    let lm_head_bias_key = "decoder.lm_head.bias";

    if tensors.contains_key(lm_head_weight_key) && !tensors.contains_key(lm_head_bias_key) {
        info!("Mapping: Detected missing lm_head.bias, injecting zero-bias patch...");
        let weight = &tensors[lm_head_weight_key];
        let out_dim = weight.dim(0)?;

        let fake_bias = Tensor::zeros(out_dim, DType::F32, &device)?;
        tensors.insert(lm_head_bias_key.to_string(), fake_bias);
    }
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

    // 5. 初始化模型
    let model = CusDonutModel::load(&config, vb)?;
    Ok((model, config, tokenizer, device))
}

pub async fn dolphin_ocr(
    model_id: &String,
    image_path: &String,
    output_dir: &String,
) -> Result<Vec<String>> {
    utils::init_tracing();

    let output_path = Path::new(output_dir);
    let (mut model, config, tokenizer, device) = load_model(model_id)?;

    println!("Dolphin-1.5 model loaded from local path.");

    let mut res_li = vec![];
    if "pdf"
        == image_path
            .rsplit('.')
            .next()
            .unwrap_or("")
            .to_lowercase()
            .as_str()
    {
        let img_ite = utils::load_pdf_images(image_path).enumerate();
        pin_mut!(img_ite);
        while let Some((idx, img_result)) = img_ite.next().await {
            run_ocr_single_img(
                &img_result?,
                output_path,
                idx,
                &mut model,
                &config,
                &tokenizer,
                &device,
                &mut res_li,
            );
        }
    } else {
        let img = image::ImageReader::open(image_path)?.decode()?;
        run_ocr_single_img(
            &img,
            output_path,
            0,
            &mut model,
            &config,
            &tokenizer,
            &device,
            &mut res_li,
        );
    }
    Ok(res_li)
}

fn run_ocr_single_img(
    img: &DynamicImage,
    output_path: &Path,
    idx: usize,
    model: &mut CusDonutModel,
    config: &DonutConfig,
    tokenizer: &Tokenizer,
    device: &Device,
    res_li: &mut Vec<String>,
) -> Result<()> {
    let copped_img_path = PathBuf::from(output_path).join("copped_img");
    let _ = fs::create_dir_all(&copped_img_path);

    info!("正在处理第{}页...", idx);
    let output_file: PathBuf = output_path.join(format!("{}_page.txt", idx));
    if fs::exists(&output_file)? {
        info!("{} exist, to process next ", &output_file.to_str().unwrap());
    }
    let img_tensor = utils::get_tensor_from_image(
        &img,
        config.image_height() as u32,
        config.image_width() as u32,
        &device,
    );
    let task_prompt = "<s>Parse the reading order of this document. <Answer/>";
    let mut layout_str = generate_text(
        model,
        &img_tensor,
        &task_prompt,
        &config,
        &tokenizer,
        &device,
    )?;

    layout_str = layout_str.replace(task_prompt, "").replace("</s>", "");
    let mut res = run_ocr_second_stage(
        &img,
        &layout_str,
        model,
        &config,
        &tokenizer,
        &device,
        idx,
        &copped_img_path,
    )?;

    let _ = fs::write(&output_file, format!("{}", &res.join("\n")));
    res_li.append(&mut res);
    Ok(())
}

fn run_ocr_second_stage(
    full_image: &DynamicImage,
    layout_str: &str,
    model: &mut CusDonutModel,
    config: &DonutConfig,
    tokenizer: &Tokenizer,
    device: &Device,
    i_index: usize,
    copped_img_path: &PathBuf,
) -> Result<Vec<String>> {
    let (img_w, img_h) = full_image.dimensions();
    let target_width = config.image_width() as u32;
    let target_height = config.image_height() as u32;
    let re = Regex::new(r"\[(\d+,\d+,\d+,\d+)\]\[([^\]]+)\]")?;

    let mut reading_order = 0;

    let layout_draw_path = copped_img_path.join(format!("{}-layout.png", i_index));
    let mut bbox_list: Vec<[u32; 4]> = Vec::new();
    let mut ocr_text_li = Vec::new();
    for cap in re.captures_iter(layout_str) {
        let coords_raw: Vec<i32> = cap[1].split(',').map(|s| s.parse().unwrap()).collect();
        let label = &cap[2];

        if label == "fig" {
            continue;
        }

        let bbox = transform_to_pixel_dynamic(
            [coords_raw[0], coords_raw[1], coords_raw[2], coords_raw[3]],
            img_w,
            img_h,
            target_width,
            target_height,
        );

        bbox_list.push(bbox);
        let cropped_img = utils::crop_image(full_image, bbox, 5);
        let pixel_values = preprocess_like_donut(&cropped_img, config, device)?;

        let prompt = match label {
            "tab" => "<s>Parse the table in the image. <Answer/>",
            "equ" => "<s>Read formula in the image. <Answer/>",
            _ => "<s>Read text in the image. <Answer/>",
        };

        info!("OCR Reading [{}]: [{:?}]", label, bbox);

        let result_text = generate_text(model, &pixel_values, prompt, config, tokenizer, device)?;

        info!("Result {}: {}", reading_order, result_text);
        reading_order += 1;
        ocr_text_li.push(format!(
            "[{}] - [{}] : {}",
            reading_order, label, result_text
        ));
    }

    utils::draw_bbox_and_save_multi(full_image, &bbox_list, 5, &layout_draw_path);
    Ok(ocr_text_li)
}

#[macro_export]
macro_rules! measure_time {

    ($desc:expr, $expr:expr) => {{
        println!("start {}", $desc);
        let start = std::time::Instant::now();
        let result: _ = $expr; // 自动推导类型
        let duration = start.elapsed();
        println!("end {}: {} ms", $desc, duration.as_millis());
        result
    }};

    ($($tt:tt)*) => {{
        println!("start");
        let start = std::time::Instant::now();
        let result: _ = { $($tt)* };
        let duration = start.elapsed();
        println!("Time: {} ms", duration.as_millis());
        result
    }};
}

fn generate_text(
    model: &mut CusDonutModel,
    pixel_values: &Tensor,
    prompt: &str,
    config: &DonutConfig,
    tokenizer: &Tokenizer,
    device: &Device,
) -> Result<String> {
    model.clean_kv();

    let decoded = measure_time! {
        let tokens = tokenizer.encode(prompt, false).map_err(E::msg)?;
        let mut token_ids = tokens.get_ids().to_vec();
        // 编码图像
        let encoder_output = model.encode(&pixel_values)?;

        // 解码循环 (带 KV Cache 优化)
        for i in 0..2048 {
            let last_token_only = i > 0;
            let decoder_input = if last_token_only {
                &token_ids[token_ids.len() - 1..]
            } else {
                &token_ids[..]
            };

            let input_tensor = Tensor::new(decoder_input, &device)?.unsqueeze(0)?;
            let logits = match model.decode(
                &input_tensor,
                &encoder_output,
                if last_token_only {
                    token_ids.len() - 1
                } else {
                    0
                },
            ) {
                std::result::Result::Ok(l) => l,
                Err(e) => {
                    println!("Candle Error detected: {:?}", e);
                    return Err(e.into());
                }
            };

            // 取最后一个 token 的 logits 并贪婪采样
            let logits = logits.squeeze(0)?.get(logits.dim(1)? - 1)?;
            let next_token = logits.argmax(0)?.to_scalar::<u32>()?;

            token_ids.push(next_token);
            if next_token == config.decoder.eos_token_id {
                break;
            }
        }

        // 8. 输出结果
        let decoded = tokenizer
            .decode(&token_ids, false)
            .map_err(E::msg)?
            .replace(prompt, "")
            .replace("</s>", "")
            .replace("\n", "");
        decoded
    };

    Ok(decoded)
}

fn transform_to_pixel_dynamic(
    bbox_coords: [i32; 4],
    img_width: u32,
    img_height: u32,
    target_width: u32,
    target_height: u32,
) -> [u32; 4] {
    let [x1, y1, x2, y2] = bbox_coords;

    // Calculate the aspect ratio preserved resize
    let scale_w = target_width as f32 / img_width as f32;
    let scale_h = target_height as f32 / img_height as f32;
    let scale = scale_w.min(scale_h); // Take minimum scale to maintain aspect ratio

    // Calculate the actual size of the resized image within the target canvas
    let resized_width = (img_width as f32 * scale) as u32;
    let resized_height = (img_height as f32 * scale) as u32;

    // Calculate the padding (black border) offsets
    let x_offset = (target_width - resized_width) / 2;
    let y_offset = (target_height - resized_height) / 2;

    // Transform from model coordinates to original image coordinates
    // First, subtract the offsets to get coordinates relative to the actual image content
    let x1_model_space = x1 as f32 - x_offset as f32;
    let y1_model_space = y1 as f32 - y_offset as f32;
    let x2_model_space = x2 as f32 - x_offset as f32;
    let y2_model_space = y2 as f32 - y_offset as f32;

    // Then scale back to original image dimensions
    let x1_orig = (x1_model_space / scale).max(0.0) as u32;
    let y1_orig = (y1_model_space / scale).max(0.0) as u32;
    let x2_orig = (x2_model_space / scale).max(0.0) as u32;
    let y2_orig = (y2_model_space / scale).max(0.0) as u32;

    // Clamp to image bounds
    [
        x1_orig.min(img_width - 1),
        y1_orig.min(img_height - 1),
        x2_orig.min(img_width - 1),
        y2_orig.min(img_height - 1),
    ]
}

fn preprocess_like_donut(
    img: &DynamicImage,
    config: &DonutConfig,
    device: &Device,
) -> Result<Tensor> {
    let target_height = config.image_height() as u32;
    let target_width = config.image_width() as u32;

    let resized = img.resize(
        target_width,
        target_height,
        image::imageops::FilterType::Triangle,
    );

    let mut canvas = RgbImage::from_pixel(target_width, target_height, image::Rgb([0, 0, 0]));

    let x_offset = (target_width - resized.width()) / 2;
    let y_offset = (target_height - resized.height()) / 2;

    image::imageops::overlay(
        &mut canvas,
        &resized.to_rgb8(),
        x_offset.into(),
        y_offset.into(),
    );

    let (w, h) = (canvas.width() as usize, canvas.height() as usize);
    let mut normalized = vec![0f32; 3 * w * h];

    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let pixel = canvas.get_pixel(x as u32, y as u32);
                let idx = c * h * w + y * w + x;
                normalized[idx] = (pixel[c] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
    }

    Ok(Tensor::from_vec(normalized, (1, 3, h, w), device)?)
}

pub fn load_hf_tokenizer<P: AsRef<Path>>(model_dir: P) -> anyhow::Result<Tokenizer> {
    let model_dir = model_dir.as_ref();

    let mut tokenizer =
        Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;

    // 1. 加载 special_tokens_map.json
    let special_path = model_dir.join("special_tokens_map.json");
    if special_path.exists() {
        let text = std::fs::read_to_string(&special_path)?;
        let json: Value = serde_json::from_str(&text)?;

        if let Some(map) = json.as_object() {
            for (_k, v) in map {
                if let Some(tok) = v.get("content").and_then(|x| x.as_str()) {
                    tokenizer.add_special_tokens(&[AddedToken::from(tok, true)]);
                }
            }
        }
    }

    // 2. 加载 tokenizer_config.json
    let config_path = model_dir.join("tokenizer_config.json");
    if config_path.exists() {
        let text = std::fs::read_to_string(&config_path)?;
        let json: Value = serde_json::from_str(&text)?;

        if let Some(bos) = json.get("bos_token").and_then(|x| x.as_str()) {
            tokenizer.add_special_tokens(&[AddedToken::from(bos, true)]);
        }

        if let Some(eos) = json.get("eos_token").and_then(|x| x.as_str()) {
            tokenizer.add_special_tokens(&[AddedToken::from(eos, true)]);
        }

        if let Some(pad) = json.get("pad_token").and_then(|x| x.as_str()) {
            tokenizer.add_special_tokens(&[AddedToken::from(pad, true)]);
        }

        if let Some(unk) = json.get("unk_token").and_then(|x| x.as_str()) {
            tokenizer.add_special_tokens(&[AddedToken::from(unk, true)]);
        }
    }

    Ok(tokenizer)
}
