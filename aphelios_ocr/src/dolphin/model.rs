use crate::dolphin::dolphin_utils;
use crate::dolphin::donut::CusDonutModel;
use anyhow::Context;
use anyhow::{Error as E, Result};
use aphelios_core::common::core_utils;
use aphelios_core::measure_time;
use candle_core::{D, DType, Device, Tensor, safetensors};
use candle_nn::VarBuilder;
use candle_transformers::models::donut::DonutConfig;
use futures_util::{StreamExt, pin_mut};
use glob::glob;
use image::{DynamicImage, GenericImageView, RgbImage};
use itertools::izip;
use regex::Regex;
use serde_json::Value;
use std::path::Path;
use std::path::PathBuf;
use std::result::Result::Ok;
use std::{env, fs};
use tokenizers::AddedToken;
use tokenizers::Tokenizer;
use tracing::{error, info};

pub struct DolphinModel {
    model: CusDonutModel,
    config: DonutConfig,
    tokenizer: Tokenizer,
    device: Device,
}

impl DolphinModel {
    pub fn load_model(model_id: &str) -> Result<Self> {
        let model = (|| -> Result<_> {
            let model_path = PathBuf::from(model_id);
            let device = Device::new_metal(0).unwrap_or(Device::Cpu);
            let config_str = std::fs::read_to_string(model_path.join("config.json"))?;
            let config: DonutConfig = serde_json::from_str(&config_str)
                .with_context(|| format!("config.json文件解析失败"))?;
            let tokenizer =
                Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(E::msg)?;
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
            Ok(Self {
                model,
                config,
                tokenizer,
                device,
            })
        })()
        .with_context(|| "模型加载失败")?;
        Ok(model)
    }

    pub async fn dolphin_ocr(&mut self, image_path: &str, output_dir: &str) -> Result<Vec<String>> {
        core_utils::init_tracing();

        let output_path = Path::new(output_dir);
        let _ = std::fs::create_dir_all(output_path);
        let mut res_li = vec![];
        let ext = image_path.rsplit('.').next().unwrap_or("").to_lowercase();
        if ext == "pdf" {
            let img_iter =
                dolphin_utils::load_pdf_images(Path::new(image_path).to_path_buf()).enumerate();
            pin_mut!(img_iter);

            while let Some((idx, img_result)) = img_iter.next().await {
                match img_result {
                    Ok(img) => {
                        if let Err(e) = self.run_ocr_single_img(&img, output_path, idx, &mut res_li)
                        {
                            error!("OCR failed for PDF page {}: {:?}", idx, e);
                        }
                    }
                    Err(e) => {
                        error!("Failed to load PDF page {}: {:?}", idx, e);
                    }
                }
            }
        } else {
            let img = image::ImageReader::open(image_path)
                .with_context(|| format!("Failed to open image file {}", image_path))?
                .decode()
                .with_context(|| format!("Failed to decode image {}", image_path))?;

            if let Err(e) = self.run_ocr_single_img(&img, output_path, 0, &mut res_li) {
                error!("OCR failed for image {}: {:?}", image_path, e);
            }
        }
        Ok(res_li)
    }

    fn run_ocr_single_img(
        &mut self,
        img: &DynamicImage,
        output_path: &Path,
        idx: usize,
        res_li: &mut Vec<String>,
    ) -> Result<()> {
        let copped_img_path = PathBuf::from(output_path).join("copped_img");
        let _ = fs::create_dir_all(&copped_img_path);

        let output_file: PathBuf = output_path.join(format!("{}_page.txt", idx));

        if fs::exists(&output_file)? {
            info!("{} exist, to process next ", &output_file.to_str().unwrap());
        }
        info!("start process page [{}] ...", idx);
        let layout_str = self.run_ocr_first_stage(img)?;

        info!("end first stage layout_str {}", layout_str);
        let res = self.run_ocr_second_stage(&img, &layout_str, idx, &copped_img_path);
        match res {
            Ok(mut ok_vec) => {
                let _ = fs::write(&output_file, ok_vec.join("\n"));
                res_li.append(&mut ok_vec);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn run_ocr_first_stage(&mut self, img: &DynamicImage) -> Result<String> {
        let img_tensor = dolphin_utils::get_tensor_from_image(
            &img,
            self.config.image_height() as u32,
            self.config.image_width() as u32,
            &self.device,
        );
        let task_prompt = "<s>Parse the reading order of this document. <Answer/>";
        let mut layout_str = self.generate_text(&img_tensor, &task_prompt)?;

        layout_str = layout_str.replace(task_prompt, "").replace("</s>", "");
        Ok(layout_str)
    }

    fn run_ocr_second_stage(
        &mut self,
        full_image: &DynamicImage,
        layout_str: &str,
        i_index: usize,
        copped_img_path: &PathBuf,
    ) -> Result<Vec<String>> {
        let (img_w, img_h) = full_image.dimensions();
        let target_width = self.config.image_width() as u32;
        let target_height = self.config.image_height() as u32;
        let re = Regex::new(r"\[(\d+,\d+,\d+,\d+)\]\[([^\]]+)\]")?;

        let layout_draw_path = copped_img_path.join(format!("{}-layout.png", i_index));
        let mut bbox_list: Vec<[u32; 4]> = Vec::new();

        let mut pixel_values_li = Vec::new();
        let mut prompt_li = Vec::new();

        let mut labels = Vec::new();
        let mut reading_order = 0;
        let mut reading_orders = Vec::new();
        for cap in re.captures_iter(layout_str) {
            let curr_reading_order = reading_order;
            reading_order += 1;
            let coords_raw: Vec<i32> = cap[1].split(',').map(|s| s.parse().unwrap()).collect();
            let label = &cap[2];

            if label == "fig" {
                continue;
            }

            let bbox = dolphin_utils::transform_to_pixel_dynamic(
                [coords_raw[0], coords_raw[1], coords_raw[2], coords_raw[3]],
                img_w,
                img_h,
                target_width,
                target_height,
            );

            bbox_list.push(bbox);
            let cropped_img = dolphin_utils::crop_image(full_image, bbox, 5);
            let pixel_values = self.preprocess_like_donut(&cropped_img)?;

            let prompt = match label {
                "tab" => "<s>Parse the table in the image. <Answer/>",
                "equ" => "<s>Read formula in the image. <Answer/>",
                _ => "<s>Read text in the image. <Answer/>",
            };
            reading_orders.push(curr_reading_order);
            labels.push(format!("[{}] - [{}]", curr_reading_order, label));
            pixel_values_li.push(pixel_values);
            prompt_li.push(prompt);
        }
        dolphin_utils::draw_bbox_and_save_multi(full_image, &bbox_list, 5, &layout_draw_path);

        info!("start seconde stage ...");
        let res = self.generate_text_batch(&pixel_values_li, &prompt_li, 1)?;

        let mut full_res = Vec::new();
        for (_, la, ocr_content) in izip!(&reading_orders, labels, res) {
            full_res.push(format!("{} : {}", la, ocr_content));
        }
        Ok(full_res)
    }

    fn generate_text_batch(
        &mut self,
        pixel_values_list: &[Tensor], // 每个 [1,C,H,W] 或 [C,H,W]
        prompts: &[&str],
        batch_size: usize,
    ) -> Result<Vec<String>> {
        if pixel_values_list.len() != prompts.len() {
            return Err(E::msg("pixel_values_list and prompts length mismatch"));
        }

        let mut outputs = Vec::with_capacity(prompts.len());

        for (batch_start, batch_end) in (0..prompts.len())
            .step_by(batch_size)
            .map(|s| (s, usize::min(s + batch_size, prompts.len())))
        {
            self.model.clean_kv();

            let batch_prompts = &prompts[batch_start..batch_end];
            let batch_pixels = &pixel_values_list[batch_start..batch_end];
            let bsz = batch_prompts.len();

            // ------------------------------------------------------------
            // 1️⃣ 拼接图片 -> [B,C,H,W]
            // ------------------------------------------------------------
            let pixel_batch = Tensor::cat(batch_pixels, 0)?;
            let encoder_output = self.model.encode(&pixel_batch)?;

            // ------------------------------------------------------------
            // 2️⃣ tokenize + padding
            // ------------------------------------------------------------
            let mut token_ids_batch: Vec<Vec<u32>> = Vec::with_capacity(bsz);

            for prompt in batch_prompts {
                let tokens = self.tokenizer.encode(*prompt, false).map_err(E::msg)?;
                token_ids_batch.push(tokens.get_ids().to_vec());
            }

            let max_prompt_len = token_ids_batch.iter().map(|v| v.len()).max().unwrap();

            // padding
            for ids in &mut token_ids_batch {
                while ids.len() < max_prompt_len {
                    ids.push(self.config.decoder.pad_token_id);
                }
            }

            // flatten 成连续数组
            let mut flat: Vec<u32> = Vec::with_capacity(bsz * max_prompt_len);
            for ids in &token_ids_batch {
                flat.extend(ids);
            }

            let input_tensor = Tensor::new(flat, &self.device)?.reshape((bsz, max_prompt_len))?;

            // ------------------------------------------------------------
            // 3️⃣ 一次性初始化 KV cache
            // ------------------------------------------------------------
            self.model.decode(&input_tensor, &encoder_output, 0)?;

            let mut finished = vec![false; bsz];
            // ------------------------------------------------------------
            // 4️⃣ 自回归生成
            // ------------------------------------------------------------
            for step in 0..2048 {
                // 取每个序列最后一个 token
                let mut last_tokens = Vec::with_capacity(bsz);
                for ids in &token_ids_batch {
                    last_tokens.push(*ids.last().unwrap());
                }

                let input_tensor = Tensor::new(last_tokens, &self.device)?.unsqueeze(1)?; // [B,1]

                let logits =
                    self.model
                        .decode(&input_tensor, &encoder_output, max_prompt_len + step - 1)?;

                // logits: [B,1,V]
                let logits = logits.squeeze(1)?; // [B,V]

                let next_tokens = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;

                for i in 0..bsz {
                    if finished[i] {
                        continue;
                    }

                    let next = next_tokens[i];
                    token_ids_batch[i].push(next);

                    if next == self.config.decoder.eos_token_id {
                        finished[i] = true;
                    }
                }

                if finished.iter().all(|&f| f) {
                    break;
                }
            }

            // ------------------------------------------------------------
            // 5️⃣ decode 输出
            // ------------------------------------------------------------
            for (i, ids) in token_ids_batch.iter().enumerate() {
                let mut decoded = self.tokenizer.decode(ids, false).map_err(E::msg)?;

                decoded = decoded
                    .replace(batch_prompts[i], "")
                    .replace("</s>", "")
                    .replace("\n", "");

                outputs.push(decoded);
            }
        }

        Ok(outputs)
    }

    fn generate_text(&mut self, pixel_values: &Tensor, prompt: &str) -> Result<String> {
        self.model.clean_kv();

        let decoded = measure_time! {
            let tokens = self.tokenizer.encode(prompt, false).map_err(E::msg)?;
            let mut token_ids = tokens.get_ids().to_vec();            // 编码图像
            let encoder_output = self.model.encode(&pixel_values)?;

            // 解码循环 (带 KV Cache 优化)
            for i in 0..2048 {
                let last_token_only = i > 0;
                let decoder_input = if last_token_only {
                    &token_ids[token_ids.len() - 1..]
                } else {
                    &token_ids[..]
                };

                let input_tensor = Tensor::new(decoder_input, &self.device)?.unsqueeze(0)?;
                let logits = match self.model.decode(
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
                if next_token == self.config.decoder.eos_token_id {
                    break;
                }
            }

            // 8. 输出结果
            let decoded = self.tokenizer
                .decode(&token_ids, false)
                .map_err(E::msg)?
                .replace(prompt, "")
                .replace("</s>", "")
                .replace("\n", "");
            decoded
        };

        Ok(decoded)
    }

    fn preprocess_like_donut(&mut self, img: &DynamicImage) -> Result<Tensor> {
        let target_height = self.config.image_height() as u32;
        let target_width = self.config.image_width() as u32;

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

        Ok(Tensor::from_vec(normalized, (1, 3, h, w), &self.device)?)
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
}

pub async fn run_ocr(pdf_path: &str, output_path: &str) -> Result<()> {
    info!("start run dolphin ocr task");

    // Allow model path to be configurable via environment variable or use default
    let model_id = env::var("DOLPHIN_MODEL_PATH")
        .unwrap_or_else(|_| "/Volumes/sw/pretrained_models/Dolphin-v1.5".to_string());

    let mut dm = DolphinModel::load_model(&model_id)?;

    let _ = &dm
        .dolphin_ocr(&pdf_path.to_string(), &output_path.to_string())
        .await?;
    info!("ocr finished");

    let page_datas = get_page_datas(output_path)?;
    info!("start merge all file to single one");
    let output = Path::new(output_path);
    let output_file: PathBuf = output.join("total_in_one.txt");
    fs::write(&output_file, format!("{}", &page_datas.join("\n")))?;
    Ok(())
}

fn get_page_datas(output_path: &str) -> Result<Vec<String>> {
    let mut page_datas: Vec<String> = Vec::new();
    let re =
        Regex::new(r"^\[(?P<id>\d+)\]\s*-\s*\[(?P<tag>[^\]]+)\]\s*:\s*(?P<content>.*)$").unwrap();

    let mut last_label = String::new();

    for entry in glob(&format!("{}/[0-9]*_page.txt", output_path))? {
        let path = entry?;
        // 读取文件内容
        let content = fs::read_to_string(&path)?;
        let mut i_index = 0;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Some(caps) = re.captures(line) {
                let mut item_tag = caps["tag"].to_string();
                let item_content = caps["content"].trim().to_string();

                // 过滤掉不需要的标签
                if ["fig", "tab", "equ", "code", "header", "foot", "fnote"]
                    .contains(&item_tag.as_str())
                {
                    continue;
                }
                // half_para 统一为 para
                if item_tag == "half_para" {
                    item_tag = "para".to_string();
                }

                // 如果同标签且是本文件第一条，追加到上一条
                if item_tag == last_label && i_index == 0 {
                    if let Some(last) = page_datas.last_mut() {
                        last.push_str(&item_content);
                    }
                } else {
                    page_datas.push(item_content);
                }

                last_label = item_tag;
                i_index += 1;
            }
        }
    }

    Ok(page_datas)
}
