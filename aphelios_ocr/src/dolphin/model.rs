use crate::dolphin::donut::CusDonutModel;
use crate::dolphin::{dolphin_utils, full_in_one, IGNORED_TAGS};
use anyhow::Context;
use anyhow::{Error as E, Result};
use aphelios_core::measure_time;
use aphelios_core::utils::common;
use candle_core::{safetensors, DType, Device, Tensor, D};
use candle_transformers::models::donut::DonutConfig;
use futures_util::{pin_mut, StreamExt};
use glob::glob;
use hf_hub::api::sync::ApiBuilder;
use image::{DynamicImage, GenericImageView, RgbImage};
use itertools::izip;
use ndarray::Array;
use rayon::prelude::*;
use regex::Regex;
use serde_json::Value;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::result::Result::Ok;
use std::sync::Arc;
use tokenizers::AddedToken;
use tokenizers::Tokenizer;
use tracing::{error, info};

const STAGE1_PAGE_BATCH_SIZE: usize = 6;
const STAGE2_CLIP_BATCH_SIZE: usize = 6;
const STAGE2_LOW_WATERMARK: usize = STAGE2_CLIP_BATCH_SIZE;
const STAGE2_HIGH_WATERMARK: usize = STAGE2_CLIP_BATCH_SIZE * 3;

struct PageTask {
    idx: usize,
    image: Arc<DynamicImage>,
}

struct ClipTask {
    page_idx: usize,
    reading_order: usize,
    label: String,
    prompt: String,
    bbox: [u32; 4],
    image: Arc<DynamicImage>,
}

struct PageProgress {
    remaining_clips: usize,
    lines: Vec<Option<String>>,
}

/// CPU-side Donut preprocessing (similar to preprocess_like_donut_sync but standalone)
fn preprocess_like_donut_cpu(
    img: &DynamicImage,
    target_width: u32,
    target_height: u32,
) -> Result<ndarray::Array3<f32>> {
    use ndarray::Array;

    // 1. Resize preserving aspect ratio
    let resized = img.resize(
        target_width,
        target_height,
        image::imageops::FilterType::Triangle,
    );

    // 2. Create black canvas and center
    let mut canvas =
        image::RgbImage::from_pixel(target_width, target_height, image::Rgb([0, 0, 0]));
    let x_offset = (target_width - resized.width()) / 2;
    let y_offset = (target_height - resized.height()) / 2;
    image::imageops::overlay(
        &mut canvas,
        &resized.to_rgb8(),
        x_offset.into(),
        y_offset.into(),
    );

    // 3. Convert to ndarray (H, W, C)
    let rgb = canvas.into_raw(); // Consumes canvas, getting raw bytes
    let h = target_height as usize;
    let w = target_width as usize;

    let array = Array::from_shape_vec((h, w, 3), rgb)
        .map_err(|e| anyhow::anyhow!("Failed to create array: {}", e))?;

    // 4. Normalize: (H, W, C) -> (C, H, W) with Donut normalization
    // Donut uses mean=0.5, std=0.5 for each channel
    let normalized = array
        .mapv(|x| (x as f32 / 127.5) - 1.0)
        .permuted_axes([2, 0, 1]);

    Ok(normalized)
}

pub struct DolphinModel {
    model: CusDonutModel,
    config: DonutConfig,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

impl DolphinModel {
    pub fn load_model(model_id: &str) -> Result<Self> {
        let (device, dtype) = common::get_device_dtype();

        // 1. Determine model file paths
        let (tokenizer_path, safetensors_path, config_path) = if model_id.starts_with('/') {
            info!("Loading model from local path: {}", model_id);
            let model_path = PathBuf::from(model_id);
            (
                model_path.join("tokenizer.json"),
                model_path.join("model.safetensors"),
                model_path.join("config.json"),
            )
        } else {
            info!("Loading model from HuggingFace hub: {}", model_id);
            let hf_api = ApiBuilder::new()
                .with_progress(false)
                // .with_cache_dir(PathBuf::from(".cache"))
                // .with_endpoint("https://hf-mirror.com".to_string())
                .build()
                .unwrap();
            let api = hf_api; //Api::new().context("Failed to create HuggingFace API")?;
            let dolphin_repo = api.repo(hf_hub::Repo::with_revision(
                model_id.to_string(),
                hf_hub::RepoType::Model,
                "main".to_string(),
            ));

            let tokenizer_path = dolphin_repo
                .get("tokenizer.json")
                .context("Failed to load tokenizer.json")?;
            let safetensors_path = dolphin_repo
                .get("model.safetensors")
                .context("Failed to load model.safetensors")?;
            let config_path = dolphin_repo
                .get("config.json")
                .context("Failed to load config.json")?;

            (tokenizer_path, safetensors_path, config_path)
        };

        // 2. Load config and tokenizer
        let config_str = fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config file: {:?}", config_path))?;
        let config: DonutConfig =
            serde_json::from_str(&config_str).context("Failed to parse config.json")?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| E::msg(format!("Failed to load tokenizer: {}", e)))?;

        // 3. Load model weights
        let mut tensors = safetensors::load(&safetensors_path, &device)
            .with_context(|| format!("Failed to load model weights: {:?}", safetensors_path))?;

        // 4. Fix missing lm_head.bias
        let lm_head_weight_key = "decoder.lm_head.weight";
        let lm_head_bias_key = "decoder.lm_head.bias";

        if tensors.contains_key(lm_head_weight_key) && !tensors.contains_key(lm_head_bias_key) {
            info!(
                "Missing lm_head.bias detected, injecting zero-bias patch with dtype {:?}...",
                dtype
            );
            let weight = &tensors[lm_head_weight_key];
            let out_dim = weight.dim(0)?;
            // Bias belongs to decoder, so it should use the target fast dtype
            let fake_bias = Tensor::zeros(out_dim, dtype, &device)?;
            tensors.insert(lm_head_bias_key.to_string(), fake_bias);
        }

        // 5. Initialize model
        let model = CusDonutModel::load(&config, tensors, dtype, &device)?;

        Ok(Self {
            model,
            config,
            tokenizer,
            device,
            dtype,
        })
    }

    pub async fn dolphin_ocr(&mut self, image_path: &str, output_dir: &str) -> Result<Vec<String>> {
        let mut res_li = vec![];
        let input_file_path = Path::new(image_path);
        let output_path = Path::new(output_dir).join(input_file_path.file_stem().unwrap());
        let _ = std::fs::create_dir_all(&output_path);
        let ext = Path::new(image_path)
            .extension()
            .unwrap()
            .to_str()
            .unwrap()
            .to_lowercase();
        if ext == "pdf" {
            let existing_pages = Self::scan_existing_pages(&output_path);
            self.dolphin_ocr_pdf(image_path, &output_path, existing_pages, &mut res_li)
                .await?;
        } else {
            let img = image::ImageReader::open(image_path)
                .with_context(|| format!("Failed to open image file {}", image_path))?
                .decode()
                .with_context(|| format!("Failed to decode image {}", image_path))?;

            if let Err(e) = self.run_ocr_single_img(&img, &output_path, 0, &mut res_li) {
                error!("OCR failed for image {}: {:?}", image_path, e);
            }
        }
        full_in_one(output_path.to_str().unwrap())?;
        Ok(res_li)
    }

    /// 扫描输出目录，返回已存在的页面编号集合
    fn scan_existing_pages(output_path: &Path) -> std::collections::HashSet<usize> {
        let mut existing = std::collections::HashSet::new();
        let glob_pattern = format!("{}/[0-9]*_page.txt", output_path.to_str().unwrap());

        if let Ok(entries) = glob(&glob_pattern) {
            for entry in entries.flatten() {
                if let Some(filename) = entry.file_name().and_then(|n| n.to_str()) {
                    // 提取页面编号：格式为 "123_page.txt"
                    if let Some(num_str) = filename.strip_suffix("_page.txt") {
                        if let Ok(num) = num_str.parse::<usize>() {
                            existing.insert(num);
                        }
                    }
                }
            }
        }
        existing
    }

    async fn dolphin_ocr_pdf(
        &mut self,
        image_path: &str,
        output_path: &PathBuf,
        existing_pages: HashSet<usize>,
        res_li: &mut Vec<String>,
    ) -> Result<()> {
        info!(
            "Found {} existing pages, stage1_batch={}, stage2_batch={}, high_watermark={}",
            existing_pages.len(),
            STAGE1_PAGE_BATCH_SIZE,
            STAGE2_CLIP_BATCH_SIZE,
            STAGE2_HIGH_WATERMARK
        );

        let copped_img_path = output_path.join("copped_img");
        let _ = fs::create_dir_all(&copped_img_path);

        let img_iter = measure_time!(
            "load images from pdf",
            dolphin_utils::load_pdf_images(Path::new(image_path).to_path_buf()).enumerate()
        );
        pin_mut!(img_iter);

        let mut pending_pages = VecDeque::new();
        let mut pending_clips = VecDeque::new();
        let mut page_progress = BTreeMap::new();
        let mut pdf_done = false;

        loop {
            if pending_clips.len() >= STAGE2_CLIP_BATCH_SIZE
                || (pdf_done && pending_pages.is_empty() && !pending_clips.is_empty())
            {
                let batch_size = pending_clips.len().min(STAGE2_CLIP_BATCH_SIZE);
                self.run_stage2_clip_batch(
                    &mut pending_clips,
                    batch_size,
                    &mut page_progress,
                    output_path,
                    res_li,
                )?;
                continue;
            }

            while !pdf_done
                && pending_pages.len() < STAGE1_PAGE_BATCH_SIZE
                && pending_clips.len() < STAGE2_HIGH_WATERMARK
            {
                match img_iter.next().await {
                    Some((idx, img_result)) => {
                        if existing_pages.contains(&idx) {
                            info!("page [{}] already processed, skip", idx);
                            continue;
                        }

                        let img = img_result
                            .with_context(|| format!("Failed to load PDF page {}", idx))?;
                        pending_pages.push_back(PageTask {
                            idx,
                            image: Arc::new(img),
                        });
                    }
                    None => {
                        pdf_done = true;
                        break;
                    }
                }
            }

            if pending_clips.len() >= STAGE2_LOW_WATERMARK {
                continue;
            }

            if !pending_pages.is_empty() {
                let batch_size = pending_pages.len().min(STAGE1_PAGE_BATCH_SIZE);
                self.run_stage1_page_batch(
                    &mut pending_pages,
                    batch_size,
                    &mut pending_clips,
                    &mut page_progress,
                    &copped_img_path,
                    output_path,
                )?;
                continue;
            }

            if pdf_done {
                if pending_clips.is_empty() {
                    break;
                }
                let batch_size = pending_clips.len().min(STAGE2_CLIP_BATCH_SIZE);
                self.run_stage2_clip_batch(
                    &mut pending_clips,
                    batch_size,
                    &mut page_progress,
                    output_path,
                    res_li,
                )?;
                continue;
            }
        }

        Ok(())
    }

    fn run_stage1_page_batch(
        &mut self,
        pending_pages: &mut VecDeque<PageTask>,
        batch_size: usize,
        pending_clips: &mut VecDeque<ClipTask>,
        page_progress: &mut BTreeMap<usize, PageProgress>,
        copped_img_path: &PathBuf,
        output_path: &PathBuf,
    ) -> Result<()> {
        let batch: Vec<PageTask> = pending_pages.drain(..batch_size).collect();
        if batch.is_empty() {
            return Ok(());
        }

        let page_indexes: Vec<usize> = batch.iter().map(|page| page.idx).collect();
        info!("stage1 batch pages {:?}", page_indexes);

        let layouts = measure_time!({ self.run_ocr_first_stage_batch(&batch)? });

        #[cfg(feature = "profiling")]
        for (page_task, layout_str) in batch.iter().zip(layouts.iter()) {
            info!("page {}, layout str = {}", page_task.idx, layout_str);
        }

        for (page, layout_str) in batch.into_iter().zip(layouts.into_iter()) {
            let output_file = output_path.join(format!("{}_page.txt", page.idx));
            if fs::exists(&output_file)? {
                info!("{} exist, skip stage2 scheduling", output_file.display());
                continue;
            }
            let page_clips =
                self.build_clip_tasks_for_page(page.idx, page.image, &layout_str, copped_img_path)?;
            let clip_count = page_clips.len();
            if clip_count == 0 {
                fs::write(&output_file, "")?;
                continue;
            }

            page_progress.insert(
                page.idx,
                PageProgress {
                    remaining_clips: clip_count,
                    lines: vec![None; clip_count],
                },
            );
            pending_clips.extend(page_clips);
        }

        Ok(())
    }

    fn run_stage2_clip_batch(
        &mut self,
        pending_clips: &mut VecDeque<ClipTask>,
        batch_size: usize,
        page_progress: &mut BTreeMap<usize, PageProgress>,
        output_path: &PathBuf,
        res_li: &mut Vec<String>,
    ) -> Result<()> {
        let batch: Vec<ClipTask> = pending_clips.drain(..batch_size).collect();
        if batch.is_empty() {
            return Ok(());
        }

        let clip_keys: Vec<(usize, usize)> = batch
            .iter()
            .map(|clip| (clip.page_idx, clip.reading_order))
            .collect();
        info!("stage2 batch clips {:?}", clip_keys);

        let target_width = self.config.image_width() as u32;
        let target_height = self.config.image_height() as u32;
        let processed: Vec<Result<(usize, usize, String, ndarray::Array3<f32>)>> = batch
            .par_iter()
            .map(|clip| {
                let cropped_img = dolphin_utils::crop_image(&clip.image, clip.bbox, 5);
                let pixel_values =
                    preprocess_like_donut_cpu(&cropped_img, target_width, target_height)
                        .map_err(|e| anyhow::anyhow!("Preprocess failed: {}", e))?;
                Ok((
                    clip.page_idx,
                    clip.reading_order,
                    format!("[{}] - [{}]", clip.reading_order, clip.label),
                    pixel_values,
                ))
            })
            .collect();
        let processed = processed.into_iter().collect::<Result<Vec<_>>>()?;

        let mut labels = Vec::with_capacity(processed.len());
        let mut pixel_values_li = Vec::with_capacity(processed.len());
        let prompts: Vec<&str> = batch.iter().map(|clip| clip.prompt.as_str()).collect();
        for (_, _, label, pixel_values) in processed {
            labels.push(label);
            let shape = pixel_values.shape();
            let tensor = Tensor::from_vec(
                pixel_values.iter().cloned().collect(),
                (shape[0], shape[1], shape[2]),
                &Device::Cpu,
            )?
            .to_dtype(self.dtype)?
            .unsqueeze(0)?
            .to_device(&self.device)?;
            pixel_values_li.push(tensor);
        }

        let results = measure_time!(
            "second stage",
            self.generate_text_batch(&pixel_values_li, &prompts, pixel_values_li.len())?
        );

        let mut completed_pages = Vec::new();
        for (clip, label, ocr_content) in
            izip!(batch.into_iter(), labels.into_iter(), results.into_iter())
        {
            if let Some(progress) = page_progress.get_mut(&clip.page_idx) {
                progress.lines[clip.reading_order] = Some(format!("{} : {}", label, ocr_content));
                progress.remaining_clips -= 1;
                if progress.remaining_clips == 0 {
                    completed_pages.push(clip.page_idx);
                }
            }
        }

        for page_idx in completed_pages {
            if let Some(progress) = page_progress.remove(&page_idx) {
                let page_lines: Vec<String> = progress.lines.into_iter().flatten().collect();
                let output_file = output_path.join(format!("{}_page.txt", page_idx));
                fs::write(&output_file, page_lines.join("\n"))?;
                res_li.extend(page_lines);
            }
        }

        Ok(())
    }

    fn run_ocr_single_img(
        &mut self,
        img: &DynamicImage,
        output_path: &PathBuf,
        idx: usize,
        res_li: &mut Vec<String>,
    ) -> Result<()> {
        let copped_img_path = PathBuf::from(output_path).join("copped_img");
        let _ = fs::create_dir_all(&copped_img_path);

        let output_file: PathBuf = output_path.join(format!("{}_page.txt", idx));

        if fs::exists(&output_file)? {
            info!("{} exist, to process next ", &output_file.to_str().unwrap());
            return Ok(());
        }
        info!("start process page [{}] first stage ...", idx);
        let layout_str = self.run_ocr_first_stage(img, idx)?;

        info!("start process page [{}] second stage ...", idx);
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

    fn run_ocr_first_stage(&mut self, img: &DynamicImage, _idx: usize) -> Result<String> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("run_ocr_first_stage", index = _idx).entered();

        let img_tensor = dolphin_utils::get_tensor_from_image(
            &img,
            self.config.image_height() as u32,
            self.config.image_width() as u32,
            &self.device,
            self.dtype,
        );
        let task_prompt = "<s>Parse the reading order of this document. <Answer/>";
        let mut layout_str = self.generate_text(&img_tensor, &task_prompt)?;

        layout_str = layout_str.replace(task_prompt, "").replace("</s>", "");
        Ok(layout_str)
    }

    fn run_ocr_first_stage_batch(&mut self, pages: &[PageTask]) -> Result<Vec<String>> {
        if pages.is_empty() {
            return Ok(Vec::new());
        }

        let prompt = "<s>Parse the reading order of this document. <Answer/>";
        let prompts = vec![prompt; pages.len()];
        let pixel_values_li: Vec<Tensor> = pages
            .iter()
            .map(|page| {
                dolphin_utils::get_tensor_from_image(
                    &page.image,
                    self.config.image_height() as u32,
                    self.config.image_width() as u32,
                    &self.device,
                    self.dtype,
                )
            })
            .collect();

        let mut layouts = self.generate_text_batch(&pixel_values_li, &prompts, pages.len())?;
        for layout in &mut layouts {
            *layout = layout.replace(prompt, "").replace("</s>", "");
        }
        Ok(layouts)
    }

    fn build_clip_tasks_for_page(
        &self,
        page_idx: usize,
        image: Arc<DynamicImage>,
        layout_str: &str,
        copped_img_path: &PathBuf,
    ) -> Result<Vec<ClipTask>> {
        let (img_w, img_h) = image.dimensions();
        let target_width = self.config.image_width() as u32;
        let target_height = self.config.image_height() as u32;
        let re = Regex::new(r"\[(\d+,\d+,\d+,\d+)\]\[([^\]]+)\]")?;

        let mut bbox_list = Vec::new();
        let mut clips = Vec::new();

        for cap in re.captures_iter(layout_str) {
            let label = cap[2].to_string();
            if IGNORED_TAGS.contains(&label.as_str()) {
                continue;
            }
            let coords_raw: Vec<i32> = cap[1].split(',').map(|s| s.parse().unwrap()).collect();
            let bbox = dolphin_utils::transform_to_pixel_dynamic(
                &[coords_raw[0], coords_raw[1], coords_raw[2], coords_raw[3]],
                img_w,
                img_h,
                target_width,
                target_height,
            );
            bbox_list.push(bbox);

            let prompt = match label.as_str() {
                "tab" => "<s>Parse the table in the image. <Answer/>",
                "equ" => "<s>Read formula in the image. <Answer/>",
                _ => "<s>Read text in the image. <Answer/>",
            };

            clips.push(ClipTask {
                page_idx,
                reading_order: clips.len(),
                label,
                prompt: prompt.to_string(),
                bbox,
                image: Arc::clone(&image),
            });
        }

        if !bbox_list.is_empty() {
            let layout_draw_path = copped_img_path.join(format!("{}-layout.png", page_idx));
            dolphin_utils::draw_bbox_and_save_multi(&image, &bbox_list, 5, &layout_draw_path);
        }

        Ok(clips)
    }

    fn run_ocr_second_stage(
        &mut self,
        full_image: &DynamicImage,
        layout_str: &str,
        i_index: usize,
        copped_img_path: &PathBuf,
    ) -> Result<Vec<String>> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("run_ocr_second_stage", index = i_index).entered();

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
        let mut tasks = Vec::new();
        for cap in re.captures_iter(layout_str) {
            let label = cap[2].to_string();
            if IGNORED_TAGS.contains(&label.as_str()) {
                info!("skip label {}", label);
                continue;
            }
            let coords_raw: Vec<i32> = cap[1].split(',').map(|s| s.parse().unwrap()).collect();
            tasks.push((reading_order, coords_raw, label));
            reading_order += 1;
        }

        info!("Pre-processing {} image crops in parallel...", tasks.len());

        #[cfg(feature = "profiling")]
        let _preprocess_tasks_span = tracing::info_span!("preprocess_tasks").entered();
        let dtype = self.dtype;
        let processed_tasks: Vec<_> = tasks
            .into_par_iter()
            .map(|(order, coords_raw, label)| {
                let bbox = dolphin_utils::transform_to_pixel_dynamic(
                    &[coords_raw[0], coords_raw[1], coords_raw[2], coords_raw[3]],
                    img_w,
                    img_h,
                    target_width,
                    target_height,
                );

                let cropped_img = dolphin_utils::crop_image(full_image, bbox, 5);
                // 这里我们需要在每个线程中进行预处理，但 Tensor 的创建需要 Device。
                // 注意：candle Tensor::from_vec 在 CPU 上是并行的，但如果是 Metal Device，则需要小心。
                // 这里的 preprocess_like_donut 内部调用了 Tensor::from_vec。
                let pixel_values = self
                    .preprocess_like_donut_sync(&cropped_img, dtype)
                    .unwrap();

                let prompt = match label.as_str() {
                    "tab" => "<s>Parse the table in the image. <Answer/>",
                    "equ" => "<s>Read formula in the image. <Answer/>",
                    _ => "<s>Read text in the image. <Answer/>",
                };

                (order, bbox, pixel_values, prompt, label)
            })
            .collect();
        #[cfg(feature = "profiling")]
        _preprocess_tasks_span.exit();

        let mut reading_orders = Vec::new();
        for (order, bbox, pixel_values, prompt, label) in processed_tasks {
            bbox_list.push(bbox);
            reading_orders.push(order);
            labels.push(format!("[{}] - [{}]", order, label));
            // 将 Tensor 从 CPU 移动到模型设备 (如 Metal)
            pixel_values_li.push(pixel_values.to_device(&self.device)?);
            prompt_li.push(prompt);
        }

        #[cfg(feature = "profiling")]
        let _draw_bbox_and_save_multi_span =
            tracing::info_span!("draw_bbox_and_save_multi").entered();
        dolphin_utils::draw_bbox_and_save_multi(full_image, &bbox_list, 5, &layout_draw_path);
        #[cfg(feature = "profiling")]
        _draw_bbox_and_save_multi_span.exit();

        info!("start seconde stage with batch inference...");

        #[cfg(feature = "profiling")]
        let _generate_text_batch_span = tracing::info_span!("generate_text_batch").entered();
        let res = self.generate_text_batch(&pixel_values_li, &prompt_li, 4)?;
        #[cfg(feature = "profiling")]
        _generate_text_batch_span.exit();

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

        let decoded = measure_time!("模型推理中...", {
            let tokens = self.tokenizer.encode(prompt, false).map_err(E::msg)?;
            let mut token_ids = tokens.get_ids().to_vec(); // 编码图像
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
                        error!("Candle Error detected: {:?}", e);
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
            let decoded = self
                .tokenizer
                .decode(&token_ids, false)
                .map_err(E::msg)?
                .replace(prompt, "")
                .replace("</s>", "")
                .replace("\n", "");
            decoded
        });

        Ok(decoded)
    }

    fn preprocess_like_donut(&mut self, img: &DynamicImage) -> Result<Tensor> {
        let dtype = self.dtype;
        let tensor = self.preprocess_like_donut_sync(img, dtype)?;
        Ok(tensor.to_device(&self.device)?)
    }

    fn preprocess_like_donut_sync(&self, img: &DynamicImage, dtype: DType) -> Result<Tensor> {
        let target_height = self.config.image_height() as u32;
        let target_width = self.config.image_width() as u32;

        // 1. Resize 保持比例
        let resized = img.resize(
            target_width,
            target_height,
            image::imageops::FilterType::Triangle,
        );

        // 2. 创建黑色画布并居中叠加
        let mut canvas = RgbImage::from_pixel(target_width, target_height, image::Rgb([0, 0, 0]));
        let x_offset = (target_width - resized.width()) / 2;
        let y_offset = (target_height - resized.height()) / 2;
        image::imageops::overlay(
            &mut canvas,
            &resized.to_rgb8(),
            x_offset.into(),
            y_offset.into(),
        );

        // 3. 将 RgbImage 转换为 ndarray (H, W, C)
        let h = target_height as usize;
        let w = target_width as usize;

        let array = Array::from_shape_vec((h, w, 3), canvas.into_raw())
            .map_err(|e| anyhow::anyhow!("Failed to create array: {}", e))?;

        // 4. 规范化并转换维度 (H, W, C) -> (C, H, W)
        let normalized = array
            .mapv(|x| (x as f32 / 127.5) - 1.0)
            .permuted_axes([2, 0, 1]); // 移动 Channel 到第一维

        // 5. 转换为 Candle Tensor (始终先在 CPU 上创建，以便并行)
        let tensor = Tensor::from_vec(
            normalized.iter().cloned().collect(),
            (3, h, w),
            &Device::Cpu,
        )?
        .to_dtype(dtype)?
        .unsqueeze(0)?;

        Ok(tensor)
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
