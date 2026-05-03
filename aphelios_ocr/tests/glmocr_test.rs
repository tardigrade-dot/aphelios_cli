use anyhow::{Result};
use aphelios_core::utils::common::get_device;
use aphelios_core::{init_logging, measure_time};
use aphelios_ocr::doc_layout::{ImageInfo, run_pp_layout};
use aphelios_ocr::dolphin::dolphin_utils::{self, group_clips_by_patches, load_pdf_images};
use aphelios_ocr::dolphin::model::DolphinModel;
use aphelios_ocr::glmocr::layout::LayoutDetector;
use aphelios_ocr::glmocr::{ClipInfo, GlmOcr, IMAGE_LABELS, prompt_for_label};
use futures_util::{pin_mut, StreamExt};
use image::{RgbImage};
use tracing::info;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

const DOCLAYOUT_MODEL_FILE_PATH: &str = "/Volumes/sw/onnx_models/PP-DocLayout/PP-DocLayout-M.onnx";
const TEST_IMG: &str = "/Volumes/sw/MyDrive/data_src/page_zht_49.png";

#[test]
fn test_features() {
    init_logging();
    info!("end");
}

//cargo test --package aphelios_ocr --test glmocr_test --features metal --features profiling -- test_1
#[test]
fn test_single_ocr() -> Result<()>{

    init_logging();
    let img_info = ImageInfo::new(TEST_IMG)?;
    let layout_detections = run_pp_layout(DOCLAYOUT_MODEL_FILE_PATH, &img_info, None::<String>);

    let model_id = Some("/Volumes/sw/pretrained_models/GLM-OCR");
    let quantize = None; //Some("q8_0");
    let ocr = GlmOcr::new_with_device(model_id, quantize, get_device())?;

    for layout_det in layout_detections?{
        if !IMAGE_LABELS.contains(&layout_det.label) {
            let text = ocr.recognize_by_layout(&img_info.image, &layout_det, 512);
            info!("class_id {}, score {}, , label {}, text {}", layout_det.class_id, layout_det.score,layout_det.label, text?.text)
        }
    }
    Ok(())
}

// samply record cargo test --package aphelios_ocr --test glmocr_test --features metal --features profiling -- test_glmocr_single_img
#[test]
fn test_glmocr_single_img() -> Result<()> {
    init_logging();

    let img_path = "/Users/larry/coderesp/aphelios_cli/test_data/page_32.png";
    // Load image
    let image = image::open(img_path)?;

    // Select device
    let device = get_device();

    let layout = true;
    let json = false;
    let max_tokens = 2048;
    let prompt = "Text Recognition:".to_string();

    // Initialize OCR model
    let start = Instant::now();
    let model_id = Some("/Volumes/sw/pretrained_models/GLM-OCR");
    let quantize = None; //Some("q8_0");
    let ocr = GlmOcr::new_with_device(model_id, quantize, device)?;
    let load_time = start.elapsed();
    eprintln!("OCR model loaded in {load_time:.2?}");

    // Run OCR
    let start = Instant::now();
    if layout {
        eprintln!("Loading layout detection model...");
        let layout_start = Instant::now();
        let mut layout = LayoutDetector::new(DOCLAYOUT_MODEL_FILE_PATH)?;
        eprintln!("Layout model loaded in {:.2?}", layout_start.elapsed());

        // Draw layout detections on the image for visualization
        let layout_image_path = Path::new(img_path).with_file_name("page_zht_2_layout.png");
        let rgb = image.to_rgb8();
        let detections = layout.detect(&rgb)?;
        dolphin_utils::draw_layout_detections(&image, &detections, 4, &layout_image_path);
        eprintln!("Layout visualization saved to {:?}", layout_image_path);

        if json {
            let doc = ocr.recognize_layout_structured(&image, &mut layout, max_tokens)?;
            let json = doc.to_json()?;
            println!("{json}");
        } else {
            let result = ocr.recognize_with_layout(&image, &mut layout, max_tokens)?;
            // let result =
            //     ocr.recognize_with_layout_batched(&image, &mut layout, max_tokens, Some(10))?;
            println!("{result}");
        }
    } else {
        if json {
            anyhow::bail!("--json requires --layout");
        }
        let result = ocr.recognize_with_max_tokens(&image, &prompt, max_tokens)?;
        println!("{result}");
    };
    let inference_time = start.elapsed();
    eprintln!("Inference completed in {inference_time:.2?}");

    Ok(())
}

#[tokio::test]
async fn test_pdf_layout_ocr() -> Result<()> {
    init_logging();

    // Load PDF pages
    let pdf_path = PathBuf::from("/Users/larry/coderesp/aphelios_cli/test_data/extracted_pages.pdf");
    let layout_output = PathBuf::from("/Users/larry/coderesp/aphelios_cli/output/test_pdf2_layout");

    if !pdf_path.exists() {
        eprintln!("PDF not found: {:?}, skipping test", pdf_path);
        return Ok(());
    }

    let output_dir = PathBuf::from(layout_output);
    if output_dir.exists() {
        fs::remove_dir_all(&output_dir)?;
    }
    fs::create_dir_all(&output_dir)?;

    eprintln!("Loading PDF: {:?}", pdf_path);

    let load_start = Instant::now();
    let img_stream = load_pdf_images(pdf_path);
    pin_mut!(img_stream);

    let mut pages: Vec<image::DynamicImage> = Vec::new();
    while let Some(img_result) = img_stream.next().await {
        pages.push(img_result?.image);
    }
    let load_time = load_start.elapsed();
    eprintln!(
        "Loaded {} PDF pages in {:?} ({:.1} ms/page)",
        pages.len(),
        load_time,
        load_time.as_secs_f64() * 1000.0 / pages.len() as f64
    );

    if pages.is_empty() {
        eprintln!("No pages found in PDF, skipping test");
        return Ok(());
    }
    // Convert all pages to RGB for layout detection
    let rgb_pages: Vec<RgbImage> = pages.iter().map(|p| p.to_rgb8()).collect();
    // Initialize layout detector
    let mut layout = measure_time!(
        "load layout model",
        LayoutDetector::new(DOCLAYOUT_MODEL_FILE_PATH)?
    );

    let pad = 4;
    let mut clip_list: Vec<ClipInfo> = vec![];

    measure_time!("prepare clip for ocr",
    for ((page_index, page_img), original_img) in rgb_pages.iter().enumerate().zip(pages){
        let layout_res = measure_time!(format!("layout model infer, page_index {}", page_index), layout.detect(&page_img)?);
        for (reading_order, layout_clip) in layout_res.iter().enumerate(){
            let (img_w, img_h) = (original_img.width(), original_img.height());
            let x1 = (layout_clip.bbox[0] as u32).saturating_sub(pad);
            let y1 = (layout_clip.bbox[1] as u32).saturating_sub(pad);
            let x2 = ((layout_clip.bbox[2] as u32) + pad).min(img_w);
            let y2 = ((layout_clip.bbox[3] as u32) + pad).min(img_h);
            let w = x2 - x1;
            let h = y2 - y1;

            if w < 10 || h < 10 {
                continue;
            }
            let crop = original_img.crop_imm(x1, y1, w, h);

            clip_list.push(ClipInfo{
                page_index: page_index,
                reading_order: reading_order,
                label: layout_clip.label.to_string(),
                patches_count: dolphin_utils::count_patches(&crop, 4),
                clip_img: crop,
            });
        }
    });

    clip_list.sort_by_key(|clip| {clip.patches_count});

    let g_clips = group_clips_by_patches(&clip_list, 10, 1024 * 6);

    for c in &clip_list{
        eprintln!("{}, {}, {}, {}", c.page_index, c.reading_order, c.label, c.patches_count);
    }
    let model_id = Some("/Volumes/sw/pretrained_models/GLM-OCR");
    let device = get_device();

    info!("device is metal {}", device.is_metal());
    let dolphin_model_id = "/Volumes/sw/pretrained_models/Dolphin-v1.5";
    let run_on_dolphin = true;

    if run_on_dolphin{
        let mut dolphin_model = measure_time!("load model", DolphinModel::load_model(&dolphin_model_id)?);

        let start = Instant::now();
        for g_clip in &g_clips{

            info!("batch size {}", g_clip.len());
            let res = dolphin_model.run_clip_ocr_batch(g_clip)?;//148s - batch:1024 * 6
            for (r,c) in res.iter().zip(g_clip){
                info!("{}, {}, {}, {}", r, c.page_index, c.reading_order, c.label);
            }

            // for clip in g_clip{ // 65s
            //     let res = dolphin_model.generate_text_by_img(&clip.clip_img, "<s>Read text in the image. <Answer/>")?;
            //     info!("page_index {}, reading_order {}, label {}, text {}", clip.page_index, clip.reading_order, clip.label, res);
            // }
        }
        info!("dolphin - {}s", start.elapsed().as_secs_f64());
    }else {
        let ocr = measure_time!("load OCR model", GlmOcr::new_with_device(model_id, None, device)?);
        for clip in &clip_list{
            let prompt = prompt_for_label(clip.label.as_str());
            let res = measure_time!(
                format!("page_index {}, reading_order {}, label {}", clip.page_index, clip.reading_order, clip.label),
                ocr.recognize_with_max_tokens(&clip.clip_img, prompt, 512)?
            );
            info!("page_index {}, reading_order {}, label {}, text {}", clip.page_index, clip.reading_order, clip.label, res);
        }
    }

    Ok(())
}

/// Test that batched layout OCR produces the same number of sections as
/// sequential processing on a single page.
#[test]
fn test_glmocr_batched_single_page() -> Result<()> {
    init_logging();

    let img_path = "/Users/larry/coderesp/aphelios_cli/test_data/page_32.png";
    if !Path::new(img_path).exists() {
        eprintln!("Test image not found, skipping");
        return Ok(());
    }
    let image = image::open(img_path)?;

    let device = get_device();
    let model_id = Some("/Volumes/sw/pretrained_models/GLM-OCR");
    let ocr = GlmOcr::new_with_device(model_id, None, device)?;
    let mut layout = LayoutDetector::new(DOCLAYOUT_MODEL_FILE_PATH)?;

    let max_tokens = 256;

    // Non-batched reference (small max_tokens for fast test)
    let start_seq = Instant::now();
    let doc_seq = ocr.recognize_layout_structured(&image, &mut layout, max_tokens)?;
    let time_seq = start_seq.elapsed();
    eprintln!(
        "Sequential: {:?} ({} sections)",
        time_seq,
        doc_seq.sections.len()
    );

    // Batched (batch_size=2)
    eprint!("start batch 2");
    let mut layout2 = LayoutDetector::new(DOCLAYOUT_MODEL_FILE_PATH)?;
    let start_batch = Instant::now();
    let doc_batch =
        ocr.recognize_layout_structured_batched(&image, &mut layout2, max_tokens, Some(2))?;
    let time_batch = start_batch.elapsed();
    eprintln!(
        "Batched (bs=2): {:?} ({} sections, speedup={:.2}x)",
        time_batch,
        doc_batch.sections.len(),
        time_seq.as_secs_f64() / time_batch.as_secs_f64()
    );

    // Batched (batch_size=4)
    eprint!("start batch 4");
    let mut layout3 = LayoutDetector::new(DOCLAYOUT_MODEL_FILE_PATH)?;
    let start_batch4 = Instant::now();
    let doc_batch4 =
        ocr.recognize_layout_structured_batched(&image, &mut layout3, max_tokens, Some(4))?;
    let time_batch4 = start_batch4.elapsed();
    eprintln!(
        "Batched (bs=4): {:?} ({} sections, speedup={:.2}x)",
        time_batch4,
        doc_batch4.sections.len(),
        time_seq.as_secs_f64() / time_batch4.as_secs_f64()
    );

    eprintln!("\n--- Section count comparison ---");
    eprintln!("  Sequential sections: {}", doc_seq.sections.len());
    eprintln!("  Batched (bs=2) sections: {}", doc_batch.sections.len());
    eprintln!("  Batched (bs=4) sections: {}", doc_batch4.sections.len());

    Ok(())
}
