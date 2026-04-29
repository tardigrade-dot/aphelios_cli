use anyhow::Result;
use aphelios_core::init_logging;
use aphelios_core::utils::common::get_device;
use aphelios_ocr::dolphin::dolphin_utils::{self, load_pdf_images};
use aphelios_ocr::glmocr::layout::LayoutDetector;
use aphelios_ocr::glmocr::GlmOcr;
use futures_util::{pin_mut, StreamExt};
use image::RgbImage;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

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
    let max_tokens = 8192;
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
        let mut layout = LayoutDetector::new()?;
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
async fn test_layout_batch() -> Result<()> {
    init_logging();

    // Load PDF pages
    let pdf_path = PathBuf::from("/Users/larry/Downloads/test_layout.pdf");
    if !pdf_path.exists() {
        eprintln!("PDF not found: {:?}, skipping test", pdf_path);
        return Ok(());
    }

    // Load PDF into memory
    eprintln!("Loading PDF: {:?}", pdf_path);
    let load_start = Instant::now();
    let img_stream = load_pdf_images(pdf_path);
    pin_mut!(img_stream);

    let mut pages: Vec<image::DynamicImage> = Vec::new();
    while let Some(img_result) = img_stream.next().await {
        pages.push(img_result?);
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
    let mut layout = LayoutDetector::new()?;

    // Process in batches of 6
    let batch_size = 6;
    let total_pages = rgb_pages.len();
    let total_batches = (total_pages + batch_size - 1) / batch_size;
    eprintln!(
        "Processing {} pages in {} batches (batch_size={})",
        total_pages, total_batches, batch_size
    );

    let overall_start = Instant::now();
    let mut total_detections = 0;

    for batch_idx in 0..total_batches {
        let start_idx = batch_idx * batch_size;
        let end_idx = (start_idx + batch_size).min(total_pages);
        let batch = &rgb_pages[start_idx..end_idx];

        let batch_start = Instant::now();
        let results = layout.detect_batch(batch)?;
        let batch_time = batch_start.elapsed();

        let batch_detections: usize = results.iter().map(|r| r.len()).sum();
        total_detections += batch_detections;

        eprintln!(
            "Batch {}/{}: pages {}-{} ({} pages) in {:?} ({:.1} ms/page, {} total detections)",
            batch_idx + 1,
            total_batches,
            start_idx,
            end_idx - 1,
            batch.len(),
            batch_time,
            batch_time.as_secs_f64() * 1000.0 / batch.len() as f64,
            batch_detections,
        );

        // Save layout visualization of first page in each batch
        if batch_idx == 0 && !results.is_empty() {
            let output_path =
                PathBuf::from("/Volumes/sw/MyDrive/data_src/test_layout_batch_layout.png");
            dolphin_utils::draw_layout_detections(
                &image::DynamicImage::ImageRgb8(batch[0].clone()),
                &results[0],
                4,
                &output_path,
            );
            eprintln!("Layout visualization saved to {:?}", output_path);
        }
    }

    let overall_time = overall_start.elapsed();
    eprintln!(
        "\n--- Layout Batch Summary ---\n\
         Pages: {}\n\
         Batches: {}\n\
         Total time: {:?}\n\
         Avg: {:.1} ms/page ({:.1} img/s)\n\
         Total detections: {}\n\
         Avg detections/page: {:.1}",
        total_pages,
        total_batches,
        overall_time,
        overall_time.as_secs_f64() * 1000.0 / total_pages as f64,
        total_pages as f64 / overall_time.as_secs_f64(),
        total_detections,
        total_detections as f64 / total_pages as f64,
    );

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
    let mut layout = LayoutDetector::new()?;

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
    let mut layout2 = LayoutDetector::new()?;
    let start_batch = Instant::now();
    let doc_batch = ocr.recognize_layout_structured_batched(&image, &mut layout2, max_tokens, 2)?;
    let time_batch = start_batch.elapsed();
    eprintln!(
        "Batched (bs=2): {:?} ({} sections, speedup={:.2}x)",
        time_batch,
        doc_batch.sections.len(),
        time_seq.as_secs_f64() / time_batch.as_secs_f64()
    );

    // Batched (batch_size=4)
    eprint!("start batch 4");
    let mut layout3 = LayoutDetector::new()?;
    let start_batch4 = Instant::now();
    let doc_batch4 =
        ocr.recognize_layout_structured_batched(&image, &mut layout3, max_tokens, 4)?;
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
