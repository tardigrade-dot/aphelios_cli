use std::path::{Path, PathBuf};

use ab_glyph::{FontRef, PxScale};
use anyhow::Result;
use async_stream::try_stream;
use candle_core::{Device, Tensor};
use futures_util::Stream;
use image::{DynamicImage, GenericImageView, Rgb, Rgba};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut, text_size};
use imageproc::rect::Rect;
use pdfium_render::prelude::*;
use tracing::info;

/// Get the directory where the running executable resides.
fn get_exe_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Initialize pdfium by searching common library locations.
///
/// Cached: pdfium's native bindings can only be initialized once per process.
fn bind_pdfium() -> Result<&'static Pdfium> {
    static PDFIUM: once_cell::sync::OnceCell<Pdfium> = once_cell::sync::OnceCell::new();

    PDFIUM
        .get_or_try_init::<_, anyhow::Error>(|| {
            let exe_dir = get_exe_dir();
            info!("Executable dir: {:?}", exe_dir);

            let pdfium_lib_path = Pdfium::pdfium_platform_library_name_at_path(&exe_dir);
            info!("Resolved pdfium lib path: {:?}", pdfium_lib_path);

            let pdfium = Pdfium::new(
                Pdfium::bind_to_library(&pdfium_lib_path)
                    .or_else(|_| {
                        Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(
                            "../libs/",
                        ))
                    })
                    .or_else(|_| {
                        Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(
                            "./libs/",
                        ))
                    })
                    .or_else(|_| Pdfium::bind_to_system_library())
                    .map_err(|e| anyhow::anyhow!("Failed to bind to pdfium library: {:?}", e))?,
            );

            Ok(pdfium)
        })
        .map_err(|e| anyhow::anyhow!("{}", e))
}

pub fn draw_bbox_and_save_multi(
    img: &DynamicImage,
    bbox_list: &Vec<[u32; 4]>,
    pad: u32,
    save_path: &PathBuf,
) {
    let (img_width, img_height) = img.dimensions();
    let mut img = img.to_rgba8();

    for bbox in bbox_list {
        let x1 = bbox[0].saturating_sub(pad).min(img_width - 1);
        let y1 = bbox[1].saturating_sub(pad).min(img_height - 1);
        let x2 = (bbox[2] + pad).min(img_width);
        let y2 = (bbox[3] + pad).min(img_height);

        let w = (x2 - x1).max(1);
        let h = (y2 - y1).max(1);

        let rect = Rect::at(x1 as i32, y1 as i32).of_size(w, h);
        draw_hollow_rect_mut(&mut img, rect, Rgba([255, 0, 0, 255]));
    }

    img.save(save_path).unwrap();
}

/// Draw layout detection results (bounding boxes + labels with scores) onto an image and save.
///
/// Each detection is drawn with a distinct colored bounding box and a label tag
/// showing the region class name and confidence score.
pub fn draw_layout_detections(
    img: &DynamicImage,
    detections: &[crate::glmocr::layout::LayoutDetection],
    pad: u32,
    save_path: &PathBuf,
) {
    let (img_width, img_height) = img.dimensions();
    let mut canvas = img.to_rgb8();

    // Load system font for label text
    let font_data =
        std::fs::read("/System/Library/Fonts/Supplemental/Arial.ttf").unwrap_or_default();
    let font_ref = if !font_data.is_empty() {
        FontRef::try_from_slice(&font_data).ok()
    } else {
        None
    };
    let scale = PxScale::from(16.0);

    let colors = [
        Rgb([0, 255, 0]),   // green
        Rgb([255, 0, 0]),   // red
        Rgb([0, 0, 255]),   // blue
        Rgb([255, 165, 0]), // orange
        Rgb([255, 0, 255]), // magenta
        Rgb([0, 255, 255]), // cyan
        Rgb([128, 0, 128]), // purple
        Rgb([255, 255, 0]), // yellow
    ];

    for (idx, det) in detections.iter().enumerate() {
        let color = colors[idx % colors.len()];

        let x1 = (det.bbox[0] as i32).saturating_sub(pad as i32).max(0);
        let y1 = (det.bbox[1] as i32).saturating_sub(pad as i32).max(0);
        let x2 = ((det.bbox[2] + pad as f32) as u32).min(img_width - 1) as i32;
        let y2 = ((det.bbox[3] + pad as f32) as u32).min(img_height - 1) as i32;
        let w = (x2 - x1).max(1) as u32;
        let h = (y2 - y1).max(1) as u32;

        let rect = Rect::at(x1, y1).of_size(w, h);
        draw_hollow_rect_mut(&mut canvas, rect, color);

        // Draw label with score
        let label = format!("{} ({:.2})", det.label, det.score);
        if let Some(ref f) = font_ref {
            let (tw, _th) = text_size(scale, f, &label);
            let label_y = y1.saturating_sub(22);
            // Filled background
            for by in label_y..label_y + 18 {
                for bx in x1..x1 + tw as i32 + 4 {
                    if bx >= 0
                        && by >= 0
                        && bx < canvas.width() as i32
                        && by < canvas.height() as i32
                    {
                        canvas.put_pixel(bx as u32, by as u32, color);
                    }
                }
            }
            draw_text_mut(
                &mut canvas,
                Rgb([255, 255, 255]),
                x1 + 2,
                label_y,
                scale,
                f,
                &label,
            );
        }
    }

    canvas.save(save_path).unwrap();
}

pub fn crop_image(img: &DynamicImage, bbox: [u32; 4], pad: u32) -> DynamicImage {
    let (img_width, img_height) = img.dimensions();

    let x1 = bbox[0].saturating_sub(pad).min(img_width - 1);
    let y1 = bbox[1].saturating_sub(pad).min(img_height - 1);
    let x2 = (bbox[2] + pad).min(img_width);
    let y2 = (bbox[3] + pad).min(img_height);

    let crop_width = (x2 - x1).max(1);
    let crop_height = (y2 - y1).max(1);

    img.crop_imm(x1, y1, crop_width, crop_height)
}

pub fn load_pdf_images(path: PathBuf) -> impl Stream<Item = Result<DynamicImage>> {
    try_stream! {
        let pdfium = bind_pdfium()?;
        let document = pdfium.load_pdf_from_file(path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid path"))?, None)?;

        let render_config = PdfRenderConfig::default();

        for page in document.pages().iter() {
            #[cfg(feature = "profiling")]
            let _span = tracing::info_span!("render_page").entered();

            let img = page.render_with_config(&render_config)?
                .as_image()?;

            #[cfg(feature = "profiling")]
            _span.exit();

            yield img;
        }
    }
}

pub fn get_tensor_from_image(
    img: &DynamicImage,
    target_height: u32,
    target_width: u32,
    device: &Device,
    dtype: candle_core::DType,
) -> Tensor {
    let resized = img.resize(
        target_width,
        target_height,
        image::imageops::FilterType::Triangle,
    );
    // Create a black canvas and center the resized image (HuggingFace uses black padding)
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

    let rgb = canvas;
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);

    // Donut uses [0.5, 0.5, 0.5] normalization
    let image_mean = [0.5f32, 0.5, 0.5];
    let image_std = [0.5f32, 0.5, 0.5];

    // Normalize: (H, W, C) -> (C, H, W) with normalization
    let mut normalized = vec![0f32; 3 * height * width];

    for (c, (&mean, &std)) in image_mean.iter().zip(image_std.iter()).enumerate() {
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                let idx = c * height * width + y * width + x;
                normalized[idx] = (pixel[c] as f32 / 255.0 - mean) / std;
            }
        }
    }
    let tensor: Tensor = Tensor::from_vec(normalized, (1, 3, height, width), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    tensor
}

pub fn transform_to_pixel_dynamic(
    bbox_coords: &[i32; 4],
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
    let x1_model_space = *x1 as f32 - x_offset as f32;
    let y1_model_space = *y1 as f32 - y_offset as f32;
    let x2_model_space = *x2 as f32 - x_offset as f32;
    let y2_model_space = *y2 as f32 - y_offset as f32;

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

/// Render a specific PDF page to a PNG image file.
///
/// `page_num` is 0-indexed.
pub fn pdf_page_to_png(pdf_path: &Path, page_num: usize, output_path: &Path) -> Result<()> {
    let pdfium = bind_pdfium()?;
    let document = pdfium.load_pdf_from_file(
        pdf_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid PDF path"))?,
        None,
    )?;

    let page = document.pages().get(page_num as PdfPageIndex)?;
    // Render at the page's natural point dimensions (1 pt ≈ 1 pixel at 72 DPI).
    // Using set_target_width with double the page width gives 2x resolution
    // (Retina-quality ~144 DPI) while maintaining aspect ratio.
    let tw = (page.width().value * 2.0) as i32;
    let render_config = PdfRenderConfig::default().set_target_width(tw);
    let image = page.render_with_config(&render_config)?.as_image()?;

    // Ensure parent directory exists
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    image.save(output_path)?;
    info!(
        "Saved page {} of {:?} to {:?}",
        page_num, pdf_path, output_path
    );

    Ok(())
}

/// Create a new PDF containing pages from `start_page` to the end of the source PDF.
///
/// `start_page` is 0-indexed.
pub fn pdf_extract_from(
    pdf_path: &Path,
    start_page: usize,
    end_page: usize,
    output_path: &Path,
) -> Result<()> {
    let pdfium = bind_pdfium()?;
    let source = pdfium.load_pdf_from_file(
        pdf_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid PDF path"))?,
        None,
    )?;

    let total_pages = source.pages().len() as usize;
    if start_page >= total_pages {
        anyhow::bail!(
            "start_page {} out of range, PDF has only {} pages",
            start_page,
            total_pages
        );
    }

    let mut dest = pdfium.create_new_pdf()?;
    dest.pages_mut().copy_page_range_from_document(
        &source,
        (start_page as i32)..=((end_page) as i32),
        0,
    )?;
    // Ensure parent directory exists
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    dest.save_to_file(output_path)?;

    info!(
        "Extracted pages {}-{} from {:?} to {:?}",
        start_page, end_page, pdf_path, output_path
    );

    Ok(())
}
