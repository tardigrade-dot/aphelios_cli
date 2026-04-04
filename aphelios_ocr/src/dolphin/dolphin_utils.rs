use std::path::PathBuf;

use anyhow::Result;
use async_stream::try_stream;
use candle_core::{Device, Tensor};
use futures_util::Stream;
use image::{DynamicImage, GenericImageView, Rgba};
use imageproc::drawing::draw_hollow_rect_mut;
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
        // On macOS, when running from an .app bundle, the executable is in
        // Contents/MacOS/, so we use the executable directory to find libpdfium.dylib
        // which is bundled in the same directory.
        let exe_dir = get_exe_dir();
        info!("Executable dir: {:?}", exe_dir);

        let pdfium_lib_path = Pdfium::pdfium_platform_library_name_at_path(&exe_dir);
        info!("Resolved pdfium lib path: {:?}", pdfium_lib_path);

        let pdfium = Pdfium::new(
            Pdfium::bind_to_library(&pdfium_lib_path)
                .or_else(|_| Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("../libs/")))
                .or_else(|_| Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("./libs/")))
                .or_else(|_| Pdfium::bind_to_system_library())
                .map_err(|e| anyhow::anyhow!("Failed to bind to pdfium library: {:?}", e))?
        );

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
