use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Error as E, Result};
use async_stream::{stream, try_stream};
use futures_util::{Stream, StreamExt, pin_mut};
use hayro::hayro_interpret::InterpreterSettings;
use hayro::hayro_syntax::Pdf;
use hayro::vello_cpu::color::palette::css::WHITE;
use hayro::{RenderSettings, render};
use image::{DynamicImage, GenericImageView, Rgba};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use mistralrs::{Device, Tensor};
use tracing::{Level, info};

static INIT: std::sync::Once = std::sync::Once::new();

pub fn init_tracing() {
    INIT.call_once(|| {
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(Level::TRACE)
            .finish();
        tracing::subscriber::set_global_default(subscriber).unwrap();
    });
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

pub fn load_pdf_images(path: &str) -> impl Stream<Item = Result<DynamicImage>> {
    try_stream! {
        let pdf_file = std::fs::read(path)?;
        let pdf = Pdf::new(Arc::new(pdf_file)).unwrap();

        let interpreter_settings = InterpreterSettings::default();
        let render_settings = RenderSettings {
            bg_color: WHITE, // 建议显式设置背景色，否则透明 PDF 可能会变黑
            ..Default::default()
        };

        for page in pdf.pages().iter() {
            let pixmap = render(page, &interpreter_settings, &render_settings);
            let png_bytes = pixmap.into_png()
                .map_err(|e| anyhow::anyhow!("PNG encoding failed: {:?}", e))?;

            // 2. 利用 image 库直接从内存字节加载为 DynamicImage
            let img = image::load_from_memory(&png_bytes)?;
            yield img;
        }
    }
}

pub fn get_tensor_from_image(
    img: &DynamicImage,
    target_height: u32,
    target_width: u32,
    device: &Device,
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
    let tensor: Tensor = Tensor::from_vec(normalized, (1, 3, height, width), device).unwrap();
    tensor
}
