use image::{DynamicImage, GenericImageView};
use mistralrs::{Device, Tensor};
use anyhow::{Error as E, Result};
use tracing::{Level, info};
use pdfium_render::prelude::*;

static INIT: std::sync::Once = std::sync::Once::new();

pub fn init_tracing() {
    INIT.call_once(|| {
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(Level::TRACE)
            .finish();
        tracing::subscriber::set_global_default(subscriber).unwrap();
    });
}

pub fn crop_image(img: &DynamicImage, bbox: [u32; 4], pad: u32) -> DynamicImage {
    let (img_width, img_height) = img.dimensions();

    let x1 = bbox[0].saturating_sub(pad).min(img_width - 1);
    let y1 = bbox[1].saturating_sub(pad).min(img_height - 1);
    let x2 = (bbox[2] + pad).min(img_width);
    let y2 = (bbox[3] + pad).min(img_height);

    let crop_width = (x2 - x1).max(1);
    let crop_height = (y2 - y1).max(1);

    info!("{} {} {} {}", x1, y1, crop_width, crop_height);
    img.crop_imm(x1, y1, crop_width, crop_height)
}

pub fn pdf_to_images(pdf_path: &str) -> Result<Vec<DynamicImage>, E> {
    // 1. 初始化 Pdfium (指向你下载的 dylib)
    // 建议将 dylib 路径参数化或放在配置文件中
    // let library_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("libpdfium.dylib");
    // let pdfium = Pdfium::new(Pdfium::bind_to_library(library_path)?);
    // let pdfium = Pdfium::new(Pdfium::bind_to_system_library()?);
    let pdfium = Pdfium::new(Pdfium::bind_to_library("/Users/larry/coderesp/aphelios_cli/libpdfium.dylib")?);
    
    // 2. 加载 PDF 文档
    let document = pdfium.load_pdf_from_file(pdf_path, None)?;

    // 3. 设置渲染配置
    // set_target_width(2000) 会在保持比例的前提下，将页面渲染为 2000 像素宽
    // 这决定了输出图片的清晰度 (DPI)
    let render_config = PdfRenderConfig::new()
        .set_target_width(2000) 
        .rotate_if_landscape(PdfPageRenderRotation::None, true);

    let mut images = Vec::new();

    // 4. 遍历并渲染每一页
    for (index, page) in document.pages().iter().enumerate() {
        
        // 将页面渲染为 PdfBitmap，然后转换为 image::DynamicImage
        let bitmap = page.render_with_config(&render_config)?;
        let dyn_image = bitmap.as_image(); // 这就是 image 库的 DynamicImage
        
        images.push(dyn_image);
    }

    Ok(images)
}

fn load_images(path: &str) -> Result<Vec<DynamicImage>> {
    match path.rsplit('.').next().unwrap_or("").to_lowercase().as_str() {
        "pdf" => pdf_to_images(path),
        _ => Ok(vec![image::ImageReader::open(path)?.decode()?]),
    }
}

pub fn load_image(path: &str, target_height: u32, target_width: u32, device: &Device) -> Result<Vec<(Tensor, DynamicImage)>> {

    let images = load_images(path)?;

    let mut img_li: Vec<(Tensor, DynamicImage)> = Vec::new();

    for img in images{

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
        let tensor = Tensor::from_vec(normalized, (1, 3, height, width), device)?;
        img_li.push((tensor, img));
    }
    Ok(img_li)
}