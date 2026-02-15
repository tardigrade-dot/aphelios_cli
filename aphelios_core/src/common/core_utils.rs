use anyhow::{Error as E, Result};
use image::{DynamicImage, GenericImageView};
use mistralrs::{Device, Tensor};
use tracing::{Level, info};

static INIT: std::sync::Once = std::sync::Once::new();

pub fn init_tracing() {
    INIT.call_once(|| {
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(Level::INFO)
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

pub fn load_image(
    path: &str,
    target_height: u32,
    target_width: u32,
    device: &Device,
) -> Result<Tensor> {
    let img = image::ImageReader::open(path)?
        .decode()
        .map_err(|e| E::msg(format!("Failed to decode image: {}", e)))?;

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

    // Create tensor: (1, 3, H, W)
    let tensor = Tensor::from_vec(normalized, (1, 3, height, width), device)?;
    Ok(tensor)
}

#[macro_export]
macro_rules! measure_time {

        ($desc:expr, $expr:expr) => {{
            let __module = module_path!();
            let __file = file!();
            let __line = line!();

            info!("start [{}] {} ({}:{})", __module, $desc, __file, __line);

            let __start = std::time::Instant::now();
            let __result = $expr;
            let __duration = __start.elapsed();

            info!(
                "end   [{}] {} -> type={} | {} ms",
                __module,
                $desc,
                std::any::type_name_of_val(&__result),
                __duration.as_millis()
            );

            __result
        }};

        ($($tt:tt)*) => {{
            let __module = module_path!();
            let __file = file!();
            let __line = line!();

            let __start = std::time::Instant::now();
            let __result = { $($tt)* };
            let __duration = __start.elapsed();

            info!(
                "exec  [{}] ({}:{}) -> type={} | {} ms",
                __module,
                __file,
                __line,
                std::any::type_name_of_val(&__result),
                __duration.as_millis()
            );

            __result
        }};
    }
