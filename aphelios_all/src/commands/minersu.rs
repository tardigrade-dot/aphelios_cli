use anyhow::{Error as E, Result};
use candle_core::backend::BackendDevice;
use image::{DynamicImage, GenericImageView, Rgb, RgbImage, imageops::FilterType};
use mistralrs::{Model, TextMessageRole, VisionMessages};
use tracing::info;

const MIN_PIXELS: u32 = 3136;
const MAX_PIXELS: u32 = 1605632;
const PATCH_SIZE: u32 = 14;

const IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const IMAGE_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

pub async fn run_mineru2(model: &Model, im: DynamicImage, prompt: &String) -> Result<String> {
    let image: DynamicImage =
        image::ImageReader::open("/Users/larry/Documents/resources/page_32.png")?
            .decode()
            .map_err(|e| E::msg(format!("Failed to decode image: {}", e)))?;

    let messages: VisionMessages = VisionMessages::new()
        .add_message(TextMessageRole::System, "You are a helpful assistant.")
        .add_image_message(TextMessageRole::User, prompt, vec![im], model)?;

    let response = model.send_chat_request(messages).await?;

    let res_str = response.choices[0].message.content.as_ref().unwrap();
    info!("response from model: -----------\n{}\n-----------", res_str);
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );
    Ok(res_str.to_string())
}

pub async fn run_mineru(model: &Model, im: DynamicImage, prompt: &String) -> Result<String> {
    let messages: VisionMessages = VisionMessages::new()
        .add_message(TextMessageRole::System, "You are a helpful assistant.")
        .add_image_message(TextMessageRole::User, prompt, vec![im], &model)?;

    let response = model.send_chat_request(messages).await?;

    let resp_str = response.choices[0].message.content.clone().unwrap();
    dbg!(
        &resp_str,
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(resp_str)
}

pub fn preprocess_to_image(img: DynamicImage) -> DynamicImage {
    let (mut w, mut h) = img.dimensions();
    let pixels = w * h;

    // 1️⃣ scale to satisfy pixel constraints
    let scale = if pixels < MIN_PIXELS {
        (MIN_PIXELS as f32 / pixels as f32).sqrt()
    } else if pixels > MAX_PIXELS {
        (MAX_PIXELS as f32 / pixels as f32).sqrt()
    } else {
        1.0
    };

    if scale != 1.0 {
        w = ((w as f32) * scale).round() as u32;
        h = ((h as f32) * scale).round() as u32;
    }

    // 2️⃣ patch alignment
    w = (w / PATCH_SIZE) * PATCH_SIZE;
    h = (h / PATCH_SIZE) * PATCH_SIZE;

    let img = img.resize_exact(w, h, FilterType::CatmullRom).to_rgb8();

    // 3️⃣ normalize -> denormalize -> image
    let mut out = RgbImage::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y).0;
            let mut rgb = [0u8; 3];

            for c in 0..3 {
                let v = p[c] as f32 / 255.0;
                let norm = (v - IMAGE_MEAN[c]) / IMAGE_STD[c];

                // 反归一化
                let denorm = (norm * IMAGE_STD[c] + IMAGE_MEAN[c]) * 255.0;
                rgb[c] = denorm.clamp(0.0, 255.0) as u8;
            }

            out.put_pixel(x, y, Rgb(rgb));
        }
    }

    DynamicImage::ImageRgb8(out)
}
