use anyhow::anyhow;
use anyhow::{Error as E, Result, bail};
use hound;
use image::{DynamicImage, GenericImageView};
use mistralrs::{Device, Tensor};
use std::path::Path;
use tracing::{Level, info};

use crate::demucs::processor::StereoAudio;

static INIT: std::sync::Once = std::sync::Once::new();

pub fn init_tracing() {
    INIT.call_once(|| {
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(Level::INFO)
            // .with_env_filter(EnvFilter::from_default_env()) // 支持 RUST_LOG
            .with_file(true) // 显示文件名
            .with_env_filter("info,ort=off")
            .with_line_number(true) // 显示行号
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
                "end [{}] {} -> type={} | {} ms",
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

/// 使用线性插值重采样音频（参考 Web Audio API 实现）
fn resample_audio_linear(channel_data: &[Vec<f32>], src_sr: u32, target_sr: u32) -> Vec<Vec<f32>> {
    let channels = channel_data.len();
    let src_length = channel_data[0].len();
    let ratio = target_sr as f64 / src_sr as f64;
    let new_length = (src_length as f64 * ratio).floor() as usize;

    let mut output: Vec<Vec<f32>> = (0..channels)
        .map(|_| Vec::with_capacity(new_length))
        .collect();

    for ch in 0..channels {
        let input = &channel_data[ch];
        for i in 0..new_length {
            let src_idx = i as f64 / ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(input.len() - 1);
            let frac = src_idx - idx0 as f64;
            let sample = input[idx0] as f64 * (1.0 - frac) + input[idx1] as f64 * frac;
            output[ch].push(sample as f32);
        }
    }

    output
}

pub fn load_and_resample_audio(path: &str, target_sr: u32) -> Result<Vec<Vec<f32>>> {
    // 1. 打开 wav
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let src_sr = spec.sample_rate;
    let channels = spec.channels as usize;

    if channels == 0 {
        bail!("Audio file has zero channels");
    }

    // 2. 读取并归一化
    let raw_samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => match spec.bits_per_sample {
            16 => reader
                .samples::<i16>()
                .map(|s| s.unwrap() as f32 / i16::MAX as f32)
                .collect(),

            24 => reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / 8_388_607.0) // 2^23 - 1
                .collect(),

            32 => reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / i32::MAX as f32)
                .collect(),

            _ => bail!("Unsupported bit depth: {}", spec.bits_per_sample),
        },
    };

    if raw_samples.is_empty() {
        bail!("Audio file contains no samples");
    }

    // 3. 拆分声道（真正独立 Vec）
    let samples_per_channel = raw_samples.len() / channels;

    let mut channel_data: Vec<Vec<f32>> = (0..channels)
        .map(|_| Vec::with_capacity(samples_per_channel))
        .collect();

    for (i, &sample) in raw_samples.iter().enumerate() {
        let ch = i % channels;
        channel_data[ch].push(sample);
    }

    // 4. 如果不需要重采样
    if src_sr == target_sr {
        return Ok(channel_data);
    }

    // 5. 使用线性插值重采样（快速）
    Ok(resample_audio_linear(&channel_data, src_sr, target_sr))
}

pub fn get_append_filename(input: &str, appender: &str) -> String {
    let path = Path::new(input);

    // 1. 获取父目录，如果没有则默认为当前目录 "."
    let parent = path.parent().unwrap_or_else(|| Path::new(""));

    // 2. 获取文件名主体 (Stem)
    let file_stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

    // 3. 获取原始扩展名 (Extension)
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{}", e)) // 如果有扩展名，补上点
        .unwrap_or_else(|| "".to_string()); // 如果没扩展名，返回空字符串

    // 4. 使用 join 构建新路径，自动处理路径分隔符
    let new_filename = format!("{}{}{}", file_stem, appender, extension);

    parent.join(new_filename).to_string_lossy().into_owned()
}

// 辅助函数：合并多个立体声音轨
pub fn combine_stereo_tracks(tracks: &[&StereoAudio]) -> StereoAudio {
    if tracks.is_empty() {
        return StereoAudio {
            left: vec![0.0; 0],
            right: vec![0.0; 0],
        };
    }

    let sample_count = tracks[0].left.len();
    let mut combined_left = vec![0.0; sample_count];
    let mut combined_right = vec![0.0; sample_count];

    for track in tracks {
        for i in 0..std::cmp::min(sample_count, track.left.len()) {
            combined_left[i] += track.left[i];
            combined_right[i] += track.right[i];
        }
    }

    StereoAudio {
        left: combined_left,
        right: combined_right,
    }
}

// 辅助函数：保存音频轨道到 WAV 文件（带指定采样率）
pub fn save_audio_track_with_spec(audio: &StereoAudio, filename: &str, sample_rate: u32) {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(filename, spec).expect("Failed to create WAV writer");

    for i in 0..std::cmp::min(audio.left.len(), audio.right.len()) {
        let left_sample = (audio.left[i] * i16::MAX as f32) as i16;
        let right_sample = (audio.right[i] * i16::MAX as f32) as i16;

        writer
            .write_sample(left_sample)
            .expect("Failed to write left sample");
        writer
            .write_sample(right_sample)
            .expect("Failed to write right sample");
    }

    writer.finalize().expect("Failed to finalize WAV writer");
}
