use anyhow::{anyhow, bail, Context, Error as E, Result};
use candle_core::{DType, Device, Tensor};
use hound;
use image::{DynamicImage, GenericImageView};
use ort::ep::{ExecutionProviderDispatch, CPU};
use std::path::Path;
use tokio::io::{AsyncWriteExt, BufWriter};
use tracing::{debug, info, warn};

use crate::{
    audio::{MonoBuffer, ResampleQuality},
    demucs::processor::StereoAudio,
    AudioLoader, Resampler,
};

pub const SAMPLE_RATE: usize = 16000;

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

pub fn truncate_by_chars(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("{msg}")]
    FileNotExists { msg: &'static str },

    #[error("{msg}")]
    PathMustDir { msg: &'static str },
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

pub fn load_and_resample_audio(path: &str, target_sr: Option<u32>) -> Result<Vec<Vec<f32>>> {
    // 1. 打开 wav
    let target_sr = target_sr.unwrap_or(16000);
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

pub fn get_append_filename_with_ext(input: &str, appender: &str, ext: &str) -> String {
    let path = Path::new(input);

    // 1. 获取父目录，如果没有则默认为当前目录 "."
    let parent = path.parent().unwrap_or_else(|| Path::new(""));

    // 2. 获取文件名主体 (Stem)
    let file_stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

    // 4. 使用 join 构建新路径，自动处理路径分隔符
    let new_filename = format!("{}{}.{}", file_stem, appender, ext);

    parent.join(new_filename).to_string_lossy().into_owned()
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

pub fn get_available_ep() -> Vec<ExecutionProviderDispatch> {
    let mut execution_providers = Vec::new();
    #[cfg(feature = "metal")]
    {
        use ort::ep::CoreML;
        execution_providers.push(CoreML::default().build().into());
    }
    #[cfg(feature = "cuda")]
    {
        use ort::ep::CUDA;
        execution_providers.push(CUDA::default().build().into());
    }
    execution_providers.push(CPU::default().build().into());
    execution_providers
}

pub fn get_default_device(cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "metal")]
    {
        return try_metal_device().with_context(|| {
            "Metal support is compiled in, but the current process could not initialize a Metal device"
        });
    }
    #[cfg(not(feature = "metal"))]
    #[cfg(feature = "cuda")]
    {
        use candle_core::utils::cuda_is_available;
        if cuda_is_available() {
            return Ok(Device::Cuda(0)?);
        }
    }
    #[cfg(not(feature = "metal"))]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        info!("Running on CPU, to run on GPU(metal), build this example with `--features metal`");
    }
    #[cfg(not(feature = "metal"))]
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        info!("Running on CPU, to run on GPU, build this example with `--features cuda`");
    }
    #[cfg(not(feature = "metal"))]
    {
        Ok(Device::Cpu)
    }
}

#[cfg(feature = "metal")]
fn try_metal_device() -> Result<Device> {
    use std::panic::{catch_unwind, set_hook, take_hook, AssertUnwindSafe};

    let previous_hook = take_hook();
    set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(|| Device::new_metal(0)));
    set_hook(previous_hook);

    match result {
        Ok(Ok(device)) => Ok(device),
        Ok(Err(err)) => Err(anyhow!("Metal device init failed: {err}")),
        Err(_) => Err(anyhow!(
            "Metal device init panicked, likely because no default Metal device was exposed to this process"
        )),
    }
}

pub fn normalize_audio<P: AsRef<Path>>(
    audio_path: P,
    target_sample_rate: u32,
) -> Result<(MonoBuffer, Vec<f32>)> {
    let audio_path = audio_path.as_ref();

    // Load audio using AudioLoader (supports multiple formats)
    let audio = AudioLoader::new()
        .load(audio_path)
        .with_context(|| format!("Failed to load audio file: {:?}", audio_path))?;

    info!(
        "Loaded audio: {} samples @ {}Hz, {} channels, {:.2}s",
        audio.len(),
        audio.sample_rate(),
        if audio.is_stereo() { 2 } else { 1 },
        audio.duration_secs()
    );

    // Convert to mono if stereo
    let mono = if audio.is_stereo() {
        debug!("Converting stereo to mono");
        audio.into_mono()
    } else {
        audio.into_mono()
    };

    // Resample to 16kHz if needed
    let mono = if mono.sample_rate != target_sample_rate {
        info!(
            "Resampling: {}Hz -> {}Hz",
            mono.sample_rate, target_sample_rate
        );
        let resampler = Resampler::new().with_quality(ResampleQuality::Fast);
        resampler.resample_mono(&mono, target_sample_rate)?
    } else {
        mono
    };

    // Convert f32 samples to i16 for feature extraction
    // AudioLoader normalizes to [-1.0, 1.0], convert back to i16 range
    let samples_i16: Vec<f32> = mono
        .samples
        .iter()
        .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0))
        .collect();

    info!(
        "Prepared audio: {} samples @ {}Hz ({:.2}s)",
        mono.samples.len(),
        mono.sample_rate,
        mono.duration_secs()
    );

    Ok((mono, samples_i16))
}

pub fn prepare_audio_16khz<P: AsRef<Path>>(audio_path: P) -> Result<(MonoBuffer, Vec<f32>)> {
    normalize_audio(audio_path, 16000)
}

pub async fn write_to_file(content: &Vec<String>, save_file: &str) -> Result<()> {
    let file = tokio::fs::File::create(save_file).await?;
    let mut writer = BufWriter::new(file);
    for line in content {
        writer.write_all(line.as_bytes()).await?;
        writer.write_all(b"\n").await?;
    }
    writer.flush().await?;
    Ok(())
}

pub const TEXIFY2_MODEL_DECODER_PATH: &str =
    "/Volumes/sw/aphelios_cli_models/onnx_models/texify2/decoder_model_merged.onnx";
pub const TEXIFY2_MODEL_ENCODER_PATH: &str =
    "/Volumes/sw/aphelios_cli_models/onnx_models/texify2/encoder_model.onnx";
pub const TEXIFY2_TOKENIZER_PATH: &str =
    "/Volumes/sw/aphelios_cli_models/onnx_models/texify2/tokenizer.json";

pub const RTDETR_V4_M: &str = "/Volumes/sw/aphelios_cli_models/onnx_models/rtdetr_v4_m.onnx";

///! without fallback
pub fn get_device() -> Device {
    get_default_device(false).unwrap_or_else(|err| {
        panic!("Device initialization failed: {err}");
    })
}

pub fn get_device_dtype() -> (Device, DType) {
    let device = get_default_device(false).unwrap_or_else(|err| {
        panic!("Device initialization failed: {err}");
    });
    let dtype = if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    };
    (device, dtype)
}

pub fn get_device_fallback() -> Device {
    get_default_device(false).unwrap_or_else(|err| {
        warn!("Device initialization failed, falling back to CPU: {err}");
        Device::Cpu
    })
}
