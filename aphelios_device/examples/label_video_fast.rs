use std::io::{Read, Write};
use std::process::{Command, Stdio};

use anyhow::Result;
use clap::Parser;
use tracing::info;

use usls::{Annotator, Config, Image, Model};

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    input: String,

    #[arg(short, long)]
    output: String,

    #[arg(short, long)]
    model: String,

    #[arg(long, default_value_t = 0.5)]
    conf: f32,

    #[arg(long, default_value_t = 8)]
    batch: usize,

    #[arg(long, default_value_t = 640)]
    width: u32,

    #[arg(long, default_value_t = 480)]
    height: u32,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().init();
    let args = Args::parse();

    let frame_size = (args.width * args.height * 3) as usize;

    // ─────────────────────────────────────
    // FFmpeg 解码（stdout → Rust）
    // ─────────────────────────────────────
    let mut decoder = Command::new("ffmpeg")
        .args([
            "-i",
            &args.input,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-vf",
            &format!("scale={}x{}", args.width, args.height),
            "-",
        ])
        .stdout(Stdio::piped())
        .spawn()?;

    // ─────────────────────────────────────
    // FFmpeg 编码（Rust → stdin）
    // ─────────────────────────────────────
    let mut encoder = Command::new("ffmpeg")
        .args([
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{}x{}", args.width, args.height),
            "-r",
            "30",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-y",
            &args.output,
        ])
        .stdin(Stdio::piped())
        .spawn()?;

    let mut reader = decoder.stdout.take().unwrap();
    let mut writer = encoder.stdin.take().unwrap();

    // ─────────────────────────────────────
    // 模型加载
    // ─────────────────────────────────────
    let config = Config::rtdetr_v4_m()
        .with_model_file(&args.model)
        .with_class_confs(&[args.conf])
        // 👉 强烈建议打开（Mac）
        // .with_model_device(usls::Device::CoreMl)
        .commit()?;

    let (mut model, mut engines) = usls::RTDETR::build(config)?;
    let annotator = Annotator::default();

    // ─────────────────────────────────────
    // 主循环（batch）
    // ─────────────────────────────────────
    let mut buffer = vec![0u8; frame_size];
    let mut frames = Vec::with_capacity(args.batch);
    let mut raw_frames = Vec::with_capacity(args.batch);

    loop {
        frames.clear();
        raw_frames.clear();

        // 读取 batch
        for _ in 0..args.batch {
            if reader.read_exact(&mut buffer).is_err() {
                break;
            }

            let img = image::RgbImage::from_raw(args.width, args.height, buffer.clone()).unwrap();

            let dyn_img = image::DynamicImage::ImageRgb8(img.clone());

            frames.push(Image::from(dyn_img.clone()));
            raw_frames.push(dyn_img);
        }

        if frames.is_empty() {
            break;
        }

        // ─────────────────────────────
        // 推理（batch）
        // ─────────────────────────────
        let ys = model.run(&mut engines, &frames)?;

        // ─────────────────────────────
        // 标注 + 写回
        // ─────────────────────────────
        for (i, y) in ys.iter().enumerate() {
            let mut img = raw_frames[i].clone();

            if !y.hbbs().is_empty() {
                img = annotator.annotate(&frames[i], y)?.into_dyn();
            }

            let rgb = img.to_rgb8();
            writer.write_all(&rgb)?;
        }
    }

    drop(writer);
    encoder.wait()?;
    decoder.wait()?;

    info!("Done!");

    Ok(())
}
