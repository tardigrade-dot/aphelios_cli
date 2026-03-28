//! RTDETR video labeling pipeline (production-ready)
//!
//! Pipeline:
//!   FFmpeg extract frames → process PNGs → FFmpeg encode
//!
//! Usage:
//!   cargo run --release --example label_video_rtdetr -- \
//!     --input input.mp4 --output output.mp4 \
//!     --model /path/to/rtdetr_v4_m.onnx \
//!     --conf 0.5

use std::path::PathBuf;
use std::process::Command;

use anyhow::Result;
use clap::Parser;
use tempfile::TempDir;
use tracing::info;

use usls::{Annotator, Config, Image, Model};

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(about = "Run RTDETR object detection on a video file with usls Annotator")]
struct Args {
    /// Input video path (any format FFmpeg can decode)
    #[arg(short, long)]
    input: PathBuf,

    /// Output video path (.mp4)
    #[arg(short, long)]
    output: PathBuf,

    /// RTDETR ONNX model path
    #[arg(short, long)]
    model: PathBuf,

    /// Confidence threshold  (default 0.5)
    #[arg(long, default_value_t = 0.5)]
    conf: f32,

    /// CRF quality (0-51, lower = better quality, default 23)
    #[arg(long, default_value_t = 23)]
    crf: u8,
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("rtdetr_video=info")
        .init();

    let args = Args::parse();

    info!("Input:  {:?}", args.input);
    info!("Output: {:?}", args.output);
    info!("Model:  {:?}", args.model);
    info!("Conf threshold: {}", args.conf);

    // Create temp directory for frames
    let temp_dir = TempDir::new()?;
    let frames_dir = temp_dir.path().join("frames");
    let output_frames_dir = temp_dir.path().join("output_frames");
    std::fs::create_dir_all(&frames_dir)?;
    std::fs::create_dir_all(&output_frames_dir)?;

    // Step 1: Extract frames from video using ffmpeg
    info!("Extracting frames from video...");
    let extract_status = Command::new("ffmpeg")
        .args([
            "-i",
            args.input.to_string_lossy().as_ref(),
            "-vf",
            "fps=30",
            "-q:v",
            "2",
            frames_dir.join("frame_%04d.png").to_string_lossy().as_ref(),
            "-y",
        ])
        .status()?;

    if !extract_status.success() {
        anyhow::bail!("Failed to extract frames from video");
    }

    // Get list of frame files
    let mut frame_files: Vec<_> = std::fs::read_dir(&frames_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map_or(false, |ext| ext == "png" || ext == "jpg")
        })
        .collect();
    frame_files.sort();

    info!("Found {} frames to process", frame_files.len());

    // Step 2: Load model
    let config = Config::rtdetr_v4_m()
        .with_model_file(args.model.to_string_lossy().as_ref())
        .with_class_confs(&[args.conf])
        .commit()?;

    let (mut model, mut engines) = usls::RTDETR::build(config)?;
    let annotator = Annotator::default();

    info!("Loaded model: {}", model.spec());

    // Step 3: Process each frame
    let mut processed_count = 0u64;

    for (idx, frame_path) in frame_files.iter().enumerate() {
        if idx % 50 == 0 {
            info!("Processing frame {}/{}", idx + 1, frame_files.len());
        }

        // Load frame
        let dynamic_image = image::open(frame_path)?;
        let image = Image::from(dynamic_image.clone());

        // Run inference
        let ys = model.run(&mut engines, &[image.clone()])?;

        // Annotate frame with detection results
        let mut annotated_image = dynamic_image;
        for y in ys.iter() {
            let hbbs: &[usls::Hbb] = y.hbbs();
            if !hbbs.is_empty() {
                annotated_image = annotator.annotate(&image, y)?.into_dyn();
                processed_count += 1;
            }
        }

        // Save annotated frame
        let output_filename = frame_path.file_name().unwrap().to_string_lossy();
        let output_path = output_frames_dir.join(output_filename.as_ref());
        annotated_image.save(&output_path)?;
    }

    info!("Processed {} frames with detections", processed_count);

    // Step 4: Reconstruct video from annotated frames
    info!("Reconstructing video...");
    let reconstruct_status = Command::new("ffmpeg")
        .args([
            "-framerate",
            "30",
            "-i",
            output_frames_dir
                .join("frame_%04d.png")
                .to_string_lossy()
                .as_ref(),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            &args.crf.to_string(),
            "-y",
            args.output.to_string_lossy().as_ref(),
        ])
        .status()?;

    if !reconstruct_status.success() {
        anyhow::bail!("Failed to reconstruct video from frames");
    }

    info!(
        "Done! Processed {} frames -> {:?}",
        frame_files.len(),
        args.output
    );

    Ok(())
}
