pub mod batch;

use anyhow::{Ok, Result};
use image::DynamicImage;
use tracing::info;
use usls::{Annotator, Config, Image, Model, RTDETR};

use std::process::Command;
use tempfile::TempDir;

use crate::utils::common::RTDETR_V4_M;

pub fn label_video(video_path: &str) -> Result<String> {
    let config = Config::rtdetr_v4_m()
        .with_model_file(RTDETR_V4_M)
        .with_class_confs(&[0.5]) // Confidence threshold
        .commit()?;

    let (mut model, mut engines) = RTDETR::build(config)?;

    info!("Loaded model: {}", model.spec());
    info!("Classes: {:?}", model.names());

    // Create temp directory for frames
    let temp_dir = TempDir::new()?;
    let frames_dir = temp_dir.path().join("frames");
    let output_frames_dir = temp_dir.path().join("output_frames");
    std::fs::create_dir_all(&frames_dir)?;
    std::fs::create_dir_all(&output_frames_dir)?;

    // Extract frames from video using ffmpeg command
    info!("Extracting frames from video: {}", video_path);
    let extract_status = Command::new("ffmpeg")
        .args([
            "-i",
            video_path,
            "-vf",
            "fps=30",
            "-q:v",
            "2",
            frames_dir.join("frame_%04d.png").to_str().unwrap(),
            "-y",
        ])
        .status()?;

    if !extract_status.success() {
        return Err(anyhow::anyhow!("Failed to extract frames from video"));
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

    let annotator = Annotator::default();
    let mut processed_count = 0;

    // Process each frame
    for (idx, frame_path) in frame_files.iter().enumerate() {
        info!("Processing frame {}/{}", idx + 1, frame_files.len());

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
                info!("Frame {}: Detected {} objects", idx + 1, hbbs.len());
                for det in hbbs.iter() {
                    let class_id: usize = det.id().unwrap_or(0);
                    let class_name = &model.names()[class_id];
                    let confidence = det.confidence().unwrap_or(0.0);
                    info!("  Class: {}, Confidence: {:.2}", class_name, confidence);
                }
                annotated_image = annotator.annotate(&image, y)?.into_dyn();
                processed_count += 1;
            }
        }

        // Save annotated frame
        let output_filename = frame_path.file_name().unwrap().to_str().unwrap();
        let output_path = output_frames_dir.join(output_filename);
        annotated_image.save(&output_path)?;
    }

    info!("Processed {} frames with detections", processed_count);

    // Get video info (fps, codec, etc.) from original video
    let probe_output = Command::new("ffprobe")
        .args([
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "v:0",
            video_path,
        ])
        .output()?;

    let probe_json = String::from_utf8_lossy(&probe_output.stdout);
    let probe_value: serde_json::Value = serde_json::from_str(&probe_json)?;

    // Extract fps from stream info
    let fps = probe_value["streams"][0]["r_frame_rate"]
        .as_str()
        .unwrap_or("30/1");

    // Parse fps string like "30/1" or "30000/1001"
    let fps_parts: Vec<&str> = fps.split('/').collect();
    let fps_num: f32 = fps_parts.get(0).unwrap_or(&"30").parse().unwrap_or(30.0);
    let fps_den: f32 = fps_parts.get(1).unwrap_or(&"1").parse().unwrap_or(1.0);
    let fps_value = fps_num / fps_den;

    // Get codec from original video
    let _codec_name = probe_value["streams"][0]["codec_name"]
        .as_str()
        .unwrap_or("h264");

    // Reconstruct video from annotated frames
    let output_path = format!(
        "output/labeled_{}.mp4",
        chrono::Local::now().format("%Y%m%d_%H%M%S")
    );

    info!("Reconstructing video at {} fps", fps_value);
    let reconstruct_status = Command::new("ffmpeg")
        .args([
            "-framerate",
            &fps_value.to_string(),
            "-i",
            output_frames_dir.join("frame_%04d.png").to_str().unwrap(),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-y",
            &output_path,
        ])
        .status()?;

    if !reconstruct_status.success() {
        return Err(anyhow::anyhow!("Failed to reconstruct video from frames"));
    }

    info!(
        "Processed {} frames, saved to: {}",
        frame_files.len(),
        output_path
    );
    Ok(output_path)
}

pub fn label_images(image_path: &str) -> Result<Vec<DynamicImage>> {
    let config = Config::rtdetr_v4_m()
        .with_model_file(RTDETR_V4_M)
        .with_class_confs(&[0.5]) // Confidence threshold
        .commit()?;

    let (mut model, mut engines) = RTDETR::build(config)?;

    info!("Loaded model: {}", model.spec());
    info!("Classes: {:?}", model.names());

    let dynamic_image = image::open(image_path)?;
    info!(
        "Loaded image: {}x{}",
        dynamic_image.width(),
        dynamic_image.height()
    );

    let image = Image::from(dynamic_image);

    // Run inference
    let ys = model.run(&mut engines, &[image.clone()])?;

    // Process detection results
    let mut image_result = Vec::new();
    for y in ys.iter() {
        // Y contains detection results - access hbbs (horizontal bounding boxes) directly
        let hbbs: &[usls::Hbb] = y.hbbs();
        if hbbs.is_empty() {
            info!("No objects detected");
            continue;
        }

        info!("Detected {} objects", hbbs.len());

        for (i, det) in hbbs.iter().enumerate() {
            let class_id: usize = det.id().unwrap_or(0);
            let class_name = &model.names()[class_id];
            let confidence = det.confidence().unwrap_or(0.0);
            let (x1, y1, x2, y2) = det.xyxy(); // [x_min, y_min, x_max, y_max]

            info!(
                "  [{}] Class: {}, Confidence: {:.2}, BBox: [{:.1}, {:.1}, {:.1}, {:.1}]",
                i, class_name, confidence, x1, y1, x2, y2
            );
        }

        // Annotate image with bounding boxes
        let annotator = usls::Annotator::default();
        let annotated = annotator.annotate(&image, y)?;

        // Save annotated image
        let output_path = format!(
            "output/rtdetr_{}.jpg",
            chrono::Local::now().format("%Y%m%d_%H%M%S")
        );
        annotated.save(&output_path)?;
        info!("Saved annotated image to: {}", output_path);

        image_result.push(annotated.into_dyn());
    }
    Ok(image_result)
}
