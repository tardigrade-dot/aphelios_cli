use anyhow::{Ok, Result};
use tracing::info;
use usls::{Annotator, Config, Device, Image, Model, RTDETR};

use std::process::Command;
use tempfile::TempDir;

use crate::utils::common::RTDETR_V4_M;

pub fn label_video_batch(video_path: &str) -> Result<String> {
    let config = Config::rtdetr_v4_m()
        .with_model_file(RTDETR_V4_M)
        .with_class_confs(&[0.5]) // Confidence threshold
        .with_model_device(Device::CoreMl)
        .with_coreml_compute_units_all(1) // 使用 GPU
        .with_batch_all(10)
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
    let extract_time = std::time::Instant::now();
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
    info!(
        "Extract frames time: {:.2}s",
        extract_time.elapsed().as_secs_f32()
    );

    let annotator = Annotator::default();
    let mut processed_count = 0;
    // Step 1: 批量加载所有帧
    let mut images_to_process: Vec<Image> = Vec::with_capacity(frame_files.len());
    let mut frame_paths: Vec<&std::path::PathBuf> = Vec::with_capacity(frame_files.len());

    for (idx, frame_path) in frame_files.iter().enumerate() {
        info!("Loading frame {}/{}", idx + 1, frame_files.len());
        let dynamic_image = image::open(frame_path)?;
        images_to_process.push(Image::from(dynamic_image));
        frame_paths.push(frame_path);
    }

    // Step 2: 批量推理
    info!(
        "Running batch inference on {} frames...",
        images_to_process.len()
    );
    let inference_start = std::time::Instant::now();
    let results = model.run(&mut engines, &images_to_process)?;
    info!(
        "Batch inference completed in {:.2}s",
        inference_start.elapsed().as_secs_f32()
    );

    // Step 3: 处理结果并保存标注后的帧
    let post_process_time = std::time::Instant::now();
    for (idx, (image, ys)) in images_to_process.iter().zip(results.iter()).enumerate() {
        let hbbs: &[usls::Hbb] = ys.hbbs();
        if !hbbs.is_empty() {
            info!("Frame {}: Detected {} objects", idx + 1, hbbs.len());
            for det in hbbs.iter() {
                let class_id: usize = det.id().unwrap_or(0);
                let class_name = &model.names()[class_id];
                let confidence = det.confidence().unwrap_or(0.0);
                info!("  - Class: {}, Confidence: {:.2}", class_name, confidence);
            }

            // 标注图像
            let annotated = annotator.annotate(image, ys)?;
            let output_filename = frame_paths[idx].file_name().unwrap().to_str().unwrap();
            let output_path = output_frames_dir.join(output_filename);
            annotated.save(&output_path)?;
            processed_count += 1;
        } else {
            // 没有检测到物体，直接复制原图
            let output_filename = frame_paths[idx].file_name().unwrap().to_str().unwrap();
            let output_path = output_frames_dir.join(output_filename);
            images_to_process[idx].save(&output_path)?;
        }
    }

    info!(
        "Post process time: {:.2}s",
        post_process_time.elapsed().as_secs_f32()
    );
    info!("Processed {} frames with detections", processed_count);

    // Get video info (fps, codec, etc.) from original video
    let reconstruct_time = std::time::Instant::now();
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
        "Reconstruct video time: {:.2}s",
        reconstruct_time.elapsed().as_secs_f32()
    );

    info!(
        "Processed {} frames, saved to: {}",
        frame_files.len(),
        output_path
    );
    Ok(output_path)
}
