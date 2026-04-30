//! PicoDet layout detection model (PP-DocLayout-M).
//!
//! Detects 23 layout region classes in document images using ONNX Runtime.
//! Ported from Kreuzberg's kreuzberg-paddle-ocr implementation.

use anyhow::{Context, Result};
use ndarray::Array4;
use ort::session::Session;

/// Input dimensions for PP-DocLayout-M: 640x640 square.
const INPUT_SIZE: u32 = 640;

/// ImageNet RGB normalization constants.
const MEAN_RGB: [f32; 3] = [0.485, 0.456, 0.406];
const STD_RGB: [f32; 3] = [0.229, 0.224, 0.225];

const DEFAULT_SCORE_THRESHOLD: f32 = 0.3;
const DEFAULT_NMS_THRESHOLD: f32 = 0.5;

/// A detected layout region with bounding box and class label.
#[derive(Debug, Clone)]
pub struct LayoutDetection {
    /// Bounding box in original image coordinates: [x1, y1, x2, y2].
    pub bbox: [f32; 4],
    /// Class ID.
    pub class_id: u32,
    /// Detection confidence score.
    pub score: f32,
    /// Class label string.
    pub label: &'static str,
}

/// PP-DocLayout-M class labels (23 classes).
const LAYOUT_LABELS: &[&str] = &[
    "paragraph_title", // 0
    "image",           // 1
    "text",            // 2
    "number",          // 3
    "abstract",        // 4
    "content",         // 5
    "figure_title",    // 6
    "formula",         // 7
    "table",           // 8
    "table_title",     // 9
    "reference",       // 10
    "doc_title",       // 11
    "footnote",        // 12
    "header",          // 13
    "algorithm",       // 14
    "footer",          // 15
    "seal",            // 16
    "chart_title",     // 17
    "chart",           // 18
    "formula_number",  // 19
    "header_image",    // 20
    "footer_image",    // 21
    "aside_text",      // 22
];

/// PicoDet layout detection engine.
pub struct LayoutDetector {
    session: Session,
    score_threshold: f32,
    nms_threshold: f32,
}

impl LayoutDetector {
    /// Create a new layout detector, downloading the ONNX model if needed.
    pub fn new(model_file_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_file_path)
            .context("Failed to load layout ONNX model")?;

        Ok(Self {
            session,
            score_threshold: DEFAULT_SCORE_THRESHOLD,
            nms_threshold: DEFAULT_NMS_THRESHOLD,
        })
    }

    /// Detect layout regions in an image.
    ///
    /// Returns detections sorted by reading order (top-to-bottom, left-to-right).
    pub fn detect(&mut self, img: &image::RgbImage) -> Result<Vec<LayoutDetection>> {
        let orig_h = img.height();
        let orig_w = img.width();

        let input_tensor = Self::preprocess(img);
        let image_tensor = ort::value::Tensor::from_array(input_tensor)?;

        // scale_factor: [scale_y, scale_x]
        let scale_y = INPUT_SIZE as f32 / orig_h as f32;
        let scale_x = INPUT_SIZE as f32 / orig_w as f32;
        let scale_array = ndarray::Array2::from_shape_vec((1, 2), vec![scale_y, scale_x])
            .context("Failed to create scale array")?;
        let scale_tensor = ort::value::Tensor::from_array(scale_array)?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs!["image" => image_tensor, "scale_factor" => scale_tensor])
            .context("Layout model inference failed")?;

        // Output 0: [num_detections, 6] with [class_id, score, x1, y1, x2, y2]
        let mut output_iter = outputs.iter();
        let (_, det_output) = output_iter
            .next()
            .context("Missing layout detection output tensor")?;

        let (shape, raw) = det_output
            .try_extract_tensor::<f32>()
            .context("Failed to extract layout detection tensor")?;
        let num_dets = *shape.first().unwrap_or(&0) as usize;
        let stride = *shape.get(1).unwrap_or(&6) as usize;

        let mut raw_detections = Vec::with_capacity(num_dets);
        for i in 0..num_dets {
            let offset = i * stride;
            if offset + 5 < raw.len() {
                let class_id = raw[offset] as u32;
                let score = raw[offset + 1];
                let x1 = raw[offset + 2].clamp(0.0, orig_w as f32);
                let y1 = raw[offset + 3].clamp(0.0, orig_h as f32);
                let x2 = raw[offset + 4].clamp(0.0, orig_w as f32);
                let y2 = raw[offset + 5].clamp(0.0, orig_h as f32);

                if score >= self.score_threshold && x2 > x1 && y2 > y1 {
                    raw_detections.push((class_id, score, [x1, y1, x2, y2]));
                }
            }
        }

        // NMS per class
        let kept = Self::nms(&raw_detections, self.nms_threshold);

        // Cross-class overlap filter: if two detections (different classes) overlap
        // significantly, keep only the higher-scoring one
        let kept = Self::cross_class_filter(&kept);

        // Convert to LayoutDetection and sort by reading order
        let mut detections: Vec<LayoutDetection> = kept
            .into_iter()
            .map(|(class_id, score, bbox)| {
                let label = LAYOUT_LABELS
                    .get(class_id as usize)
                    .copied()
                    .unwrap_or("unknown");
                LayoutDetection {
                    bbox,
                    class_id,
                    score,
                    label,
                }
            })
            .collect();

        // Sort by reading order: top-to-bottom (y1), then left-to-right (x1)
        detections.sort_by(|a, b| {
            a.bbox[1]
                .total_cmp(&b.bbox[1])
                .then(a.bbox[0].total_cmp(&b.bbox[0]))
        });

        Ok(detections)
    }

    /// Process multiple images in batch through layout detection.
    ///
    /// Each image is independently preprocessed, inferenced, and post-processed.
    /// Inference runs sequentially (ONNX session requires `&mut self`).
    /// Returns results in input order, one `Vec<LayoutDetection>` per image.
    pub fn detect_batch(
        &mut self,
        images: &[image::RgbImage],
    ) -> Result<Vec<Vec<LayoutDetection>>> {
        let start = std::time::Instant::now();

        // Preprocess all images
        let preprocessed: Vec<(u32, u32, Array4<f32>, f32, f32)> = images
            .iter()
            .map(|img| {
                let (w, h) = (img.width(), img.height());
                let tensor = Self::preprocess(img);
                let scale_y = INPUT_SIZE as f32 / h as f32;
                let scale_x = INPUT_SIZE as f32 / w as f32;
                (w, h, tensor, scale_y, scale_x)
            })
            .collect();

        let prep_time = start.elapsed();
        tracing::info!(
            "Batch preprocessed {} images in {:?} ({:.1} img/s)",
            images.len(),
            prep_time,
            images.len() as f64 / prep_time.as_secs_f64()
        );

        // Run inference sequentially
        let mut results = Vec::with_capacity(images.len());
        for (orig_w, orig_h, tensor, scale_y, scale_x) in preprocessed {
            let image_tensor =
                ort::value::Tensor::from_array(tensor).context("Failed to create image tensor")?;
            let scale_array = ndarray::Array2::from_shape_vec((1, 2), vec![scale_y, scale_x])
                .context("Failed to create scale array")?;
            let scale_tensor = ort::value::Tensor::from_array(scale_array)?;

            let outputs = self
                .session
                .run(ort::inputs!["image" => image_tensor, "scale_factor" => scale_tensor])
                .context("Layout model inference failed")?;

            let mut output_iter = outputs.iter();
            let (_, det_output) = output_iter
                .next()
                .context("Missing layout detection output tensor")?;

            let (shape, raw) = det_output
                .try_extract_tensor::<f32>()
                .context("Failed to extract layout detection tensor")?;
            let num_dets = *shape.first().unwrap_or(&0) as usize;
            let stride = *shape.get(1).unwrap_or(&6) as usize;

            let mut raw_detections = Vec::with_capacity(num_dets);
            for j in 0..num_dets {
                let offset = j * stride;
                if offset + 5 < raw.len() {
                    let class_id = raw[offset] as u32;
                    let score = raw[offset + 1];
                    let x1 = raw[offset + 2].clamp(0.0, orig_w as f32);
                    let y1 = raw[offset + 3].clamp(0.0, orig_h as f32);
                    let x2 = raw[offset + 4].clamp(0.0, orig_w as f32);
                    let y2 = raw[offset + 5].clamp(0.0, orig_h as f32);

                    if score >= self.score_threshold && x2 > x1 && y2 > y1 {
                        raw_detections.push((class_id, score, [x1, y1, x2, y2]));
                    }
                }
            }

            let kept = Self::nms(&raw_detections, self.nms_threshold);
            let kept = Self::cross_class_filter(&kept);

            let mut detections: Vec<LayoutDetection> = kept
                .into_iter()
                .map(|(class_id, score, bbox)| {
                    let label = LAYOUT_LABELS
                        .get(class_id as usize)
                        .copied()
                        .unwrap_or("unknown");
                    LayoutDetection {
                        bbox,
                        class_id,
                        score,
                        label,
                    }
                })
                .collect();

            detections.sort_by(|a, b| {
                a.bbox[1]
                    .total_cmp(&b.bbox[1])
                    .then(a.bbox[0].total_cmp(&b.bbox[0]))
            });

            results.push(detections);
        }

        let total_time = start.elapsed();
        let per_image_ms = total_time.as_secs_f64() * 1000.0 / images.len() as f64;
        tracing::info!(
            "Batch {} images finished in {:?} ({:.1} ms/img, {:.1} img/s)",
            images.len(),
            total_time,
            per_image_ms,
            images.len() as f64 / total_time.as_secs_f64()
        );

        Ok(results)
    }

    /// Preprocess an image for PP-DocLayout-M: resize to 640x640, ImageNet normalize.
    fn preprocess(img: &image::RgbImage) -> Array4<f32> {
        let resized = image::imageops::resize(
            img,
            INPUT_SIZE,
            INPUT_SIZE,
            image::imageops::FilterType::Lanczos3,
        );

        let mut tensor = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
        for y in 0..INPUT_SIZE as usize {
            for x in 0..INPUT_SIZE as usize {
                let pixel = resized.get_pixel(x as u32, y as u32);
                let r = pixel[0] as f32 / 255.0;
                let g = pixel[1] as f32 / 255.0;
                let b = pixel[2] as f32 / 255.0;

                tensor[[0, 0, y, x]] = (r - MEAN_RGB[0]) / STD_RGB[0];
                tensor[[0, 1, y, x]] = (g - MEAN_RGB[1]) / STD_RGB[1];
                tensor[[0, 2, y, x]] = (b - MEAN_RGB[2]) / STD_RGB[2];
            }
        }

        tensor
    }

    /// Remove cross-class overlapping detections using intersection/min_area metric.
    ///
    /// Uses PaddleOCR-VL's approach: compute `intersection / min(area_i, area_j)`
    /// (overlap relative to the smaller box). If > 0.7, drop the smaller-area box.
    /// Exception: if one box is "image" and the other isn't, skip (preserve overlap).
    fn cross_class_filter(detections: &[(u32, f32, [f32; 4])]) -> Vec<(u32, f32, [f32; 4])> {
        if detections.is_empty() {
            return Vec::new();
        }

        let result: Vec<_> = detections.to_vec();
        let mut dropped = vec![false; result.len()];

        for i in 0..result.len() {
            if dropped[i] {
                continue;
            }
            for j in (i + 1)..result.len() {
                if dropped[j] {
                    continue;
                }

                let inter = intersection_area(&result[i].2, &result[j].2);
                let area_i = box_area(&result[i].2);
                let area_j = box_area(&result[j].2);
                let min_area = area_i.min(area_j);

                if min_area <= 0.0 {
                    continue;
                }

                let overlap_ratio = inter / min_area;
                if overlap_ratio > 0.7 {
                    let label_i = LAYOUT_LABELS
                        .get(result[i].0 as usize)
                        .copied()
                        .unwrap_or("");
                    let label_j = LAYOUT_LABELS
                        .get(result[j].0 as usize)
                        .copied()
                        .unwrap_or("");

                    // Skip filtering when one is "image" and the other isn't
                    let i_is_image = label_i == "image";
                    let j_is_image = label_j == "image";
                    if i_is_image != j_is_image {
                        continue;
                    }

                    // Drop the smaller-area box
                    if area_i >= area_j {
                        dropped[j] = true;
                    } else {
                        dropped[i] = true;
                    }
                }
            }
        }

        result
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !dropped[*i])
            .map(|(_, d)| d)
            .collect()
    }

    /// Apply non-maximum suppression per class.
    fn nms(detections: &[(u32, f32, [f32; 4])], threshold: f32) -> Vec<(u32, f32, [f32; 4])> {
        if detections.is_empty() {
            return Vec::new();
        }

        let mut sorted: Vec<_> = detections.to_vec();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut kept = Vec::new();
        let mut suppressed = vec![false; sorted.len()];

        for i in 0..sorted.len() {
            if suppressed[i] {
                continue;
            }
            kept.push(sorted[i]);

            for j in (i + 1)..sorted.len() {
                if suppressed[j] || sorted[j].0 != sorted[i].0 {
                    continue;
                }
                if iou(&sorted[i].2, &sorted[j].2) > threshold {
                    suppressed[j] = true;
                }
            }
        }

        kept
    }
}

/// Compute area of a bounding box.
fn box_area(a: &[f32; 4]) -> f32 {
    (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0)
}

/// Compute intersection area of two bounding boxes.
fn intersection_area(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    (x2 - x1).max(0.0) * (y2 - y1).max(0.0)
}

/// Compute intersection-over-union of two axis-aligned bounding boxes.
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);

    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let union = area_a + area_b - inter;

    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}
