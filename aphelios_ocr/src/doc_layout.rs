use std::path::Path;

use crate::{dolphin::dolphin_utils::{self}, glmocr::layout::{LayoutDetection, LayoutDetector}};
use anyhow::Result;
use image::DynamicImage;

pub struct ImageInfo {
    image_file: String,
    pub image: DynamicImage
}

impl ImageInfo {
    pub fn new(image_file: impl AsRef<Path>) -> Result<ImageInfo>{
        let img_fi = image_file.as_ref();
        Ok(ImageInfo { image_file: img_fi.to_str().unwrap().to_string(), image: image::open(img_fi)? })
    }
    pub fn get_file_name(self: &Self) -> &str{
        return &self.image_file;
    }
}

pub fn run_pp_layout(pp_layout_model_file: &str, image_info: &ImageInfo, layout_output_path: Option<impl AsRef<Path>>) -> Result<Vec<LayoutDetection>>{
    let mut layout = LayoutDetector::new(pp_layout_model_file)?;
    let layout_result = layout.detect(&image_info.image.to_rgb8())?;

    if let Some(layout_output) = layout_output_path{
        dolphin_utils::draw_layout_detections(&image_info.image, &layout_result, 4, layout_output.as_ref());
    }
    Ok(layout_result)
}

pub async fn run_pp_layout_batch(pp_layout_model_file: &str, image_path: impl AsRef<Path>, layout_output_path: Option<impl AsRef<Path>>) -> Result<Vec<LayoutDetection>>{

    // let img_stream = load_pdf_images(image_path);
    // pin_mut!(img_stream);

    // let mut pages: Vec<image::DynamicImage> = Vec::new();
    // while let Some(img_result) = img_stream.next().await {
    //     pages.push(img_result?.image);
    // }

    Ok(Vec::new())
}
