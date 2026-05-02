use anyhow::Result;
use image::DynamicImage;
pub mod dolphin;
pub mod glmocr;
pub mod doc_layout;

#[derive(Debug, Clone)]
pub struct ImageData {
    pub page_index: usize,
    pub total_pages: usize,
    pub image: DynamicImage,
}

impl ImageData {
    pub fn save_jpeg(&self, output_path: impl AsRef<std::path::Path>) -> Result<()> {
        self.image
            .save_with_format(output_path, image::ImageFormat::Jpeg)
            .map_err(|e| anyhow::anyhow!("Failed to save jpeg: {}", e))
    }

    pub fn save_png(&self, output_path: impl AsRef<std::path::Path>) -> Result<()> {
        self.image
            .save_with_format(output_path, image::ImageFormat::Png)
            .map_err(|e| anyhow::anyhow!("Failed to save png: {}", e))
    }

    pub fn save_bmp(&self, output_path: impl AsRef<std::path::Path>) -> Result<()> {
        self.image
            .save_with_format(output_path, image::ImageFormat::Bmp)
            .map_err(|e| anyhow::anyhow!("Failed to save bmp: {}", e))
    }

    pub fn save_tiff(&self, output_path: impl AsRef<std::path::Path>) -> Result<()> {
        self.image
            .save_with_format(output_path, image::ImageFormat::Tiff)
            .map_err(|e| anyhow::anyhow!("Failed to save tiff: {}", e))
    }
}
