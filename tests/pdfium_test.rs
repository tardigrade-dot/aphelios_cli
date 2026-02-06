use anyhow::{Error as E, Result};
use aphelios_cli::dolphin::utils;
use pdfium_render::prelude::*;

#[test]
fn pdfium_render_test1() -> Result<(), Box<dyn std::error::Error>> {
    // 动态获取项目根目录下的 dylib 路径
    let library_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("libpdfium.dylib");

    // 绑定库
    let pdfium = Pdfium::new(Pdfium::bind_to_library(library_path)?);
    Ok(())
}

#[test]
fn pdfium_render_test() -> Result<()> {
    let image_path =
        "/Users/larry/github.com/tardigrade-dot/colab-script2/data_src/extracted_pages.pdf";
    let res = utils::pdf_to_images(image_path)?;

    println!("共[{}", res.len());
    Ok(())
}
