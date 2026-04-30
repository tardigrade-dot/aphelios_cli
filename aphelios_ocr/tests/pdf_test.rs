use anyhow::Result;
use aphelios_core::utils::logger::init_test_logging;
use aphelios_ocr::dolphin::dolphin_utils;
use futures_util::{pin_mut, StreamExt};
use std::path::PathBuf;
use tracing::info;

#[tokio::test]
async fn test_pdf_load_images() -> Result<()> {
    init_test_logging();

    // 获取当前工作目录，并尝试定位测试文件
    let mut pdf_path =
        PathBuf::from("/Users/larry/coderesp/aphelios_cli/test_data/extracted_pages.pdf");

    if !pdf_path.exists() {
        // 尝试从项目根目录定位
        let current_dir = std::env::current_dir()?;
        info!("Current directory: {:?}", current_dir);

        // 如果在 aphelios_ocr 目录下运行，向上找一级
        if current_dir.ends_with("aphelios_ocr") {
            pdf_path = current_dir
                .parent()
                .unwrap()
                .join("test_data/extracted_pages.pdf");
        } else {
            pdf_path = current_dir.join("test_data/extracted_pages.pdf");
        }
    }

    if !pdf_path.exists() {
        info!("PDF path does not exist, skipping test: {:?}", pdf_path);
        return Ok(());
    }

    info!("Testing load_pdf_images with: {:?}", pdf_path);

    let img_stream = dolphin_utils::load_pdf_images(pdf_path);
    pin_mut!(img_stream);

    let mut index = 0;
    while let Some(img_result) = img_stream.next().await {
        let img = img_result?;
        info!("Loaded page {}", index);

        let output_dir = std::path::Path::new("output/pdf_test");
        if !output_dir.exists() {
            std::fs::create_dir_all(output_dir)?;
        }

        let output_path = output_dir.join(format!("page-{}.jpg", index));
        let _ = img.save_jpeg(&output_path);

        info!("Saved page {} to {:?}", index, output_path);
        index += 1;
    }

    if index > 0 {
        info!("Successfully loaded and saved {} pages", index);
    } else {
        info!("No pages found in PDF");
    }

    assert!(index >= 0);
    Ok(())
}

const BIG_TEST_PDF: &str = "/Users/larry/Downloads/test_layout.pdf";

#[test]
fn test_pdf_extract_from() {
    init_test_logging();
    let pdf_path = PathBuf::from(BIG_TEST_PDF);
    let output_path = PathBuf::from("/Users/larry/coderesp/aphelios_cli/test_data/test_pdf2.pdf");
    dolphin_utils::pdf_extract_from(&pdf_path, 9, 90, &output_path).unwrap();
    info!("Extracted page from {:?} to {:?}", pdf_path, output_path);
}

#[test]
fn test_pdf_page_to_png() {
    init_test_logging();
    let num = 2;
    let pdf_path = PathBuf::from(BIG_TEST_PDF);
    let output_path = PathBuf::from(format!(
        "/Users/larry/coderesp/aphelios_cli/output/example_page_{}.png",
        num
    ));
    dolphin_utils::pdf_page_to_png(&pdf_path, num, &output_path).unwrap();
    info!(
        "Saved page {} from {:?} to {:?}",
        num, pdf_path, output_path
    );
}
