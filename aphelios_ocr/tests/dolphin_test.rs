use std::path::Path;

use anyhow::Result;
use aphelios_core::utils::{self, logger};
use aphelios_ocr::dolphin::model::DolphinModel;
use aphelios_ocr::dolphin::run_ocr;
use tracing::{error, info};

const TEN_PAGES_PDF: &str = "/Users/larry/Downloads/test_pdf.pdf";
const SMALL_PDF: &str = "/Users/larry/coderesp/aphelios_cli/test_data/extracted_pages.pdf";
const SINGLE_IMAGE: &str = "/Users/larry/coderesp/aphelios_cli/test_data/page_32.png";

#[tokio::test]
async fn dolphin_single_image_test() -> Result<()> {
    #[cfg(feature = "profiling")]
    let guard = logger::init_chrome_logging();

    let model_id = "/Volumes/sw/pretrained_models/Dolphin-v1.5";
    // let model_id = "ByteDance/Dolphin-1.5";
    let mut dm = DolphinModel::load_model(model_id)?;

    let output_dir = "/Users/larry/coderesp/aphelios_cli/output/3";

    info!("output_dir: {}", output_dir);
    if Path::new(output_dir).exists() {
        std::fs::remove_dir_all(output_dir).unwrap();
    }
    let res = dm
        .dolphin_ocr(&SINGLE_IMAGE.to_string(), &output_dir.to_string())
        .await;
    info!("执行完成, 打印结果....");
    match res {
        Ok(res) => {
            info!("test success {}", res.join("\n"));
        }
        Err(e) => {
            error!("Failed to load PDF page {}", e);
        }
    }

    #[cfg(feature = "profiling")]
    drop(guard);
    Ok(())
}

#[tokio::test]
async fn dolphin_pdf_test() -> Result<()> {
    let test_data = TEN_PAGES_PDF;

    #[cfg(feature = "profiling")]
    let guard = logger::init_chrome_logging();

    let model_id = "/Volumes/sw/pretrained_models/Dolphin-v1.5";

    #[cfg(feature = "profiling")]
    let _load_model_span = tracing::info_span!("load_model").entered();
    let mut dolphin_model = DolphinModel::load_model(model_id)?;

    #[cfg(feature = "profiling")]
    _load_model_span.exit();

    let output_dir = "/Users/larry/coderesp/aphelios_cli/output/10";

    let target_dir = Path::new(output_dir).join(Path::new(test_data).file_stem().unwrap());
    if target_dir.exists() {
        std::fs::remove_dir_all(target_dir).unwrap();
    }
    #[cfg(feature = "profiling")]
    let _dolphin_ocr_span = tracing::info_span!("dolphin_ocr").entered();
    let res = dolphin_model
        .dolphin_ocr(&test_data.to_string(), &output_dir.to_string())
        .await;
    #[cfg(feature = "profiling")]
    _dolphin_ocr_span.exit();
    info!("执行完成, 打印结果....");
    #[cfg(feature = "profiling")]
    drop(guard);
    match res {
        Ok(res) => {
            info!("test success {}", res.join(" "));
            Ok(())
        }
        Err(e) => {
            error!("Failed to load PDF page {}", e);
            Err(e)
        }
    }
}

#[tokio::test]
async fn dolphin_test2() -> Result<()> {
    utils::init_logging();
    let output_dir = "/Users/larry/coderesp/aphelios_cli/output/extracted_pages/2";

    if Path::new(output_dir).exists() {
        std::fs::remove_dir_all(output_dir).unwrap();
    }
    let res = run_ocr(SMALL_PDF, output_dir).await;
    match res {
        Ok(_) => {
            info!("test success ");
        }
        Err(e) => error!("{:?}", e),
    }
    Ok(())
}
