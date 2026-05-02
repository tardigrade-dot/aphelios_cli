use std::path::Path;

use anyhow::Result;
use aphelios_core::{
    init_logging, measure_time,
    utils::{self},
};
use aphelios_ocr::dolphin::model::DolphinModel;
use aphelios_ocr::dolphin::run_ocr;
use tracing::{error, info};

const TEN_PAGES_PDF: &str = "/Users/larry/Downloads/test_pdf.pdf";
const SMALL_PDF: &str = "/Users/larry/coderesp/aphelios_cli/test_data/extracted_pages.pdf";
const SINGLE_IMAGE: &str = "/Volumes/sw/MyDrive/data_src/page_zht_49.png";

// samply record cargo test --package aphelios_ocr --test dolphin_test --features metal --features profiling -- dolphin_single_image_test
// dolphin 4.9s 5.2s 1.67s 4.7s 3s 1.5s  total:20s
// GLM OCR 9.96s 9.16s 0.67s 10.28s 5.49s 0.20s total:35s
#[tokio::test]
// #[ignore = "this is a slow test"]
async fn dolphin_single_image_test() -> Result<()> {
    utils::init_logging();
    let model_id = "/Volumes/sw/pretrained_models/Dolphin-v1.5";
    // let model_id = "ByteDance/Dolphin-1.5";
    let mut dm = measure_time!("load_model", DolphinModel::load_model(model_id)?);
    let output_dir = "/Users/larry/coderesp/aphelios_cli/output/3";

    info!("output_dir: {}", output_dir);
    if Path::new(output_dir).exists() {
        std::fs::remove_dir_all(output_dir).unwrap();
    }
    let res = measure_time!(
        "dolphin_single_image_test",
        dm.dolphin_ocr(&SINGLE_IMAGE.to_string(), &output_dir.to_string(), None)
            .await
    );
    info!("执行完成, 打印结果....");
    match res {
        Ok(res) => {
            info!("test success {}", res.join("\n"));
        }
        Err(e) => {
            error!("Failed to load PDF page {}", e);
        }
    }
    Ok(())
}

#[tokio::test]
async fn dolphin_pdf_test() -> Result<()> {
    init_logging();
    let test_data = TEN_PAGES_PDF;

    let model_id = "/Volumes/sw/pretrained_models/Dolphin-v1.5";

    let mut dolphin_model = measure_time!("load_model", DolphinModel::load_model(model_id)?);

    let output_dir = "/Users/larry/coderesp/aphelios_cli/output/10";

    let target_dir = Path::new(output_dir).join(Path::new(test_data).file_stem().unwrap());
    if target_dir.exists() {
        std::fs::remove_dir_all(target_dir).unwrap();
    }
    let res = measure_time!(
        "dolphin_ocr",
        dolphin_model
            .dolphin_ocr(&test_data.to_string(), &output_dir.to_string(), None)
            .await
    );
    info!("执行完成, 打印结果....");
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
    let output_dir = "/Users/larry/coderesp/aphelios_cli/output/extracted_pages/3";

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
