use anyhow::Result;
use aphelios_core::utils::core_utils;
use aphelios_ocr::paddleocr::run_paddleocr;
use tracing::{error, info};

#[test]
fn test_test() -> Result<()> {
    core_utils::init_logging();
    let ocr_result = run_paddleocr(vec![
        "/Users/larry/Documents/resources/page_18.png".to_string()
    ]);
    match ocr_result {
        Ok(_) => info!("run paddle ocr success!"),
        Err(e) => error!("{}", e),
    };
    Ok(())
}
