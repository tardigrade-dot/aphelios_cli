use anyhow::Result;
use aphelios_core::utils::logger;
use aphelios_ocr::paddleocr::run_paddleocr;
use tracing::{error, info};

#[test]
fn test_test() -> Result<()> {
    logger::init_logging();
    let ocr_result = run_paddleocr(vec![
        "/Users/larry/Documents/resources/page_18.png".to_string()
    ]);
    match ocr_result {
        Ok(_) => info!("run paddle ocr success!"),
        Err(e) => error!("{}", e),
    };
    Ok(())
}
