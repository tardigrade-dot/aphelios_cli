use anyhow::Result;
use aphelios_core::measure_time;
use aphelios_ocr::dolphin::model::{DolphinModel, run_ocr};
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<()> {
    let model_id = "/Volumes/sw/pretrained_models/Dolphin-v1.5";
    let mut dm = DolphinModel::load_model(model_id)?;

    let image_path = "/Users/larry/coderesp/aphelios_cli/test_data/page_32.png";
    let output_dir = "/Users/larry/coderesp/aphelios_cli/output/page_32";

    let res = measure_time!(dm.dolphin_ocr(&image_path, &output_dir).await);
    info!("执行完成, 打印结果....");
    match res {
        Ok(res) => {
            info!("test success {}", res.join(""));
        }
        Err(e) => {
            error!("Failed to load PDF page {}", e);
        }
    }
    Ok(())
}
