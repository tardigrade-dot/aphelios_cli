use anyhow::Result;
use aphelios_core::measure_time;
use aphelios_ocr::dolphin::model::DolphinModel;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<()> {
    // 性能提示: 
    // 1. 确保使用 --release 模式运行
    // 2. 对于 Apple Silicon (M1/M2/M3/M4), 使用 --features metal 开启 GPU 加速
    // 3. 当前已开启并行预处理和 Batch 推理 (Batch Size = 4)
    let model_id = "/Volumes/sw/pretrained_models/Dolphin-v1.5";
    let mut dm = DolphinModel::load_model(model_id)?;

    let image_path = "/Users/larry/coderesp/aphelios_cli/test_data/page_32.png";
    let output_dir = "/Users/larry/coderesp/aphelios_cli/output/page_32";

    // let image_path = "/Users/larry/coderesp/aphelios_cli/test_data/extracted_pages.pdf";
    // let output_dir = "/Users/larry/coderesp/aphelios_cli/output/extracted_pages";

    let res = measure_time!(dm.dolphin_ocr(&image_path, &output_dir).await);
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
