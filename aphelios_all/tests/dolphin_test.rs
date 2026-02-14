use anyhow::Result;
use aphelios_cli::{dolphin::model::DolphinModel, measure_time};
use tracing::{error, info};

#[tokio::test]
async fn dolphin_test() -> Result<()> {
    measure_time! {
        let model_id = "/Volumes/sw/pretrained_models/Dolphin-v1.5";
        let mut dm = DolphinModel::load_model(model_id)?;

        let image_path = "/Users/larry/Documents/resources/page_32.png";
        let output_dir = "/Users/larry/coderesp/aphelios_cli/output/aaaa_test";

        let res = dm.dolphin_ocr(&image_path, &output_dir).await;
        info!("执行完成, 打印结果....");
        match res {
            Ok(res) => {
                info!("test success {}", res.join(""));
            }
            Err(e) => {
                error!("Failed to load PDF page {}",  e);
            }
        }
    }
    Ok(())
}

#[tokio::test]
async fn dolphin_pdf_test() -> Result<()> {
    measure_time! {
        let model_id = "/Volumes/sw/pretrained_models/Dolphin-v1.5";
        let mut dm = DolphinModel::load_model(model_id)?;
        let image_path = "/Users/larry/github.com/colab-script2/data_src/extracted_pages.pdf";
        // let image_path = "/Users/larry/Documents/resources/page_32.png";
        let output_dir = "/Users/larry/coderesp/aphelios_cli/output/extracted_pages";

        let res = dm.dolphin_ocr(&image_path.to_string(), &output_dir.to_string()).await;
        info!("执行完成, 打印结果....");
        match res {
            Ok(res) => {
                info!("test success {}", res.join(" "));
            }
            Err(e) => {
                error!("Failed to load PDF page {}", e);
            }
        }
    }
    Ok(())
}
