use anyhow::Result;
use aphelios_cli::{dolphin::{self}, measure_time};
use tracing::info;
use aphelios_cli::commands::dolphin::run_ocr;

#[tokio::test]
async fn dolphin_test2() -> Result<()> {

    // '/Volumes/sw/books/我们的敌人：国家 (伯特·杰伊·诺克) (Z-Library).pdf' /Volumes/sw/ocr_result/test
    let r = run_ocr("/Volumes/sw/books/我们的敌人：国家 (伯特·杰伊·诺克) (Z-Library).pdf", "/Volumes/sw/ocr_result/test")
        .await?;
    Ok(())
}

#[test]
fn dolphin_test() -> Result<()> {

    measure_time!{
        let model_id = "/Volumes/sw/pretrained_models/Dolphin-v1.5";
        let image_path = "/Users/larry/Documents/resources/page_32.png"; // 替换为你的测试图路径
        let output_dir = "/Volumes/sw/ocr_result/aaaa_test";
    
        let res = dolphin::model::dolphin_ocr(&model_id.to_string(), &image_path.to_string(), &output_dir.to_string())?;
        
        for r in &res{
            info!("{}", r);
        }
    }
    Ok(())
}

#[test]
fn dolphin_pdf_test() -> Result<()>{

    measure_time!{
        let model_id = "/Volumes/sw/pretrained_models/Dolphin-v1.5";
        let image_path = "/Users/larry/github.com/tardigrade-dot/colab-script2/data_src/extracted_pages.pdf";
        let output_dir = "/Users/larry/coderesp/aphelios_cli/output";
    
        let res = dolphin::model::dolphin_ocr(&model_id.to_string(), &image_path.to_string(), &output_dir.to_string())?;
        for r in &res{
            info!("{}", r);
        }
    }
    Ok(())
}