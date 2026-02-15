use std::collections::HashMap;

use anyhow::{Error as E, Result};
use aphelios_cli::commands::minersu::{self, preprocess_to_image};
use aphelios_core::common;
use candle_core::{backend::BackendDevice, MetalDevice};
use mistralrs::{Device, IsqType, Model, VisionModelBuilder};
use once_cell::sync::Lazy;
use regex::Regex;
use tracing::{info, Level};

static DEFAULT_PROMPTS: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    HashMap::from([
        ("table", "\nTable Recognition:"),
        ("equation", "\nFormula Recognition:"),
        ("[default]", "\nText Recognition:"),
        ("[layout]", "\nLayout Detection:"),
    ])
});

#[derive(Debug)]
pub struct LayoutBox {
    pub x1: u32,
    pub y1: u32,
    pub x2: u32,
    pub y2: u32,
    pub label: String,
    pub extra: String,
}
// '<|box_start|>132 068 859 267<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>\n<|box_start|>134 272 861 473<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>\n<|box_start|>279 511 664 540<|box_end|><|ref_start|>title<|ref_end|><|rotate_up|>\n<|box_start|>138 567 864 766<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>\n<|box_start|>139 773 864 857<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>\n<|box_start|>172 869 251 884<|box_end|><|ref_start|>page_number<|ref_end|><|rotate_up|>'

#[tokio::test]
async fn stage2_test() -> Result<()> {
    let line = "<|box_start|>152 089 862 108<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>
<|box_start|>152 113 862 132<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>
<|box_start|>152 136 862 156<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>
<|box_start|>152 160 862 179<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>
<|box_start|>152 183 862 202<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>";

    let test_str = String::from(line);

    let model: Model = get_mineru_model().await?;
    let _ = ocr_stage_second(&model, test_str).await;
    Ok(())
}

async fn get_mineru_model() -> Result<Model> {
    let model_id = "/Users/larry/test_dir/MinerU2.5-2509-1.2B";

    let metal: MetalDevice = MetalDevice::new(0)?;
    let model = VisionModelBuilder::new(model_id)
        // .with_isq(IsqType::Q4K)
        .with_device(Device::Metal(metal))
        .with_dtype(mistralrs::ModelDType::F16)
        .with_logging()
        .build()
        .await?;
    Ok(model)
}

async fn ocr_stage_second(model: &Model, layout_str: String) -> Result<()> {
    let re = Regex::new(
        r"^<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(.*)$"
    ).unwrap();

    let image = image::ImageReader::open("/Users/larry/Documents/resources/page_32.png")?
        .decode()
        .map_err(|e| E::msg(format!("Failed to decode image: {}", e)))?;

    let image2 = preprocess_to_image(image);

    let mut out: Vec<LayoutBox> = Vec::new();
    let lines = layout_str.split("\n");

    for i_line in lines {
        if let Some(cap) = re.captures(i_line.trim()) {
            let b = LayoutBox {
                x1: cap[1].parse().unwrap(),
                y1: cap[2].parse().unwrap(),
                x2: cap[3].parse().unwrap(),
                y2: cap[4].parse().unwrap(),
                label: cap[5].to_string(),
                extra: cap[6].to_string(),
            };

            out.push(b);
        }
    }
    dbg!(&out);
    info!("ocr second stage start ...");
    for (i, item) in out.iter().enumerate() {
        // let prompt = DEFAULT_PROMPTS.get(item.label.as_str());
        let prompt = DEFAULT_PROMPTS
            .get(item.label.as_str()) // 动态 key → Option
            .unwrap_or(&DEFAULT_PROMPTS["[default]"]);

        let i1 = common::core_utils::crop_image(&image2, [item.x1, item.x2, item.y1, item.y2], 5);
        let _ = i1.save(format!("./debug_{}.png", i));

        let res = minersu::run_mineru2(&model, i1, &prompt.to_string()).await?;

        info!(res)
    }

    Ok(())
}

#[test]
fn pic_cap() -> Result<()> {
    let image = image::ImageReader::open("/Users/larry/Documents/resources/page_32.png")?
        .decode()
        .map_err(|e| E::msg(format!("Failed to decode image: {}", e)))?;

    let image2 = preprocess_to_image(image);
    let i1 = common::core_utils::crop_image(&image2, [152, 113, 862, 132], 5);
    let _ = i1.save("./aaa.png");

    Ok(())
}

#[tokio::test]
async fn mineru_ocr_test() -> Result<()> {
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::TRACE)
        .finish();
    let _ = tracing::subscriber::set_default(subscriber);
    let layout = "<|box_start|>133 143 868 177<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>";
    let model_id = "/Users/larry/test_dir/MinerU2.5-2509-1.2B";
    // let _ = qwenvl::run_mineru2(model_id, layout).await;
    Ok(())
}

#[tokio::test]
async fn mineru_test() -> Result<()> {
    let model = get_mineru_model().await?;

    let image = image::ImageReader::open("/Users/larry/Documents/resources/page_32.png")?
        .decode()
        .map_err(|e| E::msg(format!("Failed to decode image: {}", e)))?;

    // let image2: image::DynamicImage = preprocess_to_image(image);
    // let image3: = utils::load_image(
    //     "/Users/larry/Documents/resources/page_32.png", target_height, target_width, device)

    // image2.save("./page_32.png");
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::TRACE)
        .finish();
    let _ = tracing::subscriber::set_default(subscriber);

    let layout_prompt = "\nLayout Detection:";

    let layout_str = minersu::run_mineru(&model, image, &layout_prompt.to_string()).await?;
    info!("first stage : {}", layout_str);

    // let _ = ocr_stage_second(&model, layout_str).await;
    Ok(())
}
