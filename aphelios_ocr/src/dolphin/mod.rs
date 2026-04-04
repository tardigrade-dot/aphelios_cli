use tracing::info;

use anyhow::{Error as E, Result};
use glob::glob;
use regex::Regex;
use std::path::Path;
use std::path::PathBuf;
use std::result::Result::Ok;
use std::{env, fs};

use crate::dolphin::model::DolphinModel;

pub mod dolphin_utils;
pub mod donut;
pub mod model;

pub async fn run_ocr(pdf_path: &str, output_path: &str) -> Result<()> {
    info!("start run dolphin ocr task");

    // Allow model path to be configurable via environment variable or use default
    let model_id = env::var("DOLPHIN_MODEL_PATH")
        .unwrap_or_else(|_| "/Volumes/sw/pretrained_models/Dolphin-v1.5".to_string());

    let mut dm = DolphinModel::load_model(&model_id)?;

    let _ = &dm
        .dolphin_ocr(&pdf_path.to_string(), &output_path.to_string())
        .await?;
    info!("ocr finished. start merge all file to single one");

    full_in_one(output_path)?;
    Ok(())
}

fn full_in_one(output_path: &str) -> Result<(), E> {
    let page_datas = get_page_datas(output_path)?;
    let output = Path::new(output_path);
    let output_file: PathBuf = output.join("total_in_one.txt");
    fs::write(&output_file, format!("{}", &page_datas.join("\n")))?;
    info!("all text saved in {}", &output_file.to_str().unwrap());
    Ok(())
}

fn get_page_datas(output_path: &str) -> Result<Vec<String>> {
    let mut page_datas: Vec<String> = Vec::new();
    let re =
        Regex::new(r"^\[(?P<id>\d+)\]\s*-\s*\[(?P<tag>[^\]]+)\]\s*:\s*(?P<content>.*)$").unwrap();

    let mut last_label = String::new();

    for entry in glob(&format!("{}/[0-9]*_page.txt", output_path))? {
        let path = entry?;
        // 读取文件内容
        let content = fs::read_to_string(&path)?;
        let mut i_index = 0;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Some(caps) = re.captures(line) {
                let mut item_tag = caps["tag"].to_string();
                let item_content = caps["content"].trim().to_string();

                // 过滤掉不需要的标签
                if ["fig", "tab", "equ", "code", "header", "foot", "fnote"]
                    .contains(&item_tag.as_str())
                {
                    continue;
                }
                // half_para 统一为 para
                if item_tag == "half_para" {
                    item_tag = "para".to_string();
                }

                // 如果同标签且是本文件第一条，追加到上一条
                if item_tag == last_label && i_index == 0 {
                    if let Some(last) = page_datas.last_mut() {
                        last.push_str(&item_content);
                    }
                } else {
                    page_datas.push(item_content.to_string());
                }

                last_label = item_tag;
                i_index += 1;
            }
        }
    }

    Ok(page_datas)
}
