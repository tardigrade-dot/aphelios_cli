use anyhow::{Error as E, Result};
use glob::glob;
use image::DynamicImage;
use regex::Regex;
use std::path::Path;
use std::path::PathBuf;
use std::{env, fs};
use tracing::info;

pub mod dolphin_utils;

const IGNORED_TAGS: &[&str] = &[
    "fig",       //图片
    "tab",       //表格
    "equ",       //公式
    "code",      //代码
    "header",    //页眉
    "foot",      //页脚
    "fnote",     //脚注
    "cap",       //图片说明
    "reference", //引用
];

pub async fn run_ocr(pdf_path: &str, output_path: &str) -> Result<()> {
    info!("start run dolphin ocr task");

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
    info!("start get page datas {}", output_path);
    let mut page_datas: Vec<String> = Vec::new();
    let re = Regex::new(r"^\[(?P<id>\d+)\]\s*-\s*\[(?P<tag>[^\]]+)\]\s*:\s*(?P<content>.*)$").unwrap();

    let mut last_label = String::new();

    // Collect all matching paths and sort them by numeric prefix
    let mut paths: Vec<_> = glob(&format!("{}/[0-9]*_page.txt", output_path))?.collect::<Result<Vec<_>, _>>()?;
    paths.sort_by_key(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.split('_').next())
            .and_then(|num| num.parse::<u32>().ok())
            .unwrap_or(0)
    });

    for path in paths {
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
                if IGNORED_TAGS.contains(&item_tag.as_str()) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use aphelios_core::init_logging;
    use tracing::{error, info};

    #[test]
    fn dolphin_all_in_one_test() -> Result<()> {
        init_logging();
        let output_dir = "/Volumes/sw/ocr_result/专制权力与中国社会 (刘泽华) (z-library.sk, 1lib.sk, z-lib.sk)";

        let result = full_in_one(output_dir);
        match result {
            Ok(_) => {
                info!("test success ");
            }
            Err(e) => error!("{:?}", e),
        }
        Ok(())
    }
}
