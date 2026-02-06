use std::env;
use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::dolphin::model::dolphin_ocr;
use anyhow::Result;
use tracing::info;

pub fn run_ocr(pdf_path: &str, output_path: &str) -> Result<()> {
    info!("start run dolphin ocr task");

    // Allow model path to be configurable via environment variable or use default
    let model_id = env::var("DOLPHIN_MODEL_PATH")
        .unwrap_or_else(|_| "/Volumes/sw/pretrained_models/Dolphin-v1.5".to_string());

    let r = dolphin_ocr(&model_id, &pdf_path.to_string(), &output_path.to_string())?;
    println!("finished");

    let output = Path::new(output_path);
    let output_file: PathBuf = output.join("total.txt");
    fs::write(&output_file, format!("{}", &r.join("\n")))?;
    Ok(())
}
