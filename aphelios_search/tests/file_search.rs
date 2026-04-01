use anyhow::Result;
use aphelios_core::utils::base::Error;
use aphelios_core::utils::{base, logger};
use candle_core::DType;
use fastembed::Qwen3TextEmbedding;
use std::path::Path;
use tracing::info;

#[test]
fn filename_search_init_test() -> Result<(), Error> {
    logger::init_logging();
    let dir_path = "/Volumes/sw/books";

    let dir_path = Path::new(dir_path);
    let _ = dir_path.try_exists().map_err(|e| Error::FileNotExists {
        msg: "path must exists",
    });
    if !dir_path.is_dir() {
        return Err(Error::PathMustDir { msg: "must a dir" });
    }

    let filename_list = dir_path
        .read_dir()
        .unwrap()
        .map(|f| f.unwrap().file_name().to_str().unwrap().to_string());

    for filename in filename_list {
        info!(filename)
    }
    Ok(())
}

#[test]
fn embed_test() -> Result<()> {
    let device = base::get_device();
    let model = Qwen3TextEmbedding::from_hf("Qwen/Qwen3-Embedding-0.6B", &device, DType::F32, 512)?;

    // Text-only usage with the Qwen3-VL embedding checkpoint is also supported:
    // let model = Qwen3TextEmbedding::from_hf("Qwen/Qwen3-VL-Embedding-2B", &device, DType::F32, 512)?;

    let embeddings = model.embed(&["query: ...", "passage: ..."])?;
    println!("Embeddings length: {}", embeddings.len());
    Ok(())
}
