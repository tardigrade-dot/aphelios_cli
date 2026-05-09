use anyhow::Result;
use aphelios_core::{utils::logger::init_test_logging, yt_dlp_downloader::download_with_progress};

pub fn main() -> Result<(), String> {
    init_test_logging();
    let video_url = "https://www.youtube.com/watch?v=lyWo9cQ0dY8";
    let output_path = "/Users/larry/coderesp/aphelios_cli/output/download";
    let _ = download_with_progress(video_url, output_path)?;
    Ok(())
}
