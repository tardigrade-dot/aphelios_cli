use anyhow::Result;
use aphelios_core::yt_dlp_downloader::download_with_progress;

#[test]
fn dlp_test() -> Result<()>{

    let video_url = "https://www.youtube.com/watch?v=OefgrhwaeOI";
    let output_path = "/Users/larry/coderesp/aphelios_cli/output/download";
    if let Err(e) = download_with_progress(video_url, output_path) {
        eprintln!("{}", e);
        std::process::exit(1);
    }
    Ok(())
}
