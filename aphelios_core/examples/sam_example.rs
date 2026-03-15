use anyhow::Result;

fn main() -> Result<()> {
    sam_image();
    sam_video();
    Ok(())
}

fn sam_image() {
    let image_path = "/Users/larry/coderesp/aphelios_cli/test_data/twocat.jpg";
    let promt = "cat";
}

fn sam_video() {
    let video = "/Users/larry/coderesp/aphelios_cli/test_data/dogvideo.mp4";
    let promt = "dog";
}
