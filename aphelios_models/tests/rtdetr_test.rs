use anyhow::Result;
use aphelios_core::{
    imglabel::{label_images, label_video},
    measure_time,
    utils::logger,
};

#[test]
fn label_video_test() -> Result<()> {
    logger::init_logging();
    let video_path = "/Users/larry/coderesp/aphelios_cli/test_data/dogvideo.mp4";
    let _ = measure_time!("label_video", label_video(video_path)).unwrap();
    Ok(())
}

#[test]
fn label_images_test() -> Result<()> {
    logger::init_logging();
    let image_path = "/Users/larry/coderesp/aphelios_cli/test_data/twocats.jpg";
    let _ = measure_time!("label_images", label_images(image_path));
    Ok(())
}
