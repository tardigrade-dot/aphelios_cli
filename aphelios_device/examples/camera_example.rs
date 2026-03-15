use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};
fn main() {
    let index = CameraIndex::Index(0);
    let requested =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

    let mut camera = Camera::new(index, requested).unwrap();

    let format = camera.camera_format();
    println!("Camera format: {:?}", format);

    camera.open_stream().unwrap();

    // Warm up - let camera stabilize
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Capture frames until we get a good decoded image
    for i in 0..30 {
        let frame = camera.frame().unwrap();

        match frame.decode_image::<RgbFormat>() {
            Ok(decoded) => {
                let decoded_sum: u64 = decoded.as_raw().iter().map(|&b| b as u64).sum();
                let decoded_avg = decoded_sum / decoded.as_raw().len() as u64;

                println!("Frame {}: avg={}", i, decoded_avg);

                if decoded_avg > 5 {
                    decoded.save("frame.png").unwrap();
                    println!("Saved frame.png ({}x{})", decoded.width(), decoded.height());
                    return;
                }
            }
            Err(e) => println!("Frame {}: decode error: {}", i, e),
        }
    }

    eprintln!("Failed to capture a good frame");
}
