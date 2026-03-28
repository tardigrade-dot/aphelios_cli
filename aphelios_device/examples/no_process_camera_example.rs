use anyhow::Result;
use aphelios_core::utils::{base, logger};
use async_stream;
use axum::{
    body::Body, extract::State, response::IntoResponse, response::Response, routing::get, Router,
};
use image::codecs::jpeg::JpegEncoder;
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{
        CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution,
    },
    Camera,
};
use std::io::Cursor;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::watch;
use tracing::info;

struct AppState {
    frame_rx: watch::Receiver<Vec<u8>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    logger::init_logging();

    // 使用 watch channel 传递最新帧
    let (frame_tx, frame_rx) = watch::channel(Vec::<u8>::new());

    // 摄像头线程 - 使用 spawn_blocking
    tokio::spawn(async move {
        tokio::task::spawn_blocking(move || {
            let format = CameraFormat::new(Resolution::new(640, 480), FrameFormat::MJPEG, 30);
            let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Exact(format));

            let mut camera =
                Camera::new(CameraIndex::Index(0), requested).expect("camera open failed");
            camera.open_stream().expect("stream open failed");

            info!(
                "Camera: {}x{} @ {}fps",
                camera.camera_format().resolution().width(),
                camera.camera_format().resolution().height(),
                camera.camera_format().frame_rate(),
            );

            let mut buffer = Vec::with_capacity(256 * 1024);
            let mut count = 0u64;
            let start = Instant::now();

            loop {
                let frame = match camera.frame() {
                    Ok(f) => f,
                    Err(_) => continue,
                };
                let rgb = match frame.decode_image::<RgbFormat>() {
                    Ok(img) => img,
                    Err(_) => continue,
                };

                buffer.clear();
                {
                    let mut cursor = Cursor::new(&mut buffer);
                    let mut encoder = JpegEncoder::new_with_quality(&mut cursor, 30);
                    if encoder.encode_image(&rgb).is_err() {
                        continue;
                    }
                }

                count += 1;
                if count % 100 == 0 {
                    info!(
                        "Encoded {} frames, rate: {:.1} fps, size: {} bytes",
                        count,
                        count as f64 / start.elapsed().as_secs_f64(),
                        buffer.len()
                    );
                }

                let _ = frame_tx.send(buffer.clone());
                std::thread::sleep(std::time::Duration::from_millis(33));
            }
        });
    });

    let state = Arc::new(AppState { frame_rx });
    let app = Router::new()
        .route("/", get(index_handler))
        .route("/stream", get(stream_handler))
        .with_state(state);

    let addr = "0.0.0.0:3000";
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("服务运行在 http://{}", addr);
    axum::serve(listener, app).await?;

    Ok(())
}

async fn index_handler() -> Response {
    Response::builder()
        .header("Content-Type", "text/html")
        .body(Body::from(
            r#"
<html>
<body style="background:#111;color:white;text-align:center;">
    <h2>Camera Stream - No Processing</h2>
    <img src="/stream" style="width:80%;border:2px solid #333;">
</body>
</html>
"#,
        ))
        .unwrap()
}

async fn stream_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut rx = state.frame_rx.clone();

    let stream = async_stream::stream! {
        loop {
            if rx.changed().await.is_err() {
                break;
            }

            let frame = {
                let val = rx.borrow();
                val.clone()
            };

            if frame.is_empty() {
                continue;
            }

            let header = format!(
                "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                frame.len()
            );

            let mut chunk = Vec::with_capacity(header.len() + frame.len() + 2);
            chunk.extend_from_slice(header.as_bytes());
            chunk.extend_from_slice(&frame);
            chunk.extend_from_slice(b"\r\n");

            yield Ok::<_, std::convert::Infallible>(chunk);
        }
    };

    Response::builder()
        .header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        .body(Body::from_stream(stream))
        .unwrap()
}
