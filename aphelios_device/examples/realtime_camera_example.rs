use aphelios_core::utils::{base::RTDETR_V4_M, logger};
use async_stream;
use axum::{
    body::Body,
    extract::State,
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use image::codecs::jpeg::JpegEncoder;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use std::io::Cursor;
use std::sync::Arc;
use tokio::sync::watch;
use tracing::info;
use usls::{Annotator, Config, Device, Image, Model, RTDETR};

struct AppState {
    frame_rx: watch::Receiver<Vec<u8>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    logger::init_logging();

    // 1. 初始化模型
    let model_path = RTDETR_V4_M;
    let config = Config::rtdetr_v4_m()
        .with_model_file(model_path)
        .with_class_confs(&[0.5])
        // .with_model_device(Device::CoreMl)
        // .with_coreml_compute_units_all(1) // 使用 GPU
        .commit()?;

    let (mut model, mut engines) = RTDETR::build(config)?;
    let annotator = Annotator::default();

    // 2. 使用 watch channel 传递最新帧。
    // watch 非常适合直播场景：消费者只关心“最新的一帧”，旧帧直接丢弃。
    let (frame_tx, frame_rx) = watch::channel(Vec::<u8>::new());

    // 3. 专用处理线程 (移动模型所有权，消除 Mutex)
    std::thread::spawn(move || {
        let mut camera = Camera::new(
            CameraIndex::Index(0),
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::HighestFrameRate(30)),
        )
        .expect("无法打开摄像头");

        camera.open_stream().expect("无法开启流");

        // 预分配缓冲区，减少 GC 压力
        let mut buffer = Vec::with_capacity(256 * 1024);

        loop {
            let frame = match camera.frame() {
                Ok(f) => f,
                Err(_) => continue,
            };

            let decoded = match frame.decode_image::<RgbFormat>() {
                Ok(img) => img,
                Err(_) => continue,
            };

            let usls_image = Image::from(decoded.clone());
            let mut dyn_img = image::DynamicImage::ImageRgb8(decoded);

            let start = std::time::Instant::now();
            // 执行推理 (此时 model 归此线程独有，无需 lock)
            if let Ok(ys) = model.run(&mut engines, &[usls_image.clone()]) {
                if !ys.is_empty() && !ys[0].hbbs().is_empty() {
                    if let Ok(annotated) = annotator.annotate(&usls_image, &ys[0]) {
                        dyn_img = annotated.into_dyn();
                    }
                }
            }
            info!("Inference time: {} ms", start.elapsed().as_millis());

            // JPEG 编码优化
            buffer.clear();
            {
                let mut cursor = Cursor::new(&mut buffer);
                // 质量 40-50 是 Web 流的平衡点
                let mut encoder = JpegEncoder::new_with_quality(&mut cursor, 45);
                if encoder.encode_image(&dyn_img).is_err() {
                    continue;
                }
            }

            // 更新最新帧。如果 Web 端来不及处理，旧帧会自动被覆盖，保证“丝滑”不堆积。
            let _ = frame_tx.send(buffer.clone());
        }
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
            "<html><body style='background:#111; text-align:center;'>
                <h1 style='color:white'>AI Vision Stream</h1>
                <img src='/stream' style='width:80%; border:2px solid #333;'>
            </body></html>",
        ))
        .unwrap()
}

async fn stream_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut rx = state.frame_rx.clone();

    let stream = async_stream::stream! {
        loop {
            // 等待新帧到达（不耗 CPU）
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

            // 构造 MJPEG 帧边界
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
