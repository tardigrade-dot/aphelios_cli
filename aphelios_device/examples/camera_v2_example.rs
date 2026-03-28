//! camera-stream — smooth browser-based camera viewer with AI hook
//!
//! Thread layout:
//!   [OS thread]  capture_loop   — nokhwa AVFoundation, blocking
//!       │  mpsc::sync_channel<RawFrame>(bound=2)   ← backpressure
//!   [OS thread]  encode_loop    — mozjpeg SIMD encode + process_frame hook
//!       │  watch::channel<Arc<Bytes>>              ← latest JPEG, N readers
//!   [Tokio]      axum handlers  — MJPEG multipart push to browser clients

use axum::body::Body;
use axum::{
    extract::State,
    http::header,
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use bytes::Bytes;
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};
use std::{
    sync::{mpsc, Arc},
    thread,
    time::Duration,
};
use tokio::sync::watch;
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// A raw RGB frame off the camera.
struct RawFrame {
    data: Vec<u8>,
    width: u32,
    height: u32,
}

/// Detection result placeholder — expand this when you wire in your model.
#[allow(dead_code)]
pub struct Detection {
    pub label: String,
    pub confidence: f32,
    /// Bounding box: (x, y, width, height) in pixels
    pub bbox: (u32, u32, u32, u32),
}

/// Shared axum state: a receiver for the latest JPEG bytes.
#[derive(Clone)]
struct AppState {
    jpeg_rx: watch::Receiver<Arc<Bytes>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// AI hook — put your inference logic here
// ─────────────────────────────────────────────────────────────────────────────

/// Process a raw RGB frame before it is JPEG-encoded.
///
/// This is intentionally a plain function (not async) so it runs
/// synchronously on the encode thread — the same place where you have
/// mutable access to the pixel buffer.  When you add a real model:
///
///   1. Load your ONNX session / candle weights *once* outside this
///      function (e.g. in `encode_loop`) and pass it in.
///   2. Resize + normalise `frame.data` into your model's input tensor.
///   3. Run inference, decode bounding boxes.
///   4. Call `draw_detections` (or imageproc) to paint boxes onto the buffer.
///   5. Return the detections so callers can log / forward them.
///
/// Right now it is a pure no-op: returns immediately without touching pixels.
fn process_frame(_frame: &mut RawFrame) -> Vec<Detection> {
    // ── future: AI inference goes here ──────────────────────────────────────
    //
    // let input = preprocess(&frame.data, frame.width, frame.height);
    // let outputs = session.run(inputs!["images" => input]?)?;
    // let detections = decode_yolo_output(&outputs, frame.width, frame.height);
    // draw_boxes(&mut frame.data, frame.width, &detections);
    // return detections;
    //
    // ────────────────────────────────────────────────────────────────────────
    vec![]
}

// ─────────────────────────────────────────────────────────────────────────────
// Capture thread — dedicated OS thread, never touches Tokio
// ─────────────────────────────────────────────────────────────────────────────

fn capture_loop(tx: mpsc::SyncSender<RawFrame>) {
    // AVFoundation index 0 = default camera on macOS.
    // RequestedFormatType::AbsoluteHighestFrameRate asks the driver for its
    // best fps; you can swap in a fixed CameraFormat if you need a specific
    // resolution.
    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

    let mut camera = match Camera::new(CameraIndex::Index(0), format) {
        Ok(c) => c,
        Err(e) => {
            error!("Failed to open camera: {e}");
            return;
        }
    };

    if let Err(e) = camera.open_stream() {
        error!("Failed to open camera stream: {e}");
        return;
    }

    let info = camera.info().clone();
    info!(
        "Camera opened: {} ({}x{})",
        info.human_name(),
        camera.resolution().width(),
        camera.resolution().height(),
    );

    loop {
        match camera.frame() {
            Ok(frame) => {
                let resolution = frame.resolution();
                match frame.decode_image::<RgbFormat>() {
                    Ok(img) => {
                        let raw = RawFrame {
                            data: img.into_raw(),
                            width: resolution.width(),
                            height: resolution.height(),
                        };
                        // try_send: if the encode thread is busy, drop this
                        // frame rather than piling up memory.
                        if tx.try_send(raw).is_err() {
                            // encode thread is behind; just skip this frame
                        }
                    }
                    Err(e) => warn!("Frame decode error: {e}"),
                }
            }
            Err(e) => {
                warn!("Camera frame error: {e}");
                thread::sleep(Duration::from_millis(5));
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Encode thread — CPU-heavy JPEG work, isolated from capture and axum
// ─────────────────────────────────────────────────────────────────────────────

fn encode_loop(rx: mpsc::Receiver<RawFrame>, tx: watch::Sender<Arc<Bytes>>) {
    // JPEG quality 78 hits the sweet spot on M4:
    // visually indistinguishable from 90+ in a live stream,
    // encodes ~2× faster.
    const JPEG_QUALITY: u8 = 78;

    for mut frame in rx {
        // ── AI hook ─────────────────────────────────────────────────────────
        let _detections = process_frame(&mut frame);
        // ────────────────────────────────────────────────────────────────────

        // mozjpeg: SIMD-accelerated on Apple Silicon (ARM NEON)
        let mut comp = mozjpeg::Compress::new(mozjpeg::ColorSpace::JCS_RGB);
        comp.set_size(frame.width as usize, frame.height as usize);
        comp.set_quality(JPEG_QUALITY as f32);
        // Progressive encoding improves perceived quality in browsers
        comp.set_progressive_mode();
        comp.set_optimize_coding(false); // faster on M4

        let mut started = match comp.start_compress(Vec::new()) {
            Ok(s) => s,
            Err(e) => {
                warn!("JPEG compress start error: {e}");
                continue;
            }
        };
        let _ = started.write_scanlines(&frame.data);
        let jpeg = started.finish();

        match jpeg {
            Ok(data) => {
                // Arc means all connected clients share the same allocation
                let _ = tx.send(Arc::new(Bytes::from(data)));
            }
            Err(e) => warn!("JPEG encode error: {e}"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP handlers
// ─────────────────────────────────────────────────────────────────────────────

/// MJPEG stream endpoint — browsers can display this directly with <img>.
///
/// Each connected client gets its own watch::Receiver clone; they all read
/// from the same underlying Arc<Bytes> without copying.
async fn mjpeg_handler(State(state): State<AppState>) -> Response {
    let mut rx = state.jpeg_rx.clone();

    let stream = async_stream::stream! {
        loop {
            // Wait for a new frame to arrive
            if rx.changed().await.is_err() {
                break; // sender dropped, pipeline shutting down
            }
            let jpeg: Arc<Bytes> = rx.borrow_and_update().clone();

            // MJPEG multipart boundary
            let header = format!(
                "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                jpeg.len()
            );
            yield Ok::<_, std::io::Error>(Bytes::from(header));
            yield Ok(Bytes::clone(&jpeg));
            yield Ok(Bytes::from_static(b"\r\n"));
        }
    };

    Response::builder()
        .header(
            header::CONTENT_TYPE,
            "multipart/x-mixed-replace; boundary=frame",
        )
        .header(header::CACHE_CONTROL, "no-cache, no-store")
        .header(header::PRAGMA, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap()
}

/// Serve the viewer HTML inline so no static file server is needed.
async fn index_handler() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        include_str!("../static/index.html"),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("camera_stream=info")
        .init();

    // Pipeline channels
    // mpsc bound=2: capture can be at most 1 frame ahead of encode
    let (frame_tx, frame_rx) = mpsc::sync_channel::<RawFrame>(2);
    // watch: encode writes latest JPEG; all HTTP clients read it
    let initial = Arc::new(Bytes::new());
    let (jpeg_tx, jpeg_rx) = watch::channel::<Arc<Bytes>>(initial);

    // Spawn capture thread (must not run inside Tokio)
    thread::Builder::new()
        .name("capture".into())
        .spawn(move || capture_loop(frame_tx))
        .expect("failed to spawn capture thread");

    // Spawn encode thread
    thread::Builder::new()
        .name("encode".into())
        .spawn(move || encode_loop(frame_rx, jpeg_tx))
        .expect("failed to spawn encode thread");

    // Axum router
    let state = AppState { jpeg_rx };
    let app = Router::new()
        .route("/", get(index_handler))
        .route("/stream", get(mjpeg_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = "0.0.0.0:3000";
    info!("Listening on http://{addr}");
    info!("Open http://localhost:3000 in your browser");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
