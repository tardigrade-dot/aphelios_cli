use axum::{
    extract::State,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

use ffmpeg_next::codec::traits::Encoder;
use ffmpeg_next::software;
use webrtc::{
    api::{media_engine::MediaEngine, APIBuilder},
    media::Sample,
    peer_connection::{
        configuration::RTCConfiguration, sdp::session_description::RTCSessionDescription,
    },
    rtp_transceiver::rtp_codec::RTCRtpCodecCapability,
    track::track_local::track_local_static_sample::TrackLocalStaticSample,
};

#[derive(Clone)]
struct AppState {
    peer: Arc<Mutex<Option<webrtc::peer_connection::RTCPeerConnection>>>,
}

#[derive(Deserialize)]
struct Offer {
    sdp: String,
    #[serde(rename = "type")]
    typ: String,
}

#[derive(Serialize)]
struct Answer {
    sdp: String,
    #[serde(rename = "type")]
    typ: String,
}

#[tokio::main]
async fn main() {
    let state = AppState {
        peer: Arc::new(Mutex::new(None)),
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/offer", post(handle_offer))
        .with_state(state);

    println!("http://127.0.0.1:3000");

    axum::serve(
        tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap(),
        app,
    )
    .await
    .unwrap();
}

async fn handle_offer(State(state): State<AppState>, Json(offer): Json<Offer>) -> Json<Answer> {
    let mut m = MediaEngine::default();
    m.register_default_codecs().unwrap();

    let api = APIBuilder::new().with_media_engine(m).build();

    let pc = Arc::new(
        api.new_peer_connection(RTCConfiguration::default())
            .await
            .unwrap(),
    );

    // 创建视频轨道（H264）
    let track = Arc::new(TrackLocalStaticSample::new(
        RTCRtpCodecCapability {
            mime_type: "video/H264".to_string(),
            clock_rate: 90000,
            channels: 0,
            sdp_fmtp_line: "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42001f"
                .to_string(),
            rtcp_feedback: vec![],
        },
        "video".to_string(),
        "webrtc-rs".to_string(),
    ));

    pc.add_track(track.clone()).await.unwrap();

    // 摄像头线程 - 使用 spawn_blocking 因为 nokhwa::Camera 不是 Send
    let track_for_camera = track.clone();
    std::thread::spawn(move || {
        use nokhwa::{
            pixel_format::RgbFormat,
            utils::{CameraIndex, RequestedFormat, RequestedFormatType},
            Camera,
        };

        ffmpeg_next::init().unwrap();
        println!("Opening camera...");
        let mut cam = Camera::new(
            CameraIndex::Index(0),
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::HighestFrameRate(30)),
        )
        .unwrap();

        let resolution = cam.resolution();
        println!("Camera resolution: {:?}", resolution);
        cam.open_stream().unwrap();
        println!("Camera stream opened");

        // 初始化 H264 编码器
        let codec = ffmpeg_next::codec::encoder::find(ffmpeg_next::codec::Id::H264)
            .expect("H264 codec not found");
        println!("Found H264 codec: {:?}", codec.name());
        let context = ffmpeg_next::codec::context::Context::new_with_codec(codec);
        let mut encoder = context.encoder().video().unwrap();
        encoder.set_width(resolution.x());
        encoder.set_height(resolution.y());
        encoder.set_format(ffmpeg_next::format::Pixel::YUV420P);
        encoder.set_time_base(ffmpeg_next::Rational::new(1, 30));
        encoder.set_frame_rate(Some(ffmpeg_next::Rational::new(30, 1)));
        // 设置 GOP 大小为 1，强制每帧都是关键帧（用于低延迟流）
        encoder.set_gop(1);
        // 使用 zerolatency preset 来减少编码延迟
        let mut opts = ffmpeg_next::Dictionary::new();
        opts.set("preset", "ultrafast");
        opts.set("tune", "zerolatency");
        opts.set("profile", "baseline");
        let mut encoder = encoder.open_with(opts).unwrap();
        println!("H264 encoder opened successfully");

        // 发送一个全黑帧来初始化编码器并获取输出
        let mut init_frame = ffmpeg_next::frame::Video::new(
            ffmpeg_next::format::Pixel::YUV420P,
            resolution.x(),
            resolution.y(),
        );
        init_frame.set_pts(Some(0));
        encoder.send_frame(&init_frame).unwrap();
        let mut init_packet = ffmpeg_next::Packet::empty();
        if encoder.receive_packet(&mut init_packet).is_ok() {
            println!("Initial encoder packet: size={} bytes", init_packet.size());
        } else {
            println!("No initial packet from encoder (expected for some configs)");
        }

        // RGB 到 YUV420P 转换器
        let mut scaler = software::scaling::Context::get(
            ffmpeg_next::format::Pixel::RGB24,
            resolution.x(),
            resolution.y(),
            ffmpeg_next::format::Pixel::YUV420P,
            resolution.x(),
            resolution.y(),
            software::scaling::Flags::BILINEAR,
        )
        .unwrap();

        // 使用阻塞运行时来调用 async write_sample
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut frame_count = 0;
        let mut pts = 0;

        loop {
            let frame = match cam.frame() {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Frame error: {:?}", e);
                    continue;
                }
            };

            // 使用 decode_image 获取 RGB 数据
            let decoded = match frame.decode_image::<RgbFormat>() {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Decode error: {:?}", e);
                    continue;
                }
            };

            let data = decoded.as_raw().to_vec();

            if frame_count == 0 {
                println!(
                    "First camera frame: size={} bytes, resolution={:?}",
                    data.len(),
                    resolution
                );
                println!(
                    "Expected RGB size: {} bytes",
                    resolution.x() as usize * resolution.y() as usize * 3
                );
            }

            // 创建 RGB frame 并填充数据
            let mut rgb_frame = ffmpeg_next::frame::Video::new(
                ffmpeg_next::format::Pixel::RGB24,
                resolution.x(),
                resolution.y(),
            );
            rgb_frame.set_pts(Some(pts));

            let width = resolution.x() as usize;
            let height = resolution.y() as usize;
            let dst_stride = rgb_frame.stride(0);

            unsafe {
                let dst_data = rgb_frame.data_mut(0);
                let src = data.as_slice();

                for y in 0..height {
                    let src_offset = y * width * 3;
                    let dst_offset = y * dst_stride;
                    if src_offset + width * 3 <= src.len()
                        && dst_offset + width * 3 <= dst_data.len()
                    {
                        dst_data[dst_offset..dst_offset + width * 3]
                            .copy_from_slice(&src[src_offset..src_offset + width * 3]);
                    }
                }
            }

            // 转换为 YUV420P
            let mut yuv_frame = ffmpeg_next::frame::Video::empty();
            scaler.run(&rgb_frame, &mut yuv_frame).unwrap();
            yuv_frame.set_pts(Some(pts));
            pts += 1;

            if frame_count == 0 && pts <= 3 {
                println!(
                    "YUV frame: width={}, height={}, format={:?}, pts={:?}",
                    yuv_frame.width(),
                    yuv_frame.height(),
                    yuv_frame.format(),
                    yuv_frame.pts()
                );
            }

            encoder.send_frame(&yuv_frame).unwrap();

            let mut encoded_packet = ffmpeg_next::Packet::empty();
            let packets_received = encoder.receive_packet(&mut encoded_packet);
            if packets_received.is_ok() {
                frame_count += 1;
                if frame_count <= 5 || frame_count % 30 == 0 {
                    println!(
                        "Encoded packet {}: size={} bytes, pts={:?}, key={}",
                        frame_count,
                        encoded_packet.size(),
                        encoded_packet.pts(),
                        encoded_packet.is_key()
                    );
                }

                let h264_data = encoded_packet.data().expect("No data").to_vec();
                let sample = Sample {
                    data: bytes::Bytes::from(h264_data),
                    duration: std::time::Duration::from_millis(33),
                    ..Default::default()
                };

                let track_clone = track_for_camera.clone();
                rt.block_on(async move {
                    if let Err(e) = track_clone.write_sample(&sample).await {
                        eprintln!("Write sample error: {:?}", e);
                    }
                });
            } else {
                if pts <= 5 {
                    println!(
                        "No packet received for frame {}, encoder may be buffering",
                        pts
                    );
                }
            }

            if frame_count == 0 {
                println!("Warning: No encoded packets from H264 encoder!");
            }
        }
    });

    let offer_sdp = RTCSessionDescription::offer(offer.sdp).unwrap();

    pc.set_remote_description(offer_sdp).await.unwrap();

    let answer = pc.create_answer(None).await.unwrap();
    pc.set_local_description(answer.clone()).await.unwrap();

    Json(Answer {
        sdp: answer.sdp,
        typ: "answer".to_string(),
    })
}

async fn index() -> Html<&'static str> {
    Html(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>WebRTC Example</title>
    <style>
        video { width: 640px; height: 480px; background: #000; }
    </style>
</head>
<body>
    <h1>WebRTC Camera Stream</h1>
    <video id="remoteVideo" autoplay playsinline></video>
    <div id="status"></div>
    <script>
        async function startWebRTC() {
            const status = document.getElementById('status');
            const video = document.getElementById('remoteVideo');
            
            status.textContent = 'Creating peer connection...';
            const pc = new RTCPeerConnection();

            pc.addTransceiver('video', { direction: 'recvonly' });

            pc.ontrack = (event) => {
                status.textContent = 'Track received!';
                console.log('Track event:', event);
                video.srcObject = event.streams[0];
            };
            
            pc.onicecandidate = (e) => {
                console.log('ICE candidate:', e.candidate);
            };
            
            pc.onconnectionstatechange = () => {
                status.textContent = 'Connection state: ' + pc.connectionState;
                console.log('Connection state:', pc.connectionState);
            };

            status.textContent = 'Creating offer...';
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            // Wait for ICE gathering to complete
            await new Promise(resolve => {
                if (pc.iceGatheringState === 'complete') {
                    resolve();
                } else {
                    pc.onicecandidate = () => {
                        if (pc.iceGatheringState === 'complete') resolve();
                    };
                }
            });

            status.textContent = 'Sending offer to server...';
            const response = await fetch('/offer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type })
            });

            const answer = await response.json();
            status.textContent = 'Setting remote description...';
            await pc.setRemoteDescription({ type: answer.type, sdp: answer.sdp });
            status.textContent = 'Done!';
        }

        startWebRTC();
    </script>
</body>
</html>"#,
    )
}
