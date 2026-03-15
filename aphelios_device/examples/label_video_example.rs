use anyhow::{anyhow, Result};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::Camera;
use std::io::Write;
use std::process::{Command, Stdio};
use usls::{Annotator, Config, Image, Model, RTDETR};

fn main() -> Result<()> {
    let _ = label_video_fast("output/fast_video.mp4");
    Ok(())
}

pub fn label_video_fast(output_path: &str) -> Result<()> {
    // 1. 初始化模型
    let config = Config::rtdetr_v4_m()
        .with_model_file("/Users/larry/coderesp/aphelios_cli/aphelios_core/models/rtdetr_v4_m.onnx")
        .with_class_confs(&[0.5])
        .commit()?;
    let (mut model, mut engines) = RTDETR::build(config)?;
    let annotator = Annotator::default();

    // 2. 使用 nokhwa 打开视频文件 (nokhwa 也支持文件输入)
    // 注意：如果是处理现有视频文件，nokhwa 的支持取决于后端，
    // 如果 nokhwa 不好使，建议直接用 `image` 库配合 ffmpeg 抽流。
    // 这里演示“处理摄像头并实时存为视频”的极速方案：
    let mut camera = Camera::new(
        CameraIndex::Index(0),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
    )?;
    camera.open_stream()?;
    let resolution = camera.camera_format().resolution();

    // 3. 启动 ffmpeg 写入进程 (通过管道接收数据)
    let mut ffmpeg_process = Command::new("ffmpeg")
        .args([
            "-f",
            "rawvideo", // 输入格式为原始视频
            "-pixel_format",
            "rgb24", // 像素格式
            "-video_size",
            &format!("{}x{}", resolution.width(), resolution.height()),
            "-i",
            "-", // 从 stdin 读取
            "-c:v",
            "libx264", // H.264 编码
            "-pix_fmt",
            "yuv420p", // 兼容性好的格式
            "-preset",
            "ultrafast", // 编码速度优先
            "-y",
            output_path, // 输出路径
        ])
        .stdin(Stdio::piped())
        .spawn()?;

    let mut stdin = ffmpeg_process
        .stdin
        .take()
        .ok_or_else(|| anyhow!("Failed to open stdin"))?;

    println!("正在录制并处理... 按下 Ctrl+C 停止（或设置循环次数）");

    for _ in 0..300 {
        // 假设处理 300 帧
        let frame = camera.frame()?;
        let decoded = frame.decode_image::<RgbFormat>()?;

        let mut dynamic_image = image::DynamicImage::ImageRgb8(decoded);
        let usls_image = Image::from(dynamic_image.clone());

        // 推理并标注
        if let Ok(ys) = model.run(&mut engines, &[usls_image.clone()]) {
            if !ys.is_empty() {
                dynamic_image = annotator.annotate(&usls_image, &ys[0])?.into_dyn();
            }
        }

        // --- 极速步骤：直接将原始像素推入 ffmpeg 管道 ---
        // 不经过任何 PNG 编码，直接推 RGB 字节流
        stdin.write_all(dynamic_image.to_rgb8().as_raw())?;
    }

    drop(stdin); // 关闭管道，ffmpeg 会自动结束
    ffmpeg_process.wait()?;

    println!("生成完毕：{}", output_path);
    Ok(())
}
