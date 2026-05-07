use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};


/// 解析 yt-dlp 进度行
/// 格式: [download]  45.2% of  200.00MiB at  5.20MiB/s ETA 00:25
fn parse_progress(line: &str) -> Option<ProgressData> {
    let re = Regex::new(
        r#"\[download\]\s+(\d+\.?\d*)%\s+of\s+([\d.]+)([A-Za-z]+)\s+at\s+([\d.]+)([A-Za-z]+)/s\s+ETA\s+([\d:]+)"#
    ).ok()?;

    let caps = re.captures(line)?;

    // 解析字节数 (简化处理，实际可更精确)
    let size_val: f64 = caps[2].parse().ok()?;
    let size_unit = &caps[3];
    let total_bytes = match size_unit.to_lowercase().as_str() {
        "mib" | "mb" => size_val * 1024.0 * 1024.0,
        "kib" | "kb" => size_val * 1024.0,
        "gib" | "gb" => size_val * 1024.0 * 1024.0 * 1024.0,
        _ => size_val,
    };

    Some(ProgressData {
        percent: caps[1].parse().ok()?,
        downloaded_bytes: total_bytes,
        speed: format!("{}{}/s", &caps[4], &caps[5]),
        eta: caps[6].to_string(),
        total: None, // yt-dlp 进度行不直接给 total，用 percent 反推
    })
}

#[derive(Debug)]
struct ProgressData {
    percent: f64,
    downloaded_bytes: f64,
    speed: String,
    eta: String,
    total: Option<f64>,
}

/// 构建 yt-dlp 命令参数
fn build_args(output_dir: &str, url: &str) -> Vec<String> {
    let mut params = vec![
        "-f".to_string(),
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best".to_string(),
        "--merge-output-format".to_string(),
        "mp4".to_string(),
        "--write-sub".to_string(),
        "--convert-subs".to_string(),
        "srt".to_string(),
        "--progress".to_string(),
        "--newline".to_string(),  // 确保每行输出，便于解析
    ];

    // 添加自定义下载路径
    params.push("--paths".to_string());
    params.push(output_dir.to_string());

    // 添加输出模板 --output "%(title)s.%(ext)s"
    params.push("-o".to_string());
    params.push("%(title)s.%(ext)s".to_string());

    // 添加视频地址
    params.push(url.to_string());

    params
}

/// 执行下载并显示美观进度条
pub fn download_with_progress(url: &str, output_dir: &str) -> Result<(), String> {
    let yt_args = build_args(output_dir, url);

    println!("🚀 启动下载: {}", url);
    println!("📁 下载目录: {}", output_dir);
    println!();

    let mut child = Command::new("yt-dlp")
        .args(&yt_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("❌ 无法启动 yt-dlp: {}", e))?;

    let stderr = child.stderr.take()
        .ok_or("❌ 无法捕获标准错误输出")?;
    let reader = BufReader::new(stderr);

    // 创建 indicatif 进度条
    let pb = ProgressBar::new(100); // 初始 100%，动态更新
    pb.set_style(ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {percent:>3}% {msg}",
    ).unwrap().progress_chars("█▉▊▋▌▍▎▏  "));

    // 额外信息显示
    let pb_details = ProgressBar::new(0);
    pb_details.set_style(ProgressStyle::with_template(
        "   📦 {bytes}/{total_bytes} | ⚡ {speed} | ⏱️  ETA: {eta}",
    ).unwrap().progress_chars("##>-"));

    for line_result in reader.lines() {
        let line = line_result.map_err(|e| format!("读取输出失败: {}", e))?;

        if let Some(progress) = parse_progress(&line) {
            // 更新主进度条 (0-100)
            pb.set_position(progress.percent as u64);

            // 计算总大小 (反推)
            let total_bytes = if progress.percent > 0.0 {
                (progress.downloaded_bytes / progress.percent * 100.0) as u64
            } else {
                0
            };

            // 更新详细信息
            pb_details.set_length(total_bytes);
            pb_details.set_position(progress.downloaded_bytes as u64);
            pb_details.set_message(format!(
                "speed: {} | eta: {}",
                progress.speed,
                progress.eta
            ));
        } else if !line.trim().is_empty() {
            // 输出其他日志信息
            if line.contains("Destination") || line.contains("Downloading") {
                pb.println(format!("   ℹ️  {}", line));
            }
        }
    }

    // 完成进度条
    pb.finish_with_message("✅ 下载完成");
    pb_details.finish_and_clear();

    let status = child.wait()
        .map_err(|e| format!("等待进程失败: {}", e))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("❌ yt-dlp 执行失败，退出码: {}", status))
    }
}

/// 交互式获取视频地址（当命令行未提供时）
fn prompt_url() -> String {
    print!("🔗 请输入 YouTube 视频地址: ");
    std::io::stdout().flush().unwrap();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).expect("读取输入失败");
    input.trim().to_string()
}
