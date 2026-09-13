use anyhow::Result;
use std::path::Path;
use tokio::fs::File;
use tokio::io::{AsyncWriteExt, BufWriter};

use crate::{AsrSegment, VadSegment};

pub fn format_srt_time(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let ms = total_ms % 1000;
    let s = (total_ms / 1000) % 60;
    let m = (total_ms / 60000) % 60;
    let h = total_ms / 3600000;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
}

pub async fn write_lines_to_file(lines: &[String], save_file: &str) -> Result<()> {
    let file = File::create(save_file).await?;
    let mut writer = BufWriter::new(file);
    for line in lines {
        writer.write_all(line.as_bytes()).await?;
        writer.write_all(b"\n").await?;
    }
    writer.flush().await?;
    Ok(())
}

pub async fn generate_vad_srt(segments: &[VadSegment], save_file: &str) -> Result<()> {
    let mut lines = Vec::new();
    for (i, segment) in segments.iter().enumerate() {
        lines.push(format!("{}", i + 1));
        let start_time = format_srt_time(segment.start as f64 / 1000.0);
        let end_time = format_srt_time(segment.end as f64 / 1000.0);
        lines.push(format!("{} --> {}", start_time, end_time));
        lines.push(String::new());
    }
    write_lines_to_file(&lines, save_file).await?;
    Ok(())
}

pub async fn generate_srt(segments: &[AsrSegment], save_file: &str) -> Result<()> {
    let mut counter = 1;
    let mut lines = Vec::new();

    for segment in segments {
        if !segment.sub_segments.is_empty() {
            for sub in &segment.sub_segments {
                lines.push(format!("{}", counter));
                let start_time = format_srt_time(sub.start);
                let end_time = format_srt_time(sub.end);
                lines.push(format!("{} --> {}", start_time, end_time));
                lines.push(format!("{}\n", sub.text.trim()));
                counter += 1;
            }
        } else {
            let text = segment.dr.text.trim();
            let clean_text = text.replace(|c: char| c == '<' || c == '|' || c == '>', "");
            let clean_text = clean_text.trim();
            if !clean_text.is_empty() && clean_text != "0.00" && clean_text != "0" {
                lines.push(format!("{}", counter));
                let start_time = format_srt_time(segment.start);
                let end_time = format_srt_time(segment.start + segment.duration);
                lines.push(format!("{} --> {}", start_time, end_time));
                lines.push(format!("{}\n", clean_text));
                counter += 1;
            }
        }
    }

    write_lines_to_file(&lines, save_file).await?;
    Ok(())
}

pub fn derive_srt_path(input_path: &str) -> String {
    let path = Path::new(input_path);
    path.with_extension("srt")
        .to_string_lossy()
        .to_string()
}
