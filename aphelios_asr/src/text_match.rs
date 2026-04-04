use anyhow::Result;
use similar::{Algorithm, ChangeTag, TextDiff};
use std::{collections::HashMap, io::Write, path::PathBuf};
use tracing::info;

use crate::{
    sensevoice::{sensevoice_asr, SenseVoiceConfig, SenseVoiceResult, WordTimestamp},
    text_match,
};

/// 将 ASR 生成的 word timestamp 对齐到 target 文本
/// 通过字符级别的 diff，为 target 文本生成对应的时间戳
/// 未匹配的字符时间戳为 None
fn align_timestamps_to_target(
    asr_timestamps: &[WordTimestamp],
    asr_text: &str,
    target_text: &str,
) -> Vec<Option<WordTimestamp>> {
    let diff = TextDiff::configure()
        .algorithm(Algorithm::Myers)
        .diff_chars(asr_text, target_text);

    let mut aligned_timestamps: Vec<Option<WordTimestamp>> = Vec::new();

    // 构建 ASR 文本字符到 timestamp 的映射
    let asr_char_to_ts = build_char_to_timestamp_map(asr_timestamps, asr_text);

    let mut asr_char_idx = 0;

    for op in diff.ops() {
        for change in diff.iter_changes(op) {
            match change.tag() {
                ChangeTag::Equal => {
                    // 相同部分，直接从 ASR 获取时间戳
                    let ts = asr_char_to_ts.get(&asr_char_idx).cloned();
                    aligned_timestamps.push(ts);
                    asr_char_idx += 1;
                }
                ChangeTag::Delete => {
                    // ASR 多出的部分（语气词等），跳过，不对应 target 字符
                    asr_char_idx += 1;
                }
                ChangeTag::Insert => {
                    // target 多出的部分（ASR 漏读），时间戳为 None
                    aligned_timestamps.push(None);
                }
            }
        }
    }

    aligned_timestamps
}

/// 构建字符索引到时间戳的映射
fn build_char_to_timestamp_map(
    timestamps: &[WordTimestamp],
    text: &str,
) -> HashMap<usize, WordTimestamp> {
    let mut char_to_ts = HashMap::new();
    let mut char_idx = 0;

    for ts in timestamps {
        let word_len = ts.word.chars().count();
        for i in 0..word_len {
            char_to_ts.insert(
                char_idx + i,
                WordTimestamp {
                    word: ts.word.chars().nth(i).unwrap_or(' ').to_string(),
                    start_sec: ts.start_sec,
                    end_sec: ts.end_sec,
                },
            );
        }
        char_idx += word_len;
        // 跳过空格
        if char_idx < text.chars().count() && text.chars().nth(char_idx) == Some(' ') {
            char_to_ts.insert(
                char_idx,
                WordTimestamp {
                    word: ' '.to_string(),
                    start_sec: ts.end_sec,
                    end_sec: ts.end_sec,
                },
            );
            char_idx += 1;
        }
    }

    char_to_ts
}

/// SRT 分段结构
struct SrtSegment {
    text: String,
    start_sec: f32,
    end_sec: f32,
}

/// 生成带分段的 SRT 内容
/// 按照 100-200 字从符号处切断，保证尾部符号的一致性
/// 只需要切割点附近有准确的时间戳即可
/// 如果切割点附近没有时间戳，返回错误
fn generate_srt_with_segmentation(
    timestamps: &[Option<WordTimestamp>],
    target_text: &str,
    min_len: Option<usize>,
    max_len: Option<usize>,
) -> Result<String> {
    let min_len = min_len.unwrap_or(80);
    let max_len = max_len.unwrap_or(120);
    // 中文标点符号，用于断句
    let sentence_endings = ['。', '！', '？', '；', '!', '?', ';'];
    let pause_marks = ['，', '、', '：', ':', '｜', '|'];

    let mut segments = Vec::new();
    let mut current_segment = String::new();
    let mut current_char_count = 0;

    let target_chars: Vec<char> = target_text.chars().collect();

    // 记录当前段的起始和结束时间戳
    let mut segment_start_time: Option<f32> = None;
    let mut segment_end_time: Option<f32> = None;

    // 记录上一段的结束时间，确保时间戳不重叠
    let mut prev_end_time: Option<f32> = None;

    for (char_idx, &ch) in target_chars.iter().enumerate() {
        if ch == '\n' || ch == '\r' {
            continue; // 跳过换行符
        }

        current_segment.push(ch);
        current_char_count += 1;

        // 更新当前段的时间戳范围
        if let Some(ts) = timestamps.get(char_idx).and_then(|t| t.as_ref()) {
            if segment_start_time.is_none() {
                segment_start_time = Some(ts.start_sec);
            }
            segment_end_time = Some(ts.end_sec);
        }

        // 检查是否应该在此处切断 (100-200 字规则)
        let should_break = if current_char_count >= min_len {
            // 达到最小长度，寻找合适的切断点
            sentence_endings.contains(&ch)
                || (current_char_count >= (min_len + max_len) / 2 && pause_marks.contains(&ch))
                || current_char_count >= max_len
        } else {
            current_char_count >= max_len
        };

        if should_break && !current_segment.is_empty() {
            // 确保尾部符号一致性：如果是句子结尾符号，保持完整
            let segment_end_idx = if sentence_endings.contains(&ch) {
                char_idx + 1
            } else if pause_marks.contains(&ch) && current_char_count < 200 {
                char_idx + 1
            } else {
                // 在 200 字强制切断时，寻找最近的合适切断点
                find_best_break_point(&current_segment, &sentence_endings, &pause_marks)
            };

            if segment_end_idx > 0 {
                let segment_text: String = current_segment.chars().take(segment_end_idx).collect();

                // 计算切断点在整体 target 中的位置
                // 需要找到当前段第一个字符在 target 中的位置，然后加上 segment_end_idx
                // 简化处理：使用当前字符位置向前推算
                let break_position =
                    char_idx.saturating_sub(current_char_count.saturating_sub(segment_end_idx));

                // 获取该段的结束时间戳（从切割点附近查找）
                let end_time =
                    find_timestamp_near_position(timestamps, break_position).ok_or_else(|| {
                        anyhow::anyhow!(
                            "在位置 {} 附近（段长 {} 字，切断点 {}）找不到有效时间戳，无法生成字幕分段。target 文本：\"{}\"",
                            break_position,
                            current_char_count,
                            segment_end_idx,
                            segment_text.chars().take(50).collect::<String>()
                        )
                    })?;
                // 起始时间：优先使用 segment_start_time，如果没有则使用 end_time
                // 但要确保不早于上一段的结束时间
                let mut start_time = segment_start_time.unwrap_or(end_time);
                if let Some(prev_end) = prev_end_time {
                    if start_time < prev_end {
                        start_time = prev_end;
                    }
                }

                segments.push(SrtSegment {
                    text: segment_text.trim().to_string(),
                    start_sec: start_time,
                    end_sec: end_time,
                });

                // 重置状态，开始新的一段
                current_segment = current_segment.chars().skip(segment_end_idx).collect();
                current_char_count = current_segment.chars().count();
                segment_start_time = None;
                segment_end_time = None;
                prev_end_time = Some(end_time); // 记录上一段的结束时间
            }
        }
    }

    // 添加最后一段
    if !current_segment.is_empty() {
        let trimmed_text = current_segment.trim();
        // 只添加有实际内容的段
        if !trimmed_text.is_empty() {
            // 最后一段：放弃严格检测，使用可用的时间戳或默认值
            // 因为已经没有机会通过切割来找匹配的部分了
            let last_ts = find_last_valid_timestamp(timestamps);

            let start_time = segment_start_time
                .or_else(|| last_ts.map(|ts| ts.start_sec))
                .unwrap_or(prev_end_time.unwrap_or(0.0));

            let end_time = segment_end_time
                .or_else(|| last_ts.map(|ts| ts.end_sec))
                .unwrap_or(start_time + 1.0);

            // 确保最后一段的 start 不早于上一段的 end
            let mut start_time = start_time;
            if let Some(prev_end) = prev_end_time {
                if start_time < prev_end {
                    start_time = prev_end;
                }
            }

            segments.push(SrtSegment {
                text: trimmed_text.to_string(),
                start_sec: start_time,
                end_sec: end_time,
            });
        }
    }

    // 生成 SRT 格式文本
    let mut srt_content = String::new();
    for (idx, segment) in segments.iter().enumerate() {
        srt_content.push_str(&format!("{}\n", idx + 1));
        srt_content.push_str(&format!(
            "{} --> {}\n",
            format_srt_time(segment.start_sec as f64),
            format_srt_time(segment.end_sec as f64)
        ));
        srt_content.push_str(&format!("{}\n\n", segment.text));
    }

    Ok(srt_content)
}

/// 在指定位置附近查找有效的时间戳
/// 先向后找，再向前找，找到第一个有有效时间戳的位置
fn find_timestamp_near_position(
    timestamps: &[Option<WordTimestamp>],
    position: usize,
) -> Option<f32> {
    // 先向后查找（包括当前位置）
    for i in position..timestamps.len() {
        if let Some(ts) = timestamps[i].as_ref() {
            return Some(ts.end_sec);
        }
    }
    // 再向前查找
    for i in (0..position).rev() {
        if let Some(ts) = timestamps[i].as_ref() {
            return Some(ts.end_sec);
        }
    }
    None
}

/// 查找最后一个有效的时间戳
fn find_last_valid_timestamp(timestamps: &[Option<WordTimestamp>]) -> Option<&WordTimestamp> {
    for i in (0..timestamps.len()).rev() {
        if let Some(ts) = timestamps[i].as_ref() {
            return Some(ts);
        }
    }
    None
}

/// 寻找最佳切断点，优先从符号处切断
fn find_best_break_point(segment: &str, sentence_endings: &[char], pause_marks: &[char]) -> usize {
    let chars: Vec<char> = segment.chars().collect();
    let len = chars.len();

    // 从后向前寻找句子结尾符号
    for i in (0..len).rev() {
        if sentence_endings.contains(&chars[i]) {
            return i + 1;
        }
    }

    // 寻找停顿符号（从后 50 字开始向前找）
    for i in (len.saturating_sub(50)..len).rev() {
        if pause_marks.contains(&chars[i]) {
            return i + 1;
        }
    }

    // 默认在中间切断（如果找不到任何符号）
    // 注意：这里只是返回切断位置，时间戳检测由调用方处理
    len / 2
}

/// 格式化 SRT 时间
fn format_srt_time(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let ms = total_ms % 1000;
    let s = (total_ms / 1000) % 60;
    let m = (total_ms / 60000) % 60;
    let h = total_ms / 3600000;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
}

pub fn generate_srt_with_sensevoice(
    sv_result: &SenseVoiceResult,
    target_text: &String,
    min_len: Option<usize>,
    max_len: Option<usize>,
) -> Result<String> {
    let diff = TextDiff::configure()
        .algorithm(Algorithm::Myers)
        .diff_chars(&sv_result.text, target_text);

    let mut keep_count = 0;
    for op in diff.ops() {
        for change in diff.iter_changes(op) {
            match change.tag() {
                ChangeTag::Equal => {
                    keep_count += change.value().len();
                }
                ChangeTag::Delete => {}
                ChangeTag::Insert => {}
            }
        }
    }
    info!("keep_count: {}", keep_count);
    info!("target_text.len(): {}", target_text.len());

    let match_rate = keep_count as f32 / target_text.len() as f32;
    info!("target_text match_rate: {}", match_rate);

    // 检查匹配率，如果太低说明 target 和 ASR 不匹配，无法生成字幕
    let min_match_rate = 0.8;
    if match_rate < min_match_rate {
        anyhow::bail!(
            "target 和 ASR 文本匹配率过低 ({:.1}%)，无法生成字幕。请检查 TTS 合成或文本来源是否正确。",
            match_rate * 100.0
        );
    }

    // 生成匹配 target 的 timestamp 数据，未匹配的字符时间戳为 None
    let aligned_timestamps =
        align_timestamps_to_target(&sv_result.timestamp, &sv_result.text, &target_text);

    // 生成 SRT 字幕文件
    let srt_content =
        generate_srt_with_segmentation(&aligned_timestamps, &target_text, min_len, max_len)?;
    Ok(srt_content)
}

/// 音频和文本对齐生成 SRT 字幕文件
///
/// # 参数
/// * `model_path` - SenseVoice 模型路径
/// * `input_audio` - 输入音频文件路径
/// * `target_txt_file` - 目标文本文件路径
/// * `output_path` - 输出 SRT 文件路径（如果为 None，则使用输入文件路径替换扩展名为.srt）
/// * `min_segment_len` - 最小分段长度（字符数），None 时使用默认值 80
/// * `max_segment_len` - 最大分段长度（字符数），None 时使用默认值 120
pub fn audio_text_match_with_params(
    model_path: &str,
    input_audio: &str,
    target_txt_file: &str,
    output_path: Option<&str>,
    min_segment_len: Option<usize>,
    max_segment_len: Option<usize>,
) -> Result<String> {
    let sv_result = sensevoice_asr(
        model_path,
        input_audio,
        SenseVoiceConfig {
            language: "auto".to_string(),
            ..SenseVoiceConfig::default()
        },
    )?;

    let target_text = std::fs::read_to_string(target_txt_file)?;

    info!("asr_result.text: {}", sv_result.text);
    info!("target_text: {}", target_text);

    let srt_content =
        generate_srt_with_sensevoice(&sv_result, &target_text, min_segment_len, max_segment_len)?;

    // 保存 SRT 文件
    let srt_path = if let Some(path) = output_path {
        PathBuf::from(path)
    } else {
        PathBuf::from(target_txt_file).with_extension("srt")
    };

    let mut file = std::fs::File::create(&srt_path)?;
    file.write_all(srt_content.as_bytes())?;

    Ok(srt_path.to_str().unwrap().to_string())
}

/// 音频和文本对齐生成 SRT 字幕文件（简化版本，使用默认参数）
pub fn audio_text_match(input_audio: &str, target_txt_file: &str) -> Result<String> {
    audio_text_match_with_params(
        "/Volumes/sw/onnx_models/sensevoice",
        input_audio,
        target_txt_file,
        None,
        Some(50),
        Some(80),
    )
}
