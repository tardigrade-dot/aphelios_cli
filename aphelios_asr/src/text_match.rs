use anyhow::Result;
use similar::{Algorithm, ChangeTag, TextDiff};
use std::{collections::HashMap, fs, io::Write, path::PathBuf};
use tracing::info;

use crate::sensevoice::{
    audio::{decode_audio_multi, downmix_to_mono, resample_channels},
    extract_tags,
    frontend::{FeaturePipeline, FrontendConfig},
    language_id_from_code,
    sensevoice::SensevoiceEncoder,
    vad::{SileroVad, VadConfig, VadSegment},
    SenseVoiceConfig, SenseVoiceResult, WordTimestamp,
};

// Re-export for backward compatibility
pub use crate::sensevoice::sensevoice_asr;

/// 判断是否为标点符号（不含空格）
fn is_punctuation(ch: char) -> bool {
    ch.is_ascii_punctuation()
        || matches!(
            ch,
            '。' | '，'
                | '、'
                | '！'
                | '？'
                | '；'
                | '：'
                | '「'
                | '」'
                | '『'
                | '』'
                | '【'
                | '】'
                | '〔'
                | '〕'
                | '（'
                | '）'
                | '《'
                | '》'
                | '〈'
                | '〉'
                | '｛'
                | '｝'
                | '…'
                | '—'
                | '～'
                | '·'
                | '·'
                | '•'
                | '·'
                | '．'
                | '！'
                | '？'
                | '〝'
                | '〞'
                | '＂'
                | '＇'
                | '‘'
                | '’'
                | '‚'
                | '‛'
                | '“'
                | '”'
                | '„'
                | '‟'
                | '‹'
                | '›'
                | '«'
                | '»'
                | '〈'
                | '〉'
                | '《'
                | '》'
                | '〔'
                | '〕'
                | '【'
                | '】'
                | '『'
                | '』'
                | '｛'
                | '｝'
                | '〖'
                | '〗'
                | '〘'
                | '〙'
                | '〚'
                | '〛'
                | '.'
                | ','
                | '!'
                | '?'
                | ';'
                | ':'
                | '"'
                | '\''
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '<'
                | '>'
                | '/'
                | '\\'
                | '|'
                | '@'
                | '#'
                | '$'
                | '%'
                | '^'
                | '&'
                | '*'
                | '-'
                | '_'
                | '='
                | '+'
                | '`'
                | '~'
        )
}

/// 去除标点符号，返回纯文本和原始索引映射
fn strip_punctuation(text: &str) -> (String, Vec<usize>) {
    let mut stripped = String::new();
    let mut index_map = Vec::new(); // stripped_idx -> original_idx

    for (orig_idx, ch) in text.chars().enumerate() {
        if !is_punctuation(ch) {
            stripped.push(ch);
            index_map.push(orig_idx);
        }
    }

    (stripped, index_map)
}

/// 将 ASR 生成的 word timestamp 对齐到 target 文本
/// 通过字符级别的 diff，为 target 文本生成对应的时间戳
/// 未匹配的字符时间戳为 None
/// 如果两个对齐块之间的文字数一样，就当作是完全匹配
/// 对齐时会忽略 target 中的标点符号（因为 ASR 结果通常无标点）
fn align_timestamps_to_target(
    asr_timestamps: &[WordTimestamp],
    asr_text: &str,
    target_text: &str,
) -> Vec<Option<WordTimestamp>> {
    // 先去除 target 中的标点，用纯文本与 ASR 结果对齐
    let (target_stripped, target_index_map) = strip_punctuation(target_text);

    let diff = TextDiff::configure()
        .algorithm(Algorithm::Myers)
        .diff_chars(asr_text, &target_stripped);

    // 构建 ASR 文本字符到 timestamp 的映射
    let asr_char_to_ts = build_char_to_timestamp_map(asr_timestamps, asr_text);

    // 先对 stripped target 做对齐
    let mut aligned_stripped: Vec<Option<WordTimestamp>> = Vec::new();
    let mut asr_char_idx = 0;

    let ops = diff.ops().to_vec();
    let mut op_idx = 0;

    while op_idx < ops.len() {
        let op = &ops[op_idx];

        if let Some(next_op) = ops.get(op_idx + 1) {
            let delete_count = if diff.iter_changes(op).any(|c| c.tag() == ChangeTag::Delete) {
                diff.iter_changes(op)
                    .filter(|c| c.tag() == ChangeTag::Delete)
                    .count()
            } else {
                0
            };

            let insert_count = if diff
                .iter_changes(next_op)
                .any(|c| c.tag() == ChangeTag::Insert)
            {
                diff.iter_changes(next_op)
                    .filter(|c| c.tag() == ChangeTag::Insert)
                    .count()
            } else {
                0
            };

            if delete_count > 0 && delete_count == insert_count {
                let mut delete_timestamps: Vec<WordTimestamp> = Vec::new();
                for change in diff.iter_changes(op) {
                    match change.tag() {
                        ChangeTag::Delete => {
                            if let Some(ts) = asr_char_to_ts.get(&asr_char_idx) {
                                delete_timestamps.push(ts.clone());
                            }
                            asr_char_idx += 1;
                        }
                        ChangeTag::Equal => {}
                        ChangeTag::Insert => {}
                    }
                }

                for (i, change) in diff.iter_changes(next_op).enumerate() {
                    if change.tag() == ChangeTag::Insert {
                        if let Some(ts) = delete_timestamps.get(i) {
                            aligned_stripped.push(Some(ts.clone()));
                        } else {
                            aligned_stripped.push(None);
                        }
                    }
                }

                op_idx += 2;
                continue;
            }
        }

        for change in diff.iter_changes(op) {
            match change.tag() {
                ChangeTag::Equal => {
                    let ts = asr_char_to_ts.get(&asr_char_idx).cloned();
                    aligned_stripped.push(ts);
                    asr_char_idx += 1;
                }
                ChangeTag::Delete => {
                    asr_char_idx += 1;
                }
                ChangeTag::Insert => {
                    aligned_stripped.push(None);
                }
            }
        }

        op_idx += 1;
    }

    // 将 stripped 的对齐结果映射回原始 target（含标点）
    // 标点符号位置填 None
    let mut aligned_timestamps = vec![None; target_text.chars().count()];
    for (stripped_idx, ts) in aligned_stripped.into_iter().enumerate() {
        if let Some(orig_idx) = target_index_map.get(stripped_idx) {
            aligned_timestamps[*orig_idx] = ts;
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
    // 去除标点后计算匹配率（ASR 无标点）
    let (target_stripped, _) = strip_punctuation(target_text);
    let diff = TextDiff::configure()
        .algorithm(Algorithm::Myers)
        .diff_chars(&sv_result.text, &target_stripped);

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
    let match_rate = if target_stripped.is_empty() {
        0.0
    } else {
        keep_count as f32 / target_stripped.len() as f32
    };
    info!("target_text match_rate: {:.2}%", match_rate * 100.0);

    // 检查匹配率，如果太低说明 target 和 ASR 不匹配，无法生成字幕
    let min_match_rate = 0.8;
    if match_rate < min_match_rate {
        anyhow::bail!(
            "target 和 ASR 文本匹配率过低 ({:.2}%)，无法生成字幕。请检查 TTS 合成或文本来源是否正确。",
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
    target_txt_file: Option<&str>,
    output_path: Option<&str>,
    min_segment_len: Option<usize>,
    max_segment_len: Option<usize>,
) -> Result<String> {
    let target_txt_path = target_txt_file
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(input_audio).with_extension("txt"));

    let sv_result = sensevoice_asr(
        model_path,
        input_audio,
        SenseVoiceConfig {
            language: "auto".to_string(),
            ..SenseVoiceConfig::default()
        },
    )?;

    let target_text = std::fs::read_to_string(target_txt_path)?;

    info!("asr_result.text: {}", sv_result.text);
    info!("target_text: {}", target_text);

    let srt_content =
        generate_srt_with_sensevoice(&sv_result, &target_text, min_segment_len, max_segment_len)?;

    // 保存 SRT 文件
    let srt_path = if let Some(path) = output_path {
        PathBuf::from(path)
    } else {
        PathBuf::from(input_audio).with_extension("srt")
    };

    let mut file = std::fs::File::create(&srt_path)?;
    file.write_all(srt_content.as_bytes())?;

    Ok(srt_path.to_str().unwrap().to_string())
}

/// 音频和文本对齐生成 SRT 字幕文件（简化版本，使用默认参数）
pub fn audio_text_match(input_audio: &str, target_txt_file: Option<&str>) -> Result<String> {
    audio_text_match_with_params(
        "/Volumes/sw/onnx_models/sensevoice",
        input_audio,
        target_txt_file,
        None,
        Some(50),
        Some(80),
    )
}

/// 批量处理目录下所有 wav 文件及其同名 txt 文件
/// 复用 SenseVoice 模型，避免频繁加载
/// 输出 SRT 文件到同一目录
pub fn batch_process_wav_txt_dir(
    model_path: &str,
    dir_path: &str,
    min_segment_len: Option<usize>,
    max_segment_len: Option<usize>,
) -> Result<Vec<(String, String)>> {
    let dir = PathBuf::from(dir_path);
    if !dir.is_dir() {
        anyhow::bail!("路径不是目录: {}", dir.display());
    }

    // 收集所有 wav 文件
    let entries = fs::read_dir(&dir)?;
    let mut wav_files = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().map_or(false, |e| e == "wav") {
            wav_files.push(path);
        }
    }
    wav_files.sort();

    if wav_files.is_empty() {
        info!("目录 {} 下没有找到 wav 文件", dir.display());
        return Ok(Vec::new());
    }

    info!("找到 {} 个 wav 文件，开始批量处理", wav_files.len());

    // 初始化模型（只加载一次）
    let encoder_path = PathBuf::from(model_path).join("model.int8.onnx");
    let tokens_path = PathBuf::from(model_path).join("tokens.txt");
    let vad_model_path = PathBuf::from(model_path).join("vad-model.onnx");

    let mut encoder = SensevoiceEncoder::new(&encoder_path, 1)?;
    let decoder = crate::sensevoice::tokenizer::TokenDecoder::new(&tokens_path)?;
    let fe_cfg = FrontendConfig::default();
    let target_sample_rate = fe_cfg.sample_rate as u32;

    let vad_config = VadConfig::default();
    let mut silero_vad = SileroVad::new(&vad_model_path, fe_cfg.sample_rate, 1, vad_config)?;

    let lang_id = language_id_from_code("auto");

    let mut results = Vec::new();

    for (idx, wav_path) in wav_files.iter().enumerate() {
        let txt_path = wav_path.with_extension("txt");
        let srt_path = wav_path.with_extension("srt");

        if srt_path.exists() {
            info!("跳过 {}，已存在 srt 文件", wav_path.display());
            continue;
        }
        if !txt_path.exists() {
            info!("跳过 {}，找不到同名 txt 文件", wav_path.display());
            continue;
        }

        info!(
            "处理 [{}/{}] {}",
            idx + 1,
            wav_files.len(),
            wav_path.display()
        );

        // 运行 ASR
        let sv_result = run_sensevoice_single(
            &mut encoder,
            &decoder,
            &mut silero_vad,
            wav_path,
            lang_id,
            target_sample_rate,
        )?;

        // 读取 target 文本
        let target_text = fs::read_to_string(&txt_path)?;
        // 生成 SRT
        let srt_content = generate_srt_with_sensevoice(
            &sv_result,
            &target_text,
            min_segment_len,
            max_segment_len,
        )?;

        // 保存 SRT

        let mut file = fs::File::create(&srt_path)?;
        file.write_all(srt_content.as_bytes())?;

        results.push((
            wav_path.to_str().unwrap().to_string(),
            srt_path.to_str().unwrap().to_string(),
        ));
    }

    info!("批量处理完成，共生成 {} 个 SRT 文件", results.len());
    Ok(results)
}

/// 运行单条 SenseVoice ASR（复用模型）
fn run_sensevoice_single(
    encoder: &mut SensevoiceEncoder,
    decoder: &crate::sensevoice::tokenizer::TokenDecoder,
    silero_vad: &mut SileroVad,
    audio_path: &PathBuf,
    lang_id: i32,
    target_sample_rate: u32,
) -> Result<SenseVoiceResult> {
    use std::time::Instant;
    use tracing::debug;

    let t0 = Instant::now();
    let (decoded_sample_rate, total_channels, samples_per_channel) =
        decode_audio_multi(audio_path)?;

    let mut ch = downmix_to_mono(samples_per_channel);
    if ch.is_empty() {
        anyhow::bail!("no audio samples available for transcription");
    }

    let audio_duration_sec = ch.len() as f32 / decoded_sample_rate as f32;
    if decoded_sample_rate != target_sample_rate {
        debug!(
            "resampling audio from {} Hz to {} Hz",
            decoded_sample_rate, target_sample_rate
        );
        let resampled = resample_channels(vec![ch], decoded_sample_rate, target_sample_rate)?;
        ch = resampled.into_iter().next().unwrap();
    }

    silero_vad.reset();
    let mut segments = silero_vad.collect_segments(&ch)?;
    if segments.is_empty() {
        segments.push(VadSegment {
            start: 0,
            end: ch.len(),
        });
    }

    let mut consolidated_text = String::new();
    let mut consolidated_timestamps = Vec::new();
    let mut consolidated_words = Vec::new();

    for seg in segments {
        let start = seg.start.min(ch.len());
        let end = seg.end.min(ch.len());
        if end <= start {
            continue;
        }
        let segment_samples = &ch[start..end];
        if segment_samples.is_empty() {
            continue;
        }

        let mut fe = FeaturePipeline::new(FrontendConfig::default());
        let feats = match fe.compute_features(segment_samples, target_sample_rate) {
            Ok(f) => f,
            Err(e) => {
                debug!(start = start, end = end, error = %e, "feature extraction failed for segment");
                continue;
            }
        };
        if feats.is_empty() {
            continue;
        }

        use ndarray::Axis;
        let feats = feats.insert_axis(Axis(0));
        let (raw_text, word_timestamps) =
            encoder.run_and_decode_with_timestamps(decoder, feats.view(), lang_id, false)?;

        let (clean_text, _tags) = extract_tags(&raw_text);
        if !consolidated_text.is_empty() && !clean_text.is_empty() {
            consolidated_text.push(' ');
        }
        consolidated_text.push_str(&clean_text);

        let offset_sec = start as f32 / target_sample_rate as f32;
        for wt in word_timestamps {
            consolidated_words.push(wt.token.clone());
            consolidated_timestamps.push(WordTimestamp {
                word: wt.token,
                start_sec: wt.start_sec + offset_sec,
                end_sec: wt.end_sec + offset_sec,
            });
        }
    }

    let elapsed = t0.elapsed();
    let rtf = elapsed.as_secs_f32() / audio_duration_sec.max(1e-6);
    info!("time: {:?}, rtf: {:.3}", elapsed, rtf);

    Ok(SenseVoiceResult {
        text: consolidated_text,
        timestamp: consolidated_timestamps,
        words: consolidated_words,
    })
}
