use std::{fs, thread::sleep, time::Duration};
use anyhow::{Context, Result};
use async_openai::{Client, config::OpenAIConfig};
use tracing::{error, info};

use crate::openai::infer::simple_infer;

#[derive(Debug, Clone)]
struct SrtBlock {
    index: u32,
    timestamp: String,
    text: String,
    translated_text: String,
}

/// 解析 SRT 文件内容
fn parse_srt(content: &str) -> Result<Vec<SrtBlock>> {
    let normalized = content.replace("\r\n", "\n").replace("\r", "\n");
    let mut blocks = Vec::new();
    let mut current_lines = Vec::new();

    for line in normalized.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !current_lines.is_empty() {
                process_block(&mut blocks, &current_lines)?;
                current_lines.clear();
            }
        } else {
            current_lines.push(trimmed);
        }
    }
    if !current_lines.is_empty() {
        process_block(&mut blocks, &current_lines)?;
    }
    Ok(blocks)
}

fn process_block(blocks: &mut Vec<SrtBlock>, lines: &[&str]) -> Result<()> {
    if lines.len() < 3 {
        return Ok(());
    }
    let index = lines[0]
        .parse::<u32>()
        .with_context(|| format!("无效的序号: {}", lines[0]))?;
    let timestamp = lines[1].to_string();
    let text = lines[2..].join("\n");

    blocks.push(SrtBlock {
        index,
        timestamp,
        text,
        translated_text: String::new(),
    });
    Ok(())
}

/// 判断是否为安全的句子拆分点（基于结尾标点）
fn is_sentence_boundary(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    let last = trimmed.chars().last().unwrap();
    // 支持中英文常见句子结束符及逗号，避免在句子中间切断
    matches!(last, '.' | '!' | '?' | '。' | '！' | '？' | '…' | '，' | ';' | '；' | ')' | '"')
}

/// 按句子边界和最大批次大小对字幕进行分组，返回索引批次
fn group_into_batches(blocks: &[SrtBlock], max_size: usize) -> Vec<Vec<usize>> {
    let mut batches = Vec::new();
    let mut current = Vec::new();

    for (i, block) in blocks.iter().enumerate() {
        current.push(i);
        // 达到上限 或 当前行以句子边界标点结尾时，进行切分
        if current.len() >= max_size && is_sentence_boundary(&block.text) {
            batches.push(current);
            current = Vec::new();
        }
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

/// 批量翻译：输入输出均采用 "1. 文本" 的行编号格式，便于模型理解和对齐解析
async fn translate_batch(client: &Client<OpenAIConfig>, model_id: &str, batch: &[&SrtBlock], ctx: Option<&str>) -> Result<Vec<String>> {
    // 构造输入：每行 "序号. 原文"
    let input_lines: Vec<String> = batch
        .iter()
        .enumerate()
        .map(|(i, b)| format!("{}. {}", i + 1, b.text.trim()))
        .collect();
    let input_text = input_lines.join("\n");

    let prompt;
    if let Some(c) = ctx{
        prompt = format!(
            "将以下字幕逐行翻译为中文。\n\
            这是提示信息: {}\n\
            规则：\n\
            - 输出必须恰好 {} 行，与输入行数完全一致\n\
            - 每行只翻译对应的原文，不合并不拆分\n\
            - 如果某行语义不完整，保留断句，译文可以是半句\n\
            - 不需要其他解释，只输出翻译后的序号和译文\n\
            输入：\n{}\n\n",
            c,
            batch.len(),
            input_text
        );
    }else{

        prompt = format!(
            "将以下字幕逐行翻译为中文。\n\
            规则：\n\
            - 输出必须恰好 {} 行，与输入行数完全一致\n\
            - 每行只翻译对应的原文，不合并不拆分\n\
            - 如果某行语义不完整，保留断句，译文可以是半句\n\
            - 不需要其他解释，只输出翻译后的序号和译文\n\
            输入：\n{}\n\n",
            batch.len(),
            input_text
        );
    }

    info!("prompt {}", &prompt);
    let response = simple_infer(client, model_id, prompt).await?;

    info!("response {}", &response);
    // 解析响应：按行分割，提取 "数字. 译文" 中的译文部分
    let mut translations = Vec::new();
    for line in response.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // 匹配 "数字. 内容" 或 "数字. 内容" 格式，提取内容部分
        if let Some(pos) = trimmed.find(|c: char| c == '.' || c == '．' || c == '、') {
            let content = trimmed[pos + 1..].trim();
            if !content.is_empty() {
                translations.push(content.to_string());
            }
        } else {
            // 如果找不到分隔符，整行作为译文（兼容模型输出不规范的情况）
            translations.push(trimmed.to_string());
        }
    }

    // 校验数量：如果解析出的译文数量与输入不匹配，尝试降级处理
    if translations.len() != batch.len() {
        info!(
            "⚠️ 翻译数量不匹配: 期望 {}, 解析得到 {}. 尝试按顺序截取/填充...",
            batch.len(),
            translations.len()
        );
        // 如果多了，截取前 N 个；如果少了，用空字符串填充
        translations.resize_with(batch.len(), || "[翻译缺失]".to_string());
    }

    Ok(translations)
}

/// 生成双语 SRT 内容（原文在上，译文在下）
fn generate_bilingual_srt(blocks: &[SrtBlock]) -> String {
    let mut output = String::new();
    for (i, block) in blocks.iter().enumerate() {
        output.push_str(&format!("{}\n", block.index));
        output.push_str(&format!("{}\n", block.timestamp));
        output.push_str(&format!("{}\n{}\n", block.text, block.translated_text));
        if i < blocks.len() - 1 {
            output.push('\n');
        }
    }
    output
}

pub async fn process_translator(ctx: &str, srt_path: &str, output_path: &str) -> Result<String>{
    let input_content = fs::read_to_string(srt_path)
        .with_context(|| format!("无法读取文件: {}", srt_path))?;
    let mut blocks = parse_srt(&input_content)?;

    // 配置：最大批次大小，建议 10~20
    const MAX_BATCH_SIZE: usize = 10;
    let batches = group_into_batches(&blocks, MAX_BATCH_SIZE);

    info!("✅ 解析完成，共 {} 条字幕，分为 {} 个批次翻译...", blocks.len(), batches.len());
    let api_base = "http://localhost:1234/v1";
    let model_id = "google/gemma-4-e2b"; //"hy-mt1.5-1.8b";//"qwen/qwen3.5-9b";
    let config = OpenAIConfig::new().with_api_base(api_base);
    let client = Client::with_config(config);

    for (batch_idx, indices) in batches.iter().enumerate() {
        let batch_refs: Vec<&SrtBlock> = indices.iter().map(|&i| &blocks[i]).collect();
        info!("⏳ 批次 {}/{} ({} 条)... ", batch_idx + 1, batches.len(), indices.len());

        match translate_batch(&client, model_id, &batch_refs, Some(ctx)).await {
            Ok(translations) => {
                for (&idx, trans) in indices.iter().zip(translations.iter()) {
                    blocks[idx].translated_text = trans.clone();
                }
                info!("✅ 批次完成");
            }
            Err(e) => {
                error!("❌ 批次翻译失败: {}", e);
                for &idx in indices {
                    blocks[idx].translated_text = "[翻译失败]".to_string();
                }
            }
        }
        // 批次间延迟，避免触发限流

        sleep(Duration::from_millis(800));
    }

    let output_content = generate_bilingual_srt(&blocks);
    fs::write(output_path, output_content)
        .with_context(|| format!("无法写入文件: {}", output_path))?;

    info!("\n🎉 翻译完成！双语字幕已保存至: {}", output_path);

    Ok(output_path.to_string())
}
