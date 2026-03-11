use anyhow::{Ok, Result};
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
        CreateChatCompletionRequestArgs,
    },
};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use tokio::{
    fs::File,
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter},
};
use tracing::info;

const MAX_TOKENS: u32 = 4096;
const MSG_PREFIX: &str = "zh-Hant|zh-Hans|";

/// 估算文本的 token 数量（简单估算：中文字符数 + 英文单词数）
fn estimate_tokens(text: &str) -> usize {
    let chinese_chars = text.chars().filter(|c| *c as u32 >= 0x4E00).count();
    let english_words = text.split_whitespace().count();
    chinese_chars + english_words
}

pub async fn translate_file_zh_hant_zh_hans(txt_path: &str) -> Result<()> {
    // 修正路径处理逻辑
    let path = Path::new(txt_path);

    let metadata = tokio::fs::metadata(txt_path).await?;
    let total_size = metadata.len();
    // 2. 初始化进度条
    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
        .progress_chars("#>-"));

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let target_txt_path = path.with_file_name(format!("{}_target.{}", stem, extension));

    // 使用 Tokio 的异步文件操作
    let file = File::open(txt_path).await?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let target_file = File::create(target_txt_path).await?;
    let mut writer = BufWriter::new(target_file);

    let mut batch: Vec<String> = Vec::new();
    let mut batch_tokens = 0;

    // Tokio 的 lines.next_line() 是异步的
    while let Some(line) = lines.next_line().await? {
        if line.trim().len() == 0 {
            continue;
        }
        pb.inc((line.len() + 1) as u64);
        let line_tokens = estimate_tokens(&line);
        // 如果当前 batch 加上这行会超过 token 限制，先处理当前 batch
        if batch_tokens + line_tokens > 4000 && !batch.is_empty() {
            let results = translate_zh_hant_zh_hans(batch.iter().map(|s| s).collect()).await?;
            for r in results {
                info!("model result : {}", r);
                writer.write_all(format!("{}\n", r).as_bytes()).await?;
            }
            batch.clear();
            batch_tokens = 0;
        }

        batch_tokens += line_tokens;
        batch.push(line);
    }
    // 处理剩余的 batch
    if !batch.is_empty() {
        let results = translate_zh_hant_zh_hans(batch.iter().map(|s| s).collect()).await?;
        for r in results {
            info!("model result : {}", r);
            writer.write_all(format!("{}\n", r).as_bytes()).await?;
        }
    }

    // 异步刷新缓冲区并关闭文件
    writer.flush().await?;
    pb.finish_with_message("完成！");
    Ok(())
}
pub async fn translate_zh_hant_zh_hans<S>(inputs: Vec<S>) -> Result<Vec<String>>
where
    S: AsRef<str> + Send + Clone,
{
    let api_base = "http://localhost:1234/v1";
    let model_id = "translategemma-4b-it";
    let config = OpenAIConfig::new().with_api_base(api_base);
    let client = Client::with_config(config);

    // 按 token 数量分组，每组不超过 MAX_TOKENS
    let mut batches: Vec<Vec<String>> = Vec::new();
    let mut current_batch: Vec<String> = Vec::new();
    let mut current_tokens = 0;

    for input in inputs {
        let input_str = input.as_ref().to_string();
        let tokens = estimate_tokens(&input_str);

        if current_tokens + tokens > MAX_TOKENS as usize && !current_batch.is_empty() {
            batches.push(current_batch);
            current_batch = Vec::new();
            current_tokens = 0;
        }

        current_tokens += tokens;
        current_batch.push(input_str);
    }

    if !current_batch.is_empty() {
        batches.push(current_batch);
    }

    info!("split into {} batches", batches.len());

    // 并发处理每个 batch
    let futures = batches.into_iter().map(|batch| {
        let client = client.clone();
        async move {
            // 将 batch 中的多条数据合并成一条消息，用换行符分隔
            let input_text = batch.join("\n");
            info!("input: {}", input_text);

            let messages: Vec<ChatCompletionRequestMessage> = vec![
                ChatCompletionRequestUserMessage::from(format!("{}{}", MSG_PREFIX, input_text))
                    .into(),
            ];
            let request = CreateChatCompletionRequestArgs::default()
                .max_tokens(MAX_TOKENS)
                .model(model_id)
                .messages(messages)
                .build()?;

            let response = client.chat().create(request).await?;

            let content = response
                .choices
                .into_iter()
                .next()
                .and_then(|cc| cc.message.content)
                .unwrap_or_default();

            // 按行分割结果
            let results: Vec<String> = content
                .lines()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            Ok(results)
        }
    });

    let batch_results = futures_util::future::try_join_all(futures).await?;
    // 展平所有 batch 的结果
    Ok(batch_results.into_iter().flatten().collect())
}
