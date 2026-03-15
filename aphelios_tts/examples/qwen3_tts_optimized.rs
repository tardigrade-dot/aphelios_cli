//! 优化版本 - 分段合成减少内存压力

use aphelios_core::utils::core_utils;
use aphelios_tts::{AudioBuffer, AudioPlayer, Language, Qwen3TTS, SynthesisOptions};
use std::time::Instant;

/// 将长文本分成较短的段落
fn split_text(text: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    
    for char in text.chars() {
        current.push(char);
        if current.len() >= max_chars || char == '。' || char == '！' || char == '？' {
            if !current.trim().is_empty() {
                chunks.push(current.trim().to_string());
                current = String::new();
            }
        }
    }
    
    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }
    
    chunks
}

fn main() -> anyhow::Result<()> {
    core_utils::init_logging();
    let device = core_utils::get_default_device(false)?;

    let model_path = "/Volumes/sw/pretrained_models/Qwen3-TTS-12Hz-0.6B-Base";
    let ref_audio_path = "/Users/larry/coderesp/aphelios_cli/test_data/newvoice.wav";
    let ref_text = "写这本书的目的在于通过我的走访和观察";
    
    // 900+ 字的长文本
    let long_text = "近年来的研究表明，在中国文化中'家'的含义十分广泛，学者们对其所下定义也是五花八门...";

    tracing::info!("📥 加载模型...");
    let model = Qwen3TTS::from_pretrained(model_path, device)?;

    tracing::info!("📥 加载参考音频...");
    let ref_audio = AudioBuffer::load(ref_audio_path)?;
    
    tracing::info!("🎯 创建语音克隆提示...");
    let prompt = model.create_voice_clone_prompt(&ref_audio, Some(ref_text))?;

    // 分段处理长文本
    let segments = split_text(long_text, 100); // 每段最多 100 字
    tracing::info!("📝 文本已分成 {} 段", segments.len());

    // 创建播放器
    let player = AudioPlayer::new(24000)?;
    
    let total_start = Instant::now();

    // 逐段合成和播放
    for (i, segment) in segments.iter().enumerate() {
        tracing::info!("▶️ 合成第 {}/{} 段：{} 字", i + 1, segments.len(), segment.len());
        
        let options = SynthesisOptions {
            chunk_frames: 10,
            max_length: 512, // 限制每段最大长度
            temperature: 0.9,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.05,
            min_new_tokens: 2,
            seed: Some(42),
            ..Default::default()
        };

        let segment_start = Instant::now();
        
        let session = model.synthesize_voice_clone_streaming(
            segment,
            &prompt,
            Language::Chinese,
            options,
        )?;

        for result in session {
            let audio = result?;
            player.queue(audio.samples.clone())?;
        }

        let segment_time = segment_start.elapsed();
        tracing::info!("✅ 第 {} 段完成：{:.2}s", i + 1, segment_time.as_secs_f64());
    }

    player.finish()?;
    
    let elapsed = total_start.elapsed();
    tracing::info!("🎉 全部完成！总耗时：{:.2}s", elapsed.as_secs_f64());

    Ok(())
}
