//! Qwen3-TTS CustomVoice 流式语音合成示例 - 使用预设说话人进行边合成边播放
//!
//! 这个示例展示了如何使用 CustomVoice 模型的预设说话人进行流式语音合成。
//! 合成和播放完全并行，播放几乎不等待合成。
//!
//! ## 用法
//!
//! ```bash
//! # 使用 CustomVoice 模型 (预设说话人)
//! cargo run --example qwen3_tts_streaming_custom_voice -- \
//!     --model /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice \
//!     --speaker serena
//!
//! # 可用的说话人：ryan, aiden, serena, vivian, unclefu, sohee, etc.
//! ```

use aphelios_core::utils::core_utils;
use aphelios_tts::{AudioPlayer, Language, Qwen3TTS, Speaker, SynthesisOptions};
use std::env;
use std::time::Instant;

fn print_help() {
    println!("Qwen3-TTS CustomVoice 流式语音合成示例");
    println!();
    println!("用法：cargo run --example qwen3_tts_streaming_custom_voice [选项]");
    println!();
    println!("选项:");
    println!("  --sdpa, -s       启用 SDPA (默认)");
    println!("  --no-sdpa, -n    禁用 SDPA");
    println!("  --model, -m      CustomVoice 模型路径");
    println!("  --speaker, -p    说话人 (ryan, aiden, serena, vivian, unclefu, sohee 等)");
    println!("  --text, -t       要合成的文本");
    println!("  --help, -h       显示帮助");
    println!();
    println!("可用的说话人:");
    println!("  - ryan: 英语男声");
    println!("  - aiden: 英语男声");
    println!("  - serena: 中文女声 (推荐)");
    println!("  - vivian: 英语女声");
    println!("  - unclefu: 中文男声");
    println!("  - sohee: 韩语女声");
}

fn parse_speaker(speaker_str: &str) -> anyhow::Result<Speaker> {
    match speaker_str.to_lowercase().as_str() {
        "ryan" => Ok(Speaker::Ryan),
        "aiden" => Ok(Speaker::Aiden),
        "serena" => Ok(Speaker::Serena),
        "vivian" => Ok(Speaker::Vivian),
        "unclefu" => Ok(Speaker::UncleFu),
        "sohee" => Ok(Speaker::Sohee),
        _ => anyhow::bail!(
            "未知的说话人：{}. 可用的说话人：ryan, aiden, serena, vivian, unclefu, sohee",
            speaker_str
        ),
    }
}

fn main() -> anyhow::Result<()> {
    core_utils::init_tracing();
    let device = core_utils::get_default_device(false)?;

    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    let mut use_sdpa = true;
    let mut model_path = "/Volumes/sw/pretrained_models/Qwen3-TTS-12Hz-1.7B-CustomVoice";
    let mut speaker_str = "serena";
    let mut text_to_speech = r"

    近来研究表明，那种庞大、复杂、联合式的宗族在中国并不普遍，可能只存在于华南及江南的某些地区。
    研究者发现，北方那样的多族共居村庄在新界殖民地仍很普遍。对旧的宗族研究范式的批评甚至比这些论点还要深入。
    斯蒂芬·桑格伦对莫利斯·弗雷德曼以及休·贝克提出的男系宗族占主要地位的观点提出激烈的批评，

    他认为以前对中国宗族的研究只是从各种规定及法理出发的，而未分析宗族的实际结构、职能及运作效果。
    对宗族集团实际作为的探究不仅可以揭示社会分析学家从中国男系意识形态中借用来的观念具体化和理论化，
    而且可以揭示宗族是如何融合于中国社会的组织体系之中的。

    ";

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--no-sdpa" | "-n" => use_sdpa = false,
            "--sdpa" | "-s" => use_sdpa = true,
            "--model" | "-m" => {
                if i + 1 < args.len() {
                    model_path = &args[i + 1];
                    i += 1;
                }
            }
            "--speaker" | "-p" => {
                if i + 1 < args.len() {
                    speaker_str = &args[i + 1];
                    i += 1;
                }
            }
            "--text" | "-t" => {
                if i + 1 < args.len() {
                    text_to_speech = &args[i + 1];
                    i += 1;
                }
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    // Parse speaker
    let speaker = parse_speaker(speaker_str)?;

    tracing::info!("📥 加载 Qwen3-TTS CustomVoice 模型...");
    let model = Qwen3TTS::from_pretrained(model_path, device)?;

    // Verify model supports preset speakers
    if !model.supports_preset_speakers() {
        anyhow::bail!(
            "当前模型不支持预设说话人。请使用 CustomVoice 模型 (如 Qwen3-TTS-12Hz-1.7B-CustomVoice)"
        );
    }

    tracing::info!("✅ 模型已加载，使用说话人：{:?}", speaker);

    // 配置流式合成选项 - 优化为小 chunk，实现流畅的边合成边播放
    let options = SynthesisOptions {
        chunk_frames: 20, // 20 帧 = 1.6s，更小的 chunk 让播放更及时
        max_length: 20480,
        temperature: 0.9,
        top_k: 20,
        top_p: 0.9,
        repetition_penalty: 1.05,
        min_new_tokens: 2,
        seed: Some(42),
        use_sdpa,
        ..Default::default()
    };

    tracing::info!("🚀 开始流式语音合成...");
    tracing::info!("📊 SDPA: {}", if use_sdpa { "启用" } else { "禁用" });
    let total_start = Instant::now();

    // 创建音频播放器 - 预缓冲 1 秒后立即开始播放
    tracing::info!("🔊 初始化音频播放器...");
    let player = AudioPlayer::new(24000)?;
    tracing::info!("✅ 音频播放器已就绪（预缓冲 1 秒后开始播放）");

    // 创建流式合成会话 - 使用预设说话人
    tracing::info!("📝 创建流式合成会话 (说话人：{:?})...", speaker);
    let session =
        model.synthesize_streaming(text_to_speech, speaker, Language::Chinese, options)?;

    let mut total_chunks = 0;
    let mut total_samples = 0;
    let mut first_chunk_time = None;
    let mut last_queue_time = Instant::now();

    // 流式合成并播放 - 完全并行
    tracing::info!("▶️ 开始合成和播放（并行模式）...");
    for result in session {
        let audio = result?;

        if first_chunk_time.is_none() {
            first_chunk_time = Some(Instant::now());
            tracing::info!("🎵 收到第一个音频块，发送到播放器...");
        }

        let chunk_samples = audio.samples.len();
        let chunk_duration = chunk_samples as f64 / 24000.0;

        total_chunks += 1;
        total_samples += chunk_samples;

        // 立即发送到播放器（不等待播放完成）
        let queue_start = Instant::now();
        player.queue(audio.samples.clone())?;
        let queue_time = queue_start.duration_since(last_queue_time);

        tracing::info!(
            "📦 [块 {}] {} 样本 ({:.2}ms), 累计：{:.2}s, 合成间隔：{:.2}s",
            total_chunks,
            chunk_samples,
            chunk_duration * 1000.0,
            total_samples as f64 / 24000.0,
            queue_time.as_secs_f64()
        );

        last_queue_time = queue_start;
    }

    let synthesis_time = total_start.elapsed();
    tracing::info!(
        "\n✅ 合成完成！总计：{} 块，{} 样本，{:.2}s 音频，耗时：{:.2}s",
        total_chunks,
        total_samples,
        total_samples as f64 / 24000.0,
        synthesis_time.as_secs_f64()
    );

    if let Some(first_time) = first_chunk_time {
        let first_chunk_delay = first_time.duration_since(total_start);
        tracing::info!(
            "⏱ 首块延迟：{:.2}s (模型加载 + 预填充 + 第一块合成)",
            first_chunk_delay.as_secs_f64()
        );
    }

    // 等待播放完成
    tracing::info!("\n🔊 等待播放完成...");
    player.finish()?;

    let elapsed = total_start.elapsed();
    let audio_duration = total_samples as f64 / 24000.0;
    let rtf = synthesis_time.as_secs_f64() / audio_duration;

    // 计算播放等待时间
    let playback_wait = elapsed.as_secs_f64() - synthesis_time.as_secs_f64();

    tracing::info!("\n🎉 全部完成！");
    tracing::info!("   总耗时：{:.2}s", elapsed.as_secs_f64());
    tracing::info!("   合成时间：{:.2}s", synthesis_time.as_secs_f64());
    tracing::info!("   音频长度：{:.2}s", audio_duration);
    tracing::info!(
        "   播放等待：{:.2}s {}",
        playback_wait,
        if playback_wait < 1.0 {
            "✅ 几乎无需等待"
        } else if playback_wait < 5.0 {
            "⚠️ 少量等待"
        } else {
            "❌ 大量等待 - 合成太慢"
        }
    );
    tracing::info!(
        "   实时率 (RTF): {:.2}x {}",
        rtf,
        if rtf < 1.0 {
            "✅ 实时合成"
        } else if rtf < 2.0 {
            "⚠️ 合成稍慢"
        } else {
            "❌ 合成很慢"
        }
    );

    // 分析合成和播放的重叠情况
    let overlap_ratio = audio_duration / synthesis_time.as_secs_f64();
    tracing::info!("\n📊 并行度分析:");
    tracing::info!(
        "   重叠率：{:.1}% (合成时播放的进度)",
        overlap_ratio * 100.0
    );
    tracing::info!(
        "   说明：合成进行到 {:.1}% 时，播放已经开始",
        (1.0 - overlap_ratio) * 100.0
    );

    Ok(())
}
