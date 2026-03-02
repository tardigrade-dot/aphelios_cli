//! Qwen3-TTS 流式语音合成示例 - 真正的边合成边播放
//!
//! 合成和播放完全并行，播放几乎不等待合成

use aphelios_core::utils::core_utils;
use aphelios_tts::{AudioBuffer, AudioPlayer, Language, Qwen3TTS, SynthesisOptions};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    core_utils::init_tracing();
    let device = core_utils::get_default_device(false)?;

    let model_path = "/Volumes/sw/pretrained_models/Qwen3-TTS-12Hz-0.6B-Base";
    let ref_audio_path = "/Users/larry/coderesp/aphelios_cli/test_data/newvoice.wav";
    let ref_text = "写这本书的目的在于通过我的走访和观察";
    let text_to_speech = r"近来研究表明，那种庞大、复杂、联合式的宗族在中国并不普遍，可能只存在于华南及江南的某些地区。研究者发现，北方那样的多族共居村庄在新界殖民地仍很普遍。对旧的宗族研究范式的批评甚至比这些论点还要深入。斯蒂芬·桑格伦对莫利斯·弗雷德曼以及休·贝克提出的男系宗族占主要地位的观点提出激烈的批评，
    
    他认为以前对中国宗族的研究只是从各种规定及法理思想出发的，而未分析宗族的实际结构、职能及运作效果。对宗族集团实际作为的探究不仅可以揭示社会分析学家从中国男系意识形态中借用来的观念具体化和理论化，而且可以揭示宗族是如何融合于中国社会的组织体系之中的。

上述观点为我们研究华北地区宗族组织提供了新的理论基础和方法。摆脱一族统治村庄的旧思想，我们便会发现，北方宗族并不是苍白无力的，虽然它并不庞大、复杂，并未拥有巨额族产、强大的同族意识，但在乡村社会中，它仍起着具体而重要的作用。我这里使用的宗族是一个广义的概念:它是由同一祖先繁衍下来的人群，通常由共同财产和婚丧庆吊联系在一起，并且居住于同一村庄。

同时，我还将证明过去对中国的宗族的认识，特别是其从历史上考察中国的宗族观点，仍有其合理的一面，值得借鉴，过分强调宗族的现实职能往往会诱使我们忽视宗族在历史上的重要作用。考察中国的男系意识，我觉得它与帕翠莎·艾伯瑞所称的“宗”更为接近。从布劳代尔的观点来看，男系嫡传并不是构成北方农村血缘团体的唯一因素。但是，如同离开具体变种就无法理解血缘团体和宗族组织的职能一样，不弄清楚宗族的官方定义或模式也无法掌握其具体作用。
";

    tracing::info!("📥 加载 Qwen3-TTS 模型...");
    let model = Qwen3TTS::from_pretrained(model_path, device)?;

    // 加载参考音频进行语音克隆
    tracing::info!("📥 加载参考音频：{}", ref_audio_path);
    let ref_audio = AudioBuffer::load(ref_audio_path)?;
    tracing::info!(
        "📊 参考音频：{} 样本，{:.2}秒",
        ref_audio.samples.len(),
        ref_audio.samples.len() as f64 / 24000.0
    );

    tracing::info!("🎯 创建语音克隆提示 (ICL 模式)...");
    let prompt = model.create_voice_clone_prompt(&ref_audio, Some(ref_text))?;
    tracing::info!("✅ 语音克隆提示已创建");

    // 配置流式合成选项 - 优化为小 chunk，实现流畅的边合成边播放
    let options = SynthesisOptions {
        chunk_frames: 20, // 5 帧 = 400ms，更小的 chunk 让播放更及时
        max_length: 20480,
        temperature: 0.9,
        top_k: 50,
        top_p: 0.9,
        repetition_penalty: 1.05,
        min_new_tokens: 2,
        seed: Some(42),
        ..Default::default()
    };

    tracing::info!("🚀 开始流式语音合成...");
    let total_start = Instant::now();

    // 创建音频播放器 - 预缓冲 1 秒后立即开始播放
    tracing::info!("🔊 初始化音频播放器...");
    let player = AudioPlayer::new(24000)?;
    tracing::info!("✅ 音频播放器已就绪（预缓冲 1 秒后开始播放）");

    // 创建流式合成会话
    tracing::info!("📝 创建流式合成会话...");
    let session = model.synthesize_voice_clone_streaming(
        text_to_speech,
        &prompt,
        Language::Chinese,
        options,
    )?;

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
