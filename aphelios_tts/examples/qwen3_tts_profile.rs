//! 性能分析版本 - 带详细计时

use aphelios_core::utils::core_utils;
use aphelios_tts::{AudioBuffer, Language, Qwen3TTS, SynthesisOptions};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    core_utils::init_tracing();
    let device = core_utils::get_default_device(false)?;

    println!("🔍 设备信息：{:?}", device);
    println!("📊 文本长度：{} 字\n", 900);

    let model_path = "/Volumes/sw/pretrained_models/Qwen3-TTS-12Hz-0.6B-Base";
    let ref_audio_path = "/Users/larry/coderesp/aphelios_cli/test_data/newvoice.wav";
    let ref_text = "写这本书的目的在于通过我的走访和观察";
    let text_to_speech =
        "近年来的研究表明，在中国文化中'家'的含义十分广泛，学者们对其所下定义也是五花八门";

    tracing::info!("加载模型...");
    let t0 = Instant::now();
    let model = Qwen3TTS::from_pretrained(model_path, device.clone())?;
    let load_time = t0.elapsed();
    tracing::info!("模型加载完成：{:.2}s", load_time.as_secs_f64());

    tracing::info!("加载参考音频...");
    let ref_audio = AudioBuffer::load(ref_audio_path)?;
    
    tracing::info!("创建语音克隆提示...");
    let t1 = Instant::now();
    let prompt = model.create_voice_clone_prompt(&ref_audio, Some(ref_text))?;
    let prompt_time = t1.elapsed();
    tracing::info!("提示创建：{:.2}s", prompt_time.as_secs_f64());

    // 配置
    let options = SynthesisOptions {
        chunk_frames: 10,
        max_length: 2048,
        temperature: 0.9,
        top_k: 50,
        top_p: 0.9,
        repetition_penalty: 1.05,
        min_new_tokens: 2,
        seed: Some(42),
        ..Default::default()
    };

    tracing::info!("开始流式合成...");
    let session = model.synthesize_voice_clone_streaming(
        text_to_speech,
        &prompt,
        Language::Chinese,
        options,
    )?;

    let mut total_chunks = 0;
    let mut total_samples = 0;
    let mut chunk_times = Vec::new();
    let mut last_chunk_time = Instant::now();

    for result in session {
        let audio = result?;
        
        let now = Instant::now();
        let chunk_time = now.duration_since(last_chunk_time).as_secs_f64();
        chunk_times.push(chunk_time);
        last_chunk_time = now;

        let chunk_samples = audio.samples.len();
        let chunk_duration = chunk_samples as f64 / 24000.0;

        total_chunks += 1;
        total_samples += chunk_samples;

        let rtf = chunk_time / chunk_duration;
        tracing::info!(
            "[块 {}] 耗时：{:.2}s, 音频：{:.2}s, RTF: {:.2}x {}",
            total_chunks,
            chunk_time,
            chunk_duration,
            rtf,
            if rtf < 1.0 { "✅" } else { "⚠️" }
        );
    }

    let audio_duration = total_samples as f64 / 24000.0;
    
    println!("\n📊 性能分析:");
    println!("   总块数：{}", total_chunks);
    println!("   总样本：{}", total_samples);
    println!("   音频长度：{:.2}s", audio_duration);
    println!("   总耗时：{:.2}s", chunk_times.iter().sum::<f64>());
    println!("   平均 RTF: {:.2}x", chunk_times.iter().sum::<f64>() / audio_duration);
    println!("   平均块耗时：{:.2}s", chunk_times.iter().sum::<f64>() / chunk_times.len() as f64);
    
    // 分析 RTF 分布
    let rtfs: Vec<f64> = chunk_times.iter()
        .zip((0..).map(|i| {
            let samples = if i < total_chunks - 1 { 19200 } else { total_samples % 19200 };
            samples as f64 / 24000.0
        }))
        .map(|(t, d)| t / d)
        .collect();
    
    let avg_rtf = rtfs.iter().sum::<f64>() / rtfs.len() as f64;
    let max_rtf = rtfs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_rtf = rtfs.iter().cloned().fold(f64::INFINITY, f64::min);
    
    println!("\n   RTF 分布:");
    println!("      平均：{:.2}x", avg_rtf);
    println!("      最大：{:.2}x", max_rtf);
    println!("      最小：{:.2}x", min_rtf);
    println!("      <1.0: {} 块 ({:.1}%)", 
        rtfs.iter().filter(|&&r| r < 1.0).count(),
        rtfs.iter().filter(|&&r| r < 1.0).count() as f64 / rtfs.len() as f64 * 100.0
    );

    Ok(())
}
