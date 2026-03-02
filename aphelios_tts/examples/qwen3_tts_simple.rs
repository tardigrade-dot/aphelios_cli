//! 简单版本：先合成再播放
//!
//! 这个版本先将所有音频合成到一个文件中，然后播放，用于调试

use aphelios_core::utils::core_utils;
use aphelios_tts::{AudioBuffer, Language, Qwen3TTS, SynthesisOptions};
use rodio::{Sink, Source, OutputStream};

fn main() -> anyhow::Result<()> {
    core_utils::init_tracing();
    let device = core_utils::get_default_device(false)?;

    let model_path = "/Volumes/sw/pretrained_models/Qwen3-TTS-12Hz-0.6B-Base";
    let ref_audio_path = "/Users/larry/coderesp/aphelios_cli/test_data/newvoice.wav";
    let ref_text = "写这本书的目的在于通过我的走访和观察";
    let text_to_speech = "你好，这是语音克隆测试";

    tracing::info!("加载模型...");
    let model = Qwen3TTS::from_pretrained(model_path, device)?;

    tracing::info!("加载参考音频...");
    let ref_audio = AudioBuffer::load(ref_audio_path)?;
    
    tracing::info!("创建语音克隆提示...");
    let prompt = model.create_voice_clone_prompt(&ref_audio, Some(ref_text))?;

    tracing::info!("合成语音...");
    let audio = model.synthesize_voice_clone(
        text_to_speech,
        &prompt,
        Language::Chinese,
        Some(SynthesisOptions::default()),
    )?;

    tracing::info!("合成完成：{} 样本，{:.2}秒", audio.samples.len(), audio.samples.len() as f64 / 24000.0);

    // 保存为 WAV 文件
    audio.save("/tmp/tts_test.wav")?;
    tracing::info!("已保存到 /tmp/tts_test.wav");

    // 直接播放
    tracing::info!("开始播放...");
    let (_stream, stream_handle) = OutputStream::try_default()?;
    let sink = Sink::try_new(&stream_handle)?;
    sink.set_volume(1.0);

    let source = rodio::buffer::SamplesBuffer::new(1, 24000, audio.samples.clone());
    sink.append(source);

    tracing::info!("播放中...");
    sink.sleep_until_end();
    tracing::info!("播放完成！");

    Ok(())
}
