//! 测试 rodio 音频播放
//!
//! 运行这个示例测试你的系统是否能正常播放声音

use rodio::{OutputStream, Sink, Source};
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    println!("🔊 测试音频播放...");

    // 获取输出流
    let (_stream, stream_handle) = OutputStream::try_default()?;
    println!("✅ 音频设备已初始化");

    // 创建 Sink
    let sink = Sink::try_new(&stream_handle)?;
    sink.set_volume(1.0);

    // 生成一个测试音（440Hz 正弦波，1 秒）
    println!("🎵 生成测试音 (440Hz, 1 秒)...");
    let sample_rate = 44100;
    let duration_secs = 1.0;
    let samples: Vec<f32> = (0..(sample_rate as f32 * duration_secs) as usize)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5
        })
        .collect();

    println!("📢 播放测试音...");
    let source = rodio::buffer::SamplesBuffer::new(1, sample_rate, samples);
    sink.append(source);

    // 等待播放完成
    sink.sleep_until_end();

    println!("✅ 播放完成！如果你听到了'滴'声，说明音频播放正常工作。");

    // 再测试一个不同频率的音
    println!("\n🎵 生成第二个测试音 (880Hz, 0.5 秒)...");
    let samples2: Vec<f32> = (0..(sample_rate as f32 * 0.5) as usize)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (t * 880.0 * 2.0 * std::f32::consts::PI).sin() * 0.5
        })
        .collect();

    println!("📢 播放第二个测试音...");
    let source2 = rodio::buffer::SamplesBuffer::new(1, sample_rate, samples2);
    sink.append(source2);
    sink.sleep_until_end();

    println!("✅ 全部测试完成！");

    Ok(())
}
