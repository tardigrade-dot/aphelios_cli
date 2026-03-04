use std::{
    collections::VecDeque,
    io::Write,
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::Result;
use aphelios_tts::audio::player::AudioPlayer;
use vibevoice::{Device, ModelVariant, Progress, VibeVoice};

/// 带缓冲的流式播放器
/// 先积累足够的音频再开始播放，避免卡顿
struct BufferedPlayer {
    player: AudioPlayer,
    buffer: Arc<Mutex<(VecDeque<Vec<f32>>, usize, bool)>>, // (buffer, samples_count, started)
    buffer_duration_secs: f64,
    sample_rate: u32,
}

impl BufferedPlayer {
    fn new(sample_rate: u32, buffer_duration_secs: f64) -> Result<Self> {
        let player = AudioPlayer::new(sample_rate)?;
        let buffer = Arc::new(Mutex::new((VecDeque::new(), 0, false)));
        Ok(Self {
            player,
            buffer,
            buffer_duration_secs,
            sample_rate,
        })
    }

    /// 获取缓冲状态用于回调
    fn get_buffer_state(&self) -> (usize, bool) {
        let guard = self.buffer.lock().unwrap();
        (guard.1, guard.2)
    }

    /// 添加音频块到播放器
    fn add_chunk(&self, samples: Vec<f32>) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }

        let mut guard = self.buffer.lock().unwrap();
        let (buffer, total_samples, started) = &mut *guard;

        if !*started {
            // 缓冲中
            buffer.push_back(samples);
            *total_samples += buffer.back().unwrap().len();

            let buffered_secs = *total_samples as f64 / self.sample_rate as f64;
            print!("\r  缓冲中：{:.1}s / {:.1}s", buffered_secs, self.buffer_duration_secs);
            let _ = std::io::stdout().flush();

            // 缓冲足够后开始播放
            if buffered_secs >= self.buffer_duration_secs {
                println!("\r  缓冲完成，开始播放！");
                while let Some(chunk) = buffer.pop_front() {
                    self.player.queue(chunk)?;
                }
                *total_samples = 0;
                *started = true;
            }
        } else {
            // 已经开始播放，直接送入播放器
            drop(guard);
            self.player.queue(samples)?;
        }
        Ok(())
    }

    fn finish(self) -> Result<()> {
        // 播放剩余的缓冲
        let guard = self.buffer.lock().unwrap();
        let (buffer, _, started) = &*guard;
        if !*started {
            for chunk in buffer.iter() {
                self.player.queue(chunk.clone())?;
            }
        }
        drop(guard);
        self.player.finish()?;
        Ok(())
    }
}

fn main() -> Result<()> {
    let model_path = "/Volumes/sw/pretrained_models/VibeVoice-Realtime-0.5B";
    let mut vv = VibeVoice::builder()
        .model_path(model_path)
        .variant(ModelVariant::Realtime)
        .device(Device::Metal)
        .diffusion_steps(5)
        .build()?;

    let gen_start = Instant::now();

    let voice_cache_path = "/Users/larry/coderesp/aphelios_cli/test_data/en-Carter_man.safetensors";
    let text = "Hello! This is the realtime streaming model. It generates audio chunk by chunk for low latency applications.";

    // 创建带缓冲的播放器 (24kHz 采样率，先缓冲 3 秒)
    let player = Arc::new(BufferedPlayer::new(24000, 3.0)?);
    let player_weak = Arc::downgrade(&player);

    // 用于统计生成的音频块数量
    let mut chunk_count = 0;
    let mut total_samples = 0;

    let audio = vv.synthesize_with_callback(
        text,
        Some(voice_cache_path),
        Some(Box::new(move |progress: Progress| {
            chunk_count += 1;
            if let Some(ref chunk) = progress.audio_chunk {
                let samples = chunk.samples().to_vec();
                total_samples += samples.len();
                // 将音频块送入播放器
                if let Some(player_ref) = player_weak.upgrade() {
                    if let Err(e) = player_ref.add_chunk(samples) {
                        eprintln!("播放错误：{}", e);
                    }
                }
            }
            if let Some(player_ref) = player_weak.upgrade() {
                let (_, started) = player_ref.get_buffer_state();
                if started {
                    print!(
                        "\r  已生成并播放：{} 块 ({:.1}s 音频)",
                        chunk_count,
                        total_samples as f64 / 24000.0
                    );
                    let _ = std::io::stdout().flush();
                }
            }
        })),
    )?;

    let gen_time = gen_start.elapsed();
    let rtf = audio.duration_secs() / gen_time.as_secs_f32();

    println!();
    println!();

    // 等待所有音频播放完成
    Arc::try_unwrap(player)
        .map_err(|_| anyhow::anyhow!("Failed to unwrap player"))?
        .finish()?;

    audio.save_wav("output/streaming_output.wav")?;

    println!("---");
    println!("Audio duration: {:.1}s", audio.duration_secs());
    println!("Generation time: {:.1}s", gen_time.as_secs_f32());
    println!("Real-time factor: {:.2}x", rtf);
    println!("Saved to: output/streaming_output.wav");

    if rtf > 1.0 {
        println!();
        println!("Faster than real-time - suitable for streaming!");
    }

    Ok(())
}
