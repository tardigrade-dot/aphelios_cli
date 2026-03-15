//! 音频播放工具 - 使用 rodio，真正的异步播放
//!
//! 提供流畅的音频播放功能，合成和播放完全并行

use rodio::{OutputStream, Sink, Source, buffer::SamplesBuffer};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// 音频播放器 - 真正的异步播放
pub struct AudioPlayer {
    sender: mpsc::SyncSender<Option<Vec<f32>>>,
    _handle: thread::JoinHandle<()>,
    /// 可选的录音路径，用于调试
    record_path: Option<String>,
}

impl AudioPlayer {
    /// 创建新的音频播放器
    pub fn new(sample_rate: u32) -> Result<Self, anyhow::Error> {
        Self::with_recording(sample_rate, None)
    }

    /// 创建新的音频播放器，并可选录制输出到 WAV 文件
    pub fn with_recording(
        sample_rate: u32,
        record_path: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        // 使用有缓冲的 channel，允许最多 5 个待播放的音频块在队列中
        // 这样可以防止播放线程暂时阻塞时导致主线程的 queue() 调用失败
        // 较小的缓冲区可以减少延迟，但太大会导致内存压力
        let (sender, receiver) = mpsc::sync_channel(5);
        let record_path_owned = record_path.map(|s| s.to_string());
        let record_path_for_struct = record_path_owned.clone();

        let handle = thread::spawn(move || {
            if let Err(e) = Self::playback_loop(receiver, sample_rate, record_path_owned) {
                tracing::error!("音频播放错误：{}", e);
            }
        });

        Ok(Self {
            sender,
            _handle: handle,
            record_path: record_path_for_struct,
        })
    }

    /// 播放音频数据 - 非阻塞，立即返回
    pub fn queue(&self, samples: Vec<f32>) -> Result<(), anyhow::Error> {
        // 尝试发送，如果通道已满则阻塞等待
        // 这可以防止无限缓冲导致内存爆炸
        self.sender.send(Some(samples))?;
        Ok(())
    }

    /// 等待所有音频播放完成
    pub fn finish(self) -> Result<(), anyhow::Error> {
        // 发送结束信号
        let _ = self.sender.send(None);
        // 等待播放线程完成
        drop(self);
        Ok(())
    }

    /// 播放循环 - 连续追加，不等待
    fn playback_loop(
        receiver: mpsc::Receiver<Option<Vec<f32>>>,
        sample_rate: u32,
        record_path: Option<String>,
    ) -> Result<(), anyhow::Error> {
        // 获取输出流
        let (_stream, stream_handle) = OutputStream::try_default()?;
        tracing::info!("音频设备已初始化，采样率：{}Hz", sample_rate);

        // 创建 Sink
        let sink = Sink::try_new(&stream_handle)?;
        sink.set_volume(1.0);

        // 可选的 WAV 录制器
        let mut wav_writer: Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>> = None;
        if let Some(ref path) = record_path {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            match hound::WavWriter::create(path, spec) {
                Ok(w) => {
                    wav_writer = Some(w);
                    tracing::info!("开始录制音频到：{}", path);
                }
                Err(e) => {
                    tracing::warn!("无法创建 WAV 文件 {}: {}", path, e);
                }
            }
        }

        let mut total_samples = 0;
        let mut first = true;

        // 持续接收并播放数据
        while let Ok(maybe_samples) = receiver.recv() {
            match maybe_samples {
                Some(samples) => {
                    if samples.is_empty() {
                        continue;
                    }

                    total_samples += samples.len();

                    // 录制到 WAV 文件
                    if let Some(ref mut writer) = wav_writer {
                        for &sample in &samples {
                            let _ = writer.write_sample(sample);
                        }
                    }

                    // 追加到播放队列
                    let source = SamplesBuffer::new(1, sample_rate, samples);
                    sink.append(source);

                    if first {
                        tracing::info!(
                            "▶️ 开始播放：{} 样本 ({:.2}秒)",
                            total_samples,
                            total_samples as f64 / sample_rate as f64
                        );
                        first = false;
                    }
                }
                None => {
                    // 没有更多数据了
                    tracing::info!(
                        "⏹ 收到结束信号，累计：{} 样本 ({:.2}s)",
                        total_samples,
                        total_samples as f64 / sample_rate as f64
                    );
                    break;
                }
            }
        }

        // 等待所有音频播放完成
        if total_samples > 0 {
            tracing::info!(
                "⏳ 等待播放完成：{} 样本，{:.2}秒，队列中：{} 个源",
                total_samples,
                total_samples as f64 / sample_rate as f64,
                sink.len()
            );

            // 等待播放完成
            sink.sleep_until_end();
            tracing::info!("✅ 播放完成！");
        }

        // 完成 WAV 录制
        if let Some(writer) = wav_writer {
            let _ = writer.finalize();
            if let Some(ref path) = record_path {
                tracing::info!("WAV 文件已保存到：{}", path);
            }
        }

        Ok(())
    }
}

/// 便捷的流式播放函数
pub fn play_streaming<I>(chunks: I, sample_rate: u32) -> Result<(), anyhow::Error>
where
    I: Iterator<Item = Vec<f32>>,
{
    let player = AudioPlayer::new(sample_rate)?;

    for chunk in chunks {
        player.queue(chunk)?;
    }

    player.finish()
}
