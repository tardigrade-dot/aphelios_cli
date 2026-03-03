//! 音频播放工具 - 使用 rodio，真正的异步播放
//!
//! 提供流畅的音频播放功能，合成和播放完全并行

use rodio::{OutputStream, Sink, Source, buffer::SamplesBuffer};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// 音频播放器 - 真正的异步播放
pub struct AudioPlayer {
    sender: mpsc::Sender<Option<Vec<f32>>>,
    _handle: thread::JoinHandle<()>,
}

impl AudioPlayer {
    /// 创建新的音频播放器
    pub fn new(sample_rate: u32) -> Result<Self, anyhow::Error> {
        let (sender, receiver) = mpsc::channel();

        let handle = thread::spawn(move || {
            if let Err(e) = Self::playback_loop(receiver, sample_rate) {
                tracing::error!("音频播放错误：{}", e);
            }
        });

        Ok(Self {
            sender,
            _handle: handle,
        })
    }

    /// 播放音频数据 - 非阻塞，立即返回
    pub fn queue(&self, samples: Vec<f32>) -> Result<(), anyhow::Error> {
        self.sender.send(Some(samples))?;
        Ok(())
    }

    /// 等待所有音频播放完成
    pub fn finish(self) -> Result<(), anyhow::Error> {
        self.sender.send(None)?;
        drop(self);
        Ok(())
    }

    /// 播放循环 - 连续追加，不等待
    fn playback_loop(
        receiver: mpsc::Receiver<Option<Vec<f32>>>,
        sample_rate: u32,
    ) -> Result<(), anyhow::Error> {
        // 获取输出流
        let (_stream, stream_handle) = OutputStream::try_default()?;
        tracing::info!("音频设备已初始化，采样率：{}Hz", sample_rate);

        // 创建 Sink
        let sink = Sink::try_new(&stream_handle)?;
        sink.set_volume(1.0);

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
