//! 音频文件加载器
//!
//! 支持多种音频格式，使用 symphonia 进行解码

use anyhow::{bail, Result};
use hound::{SampleFormat, WavReader};
use tracing::info;
use std::path::Path;

use symphonia::core::audio::Signal;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphErr;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::{Limit, MetadataOptions};
use symphonia::core::probe::Hint;

use super::types::{AudioBuffer, MonoBuffer};

/// 目标采样率（所有加载的音频都会 resample 到这个值）
const TARGET_SAMPLE_RATE: u32 = 16000;

/// 音频格式信息
#[derive(Debug, Clone)]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub sample_format: SampleFormat,
    pub duration_secs: f64,
}

/// 音频加载器
pub struct AudioLoader {
    normalize: bool,
}

impl AudioLoader {
    pub fn new() -> Self {
        Self { normalize: true }
    }

    /// 是否归一化音频样本到 [-1, 1] 范围
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// 从文件加载音频，统一返回 16kHz 单声道 AudioBuffer
    pub fn load(&self, path: impl AsRef<Path>) -> Result<AudioBuffer> {
        let path = path.as_ref();

        if !path.exists() {
            bail!("Audio file not found: {}", path.display());
        }

        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "wav" => self.load_wav(path),
            _ => self.load_with_symphonia(path),
        }
    }

    /// 使用 symphonia 加载音频文件（支持 mp4, mp3, m4a, flac, mkv, mov 等）
    fn load_with_symphonia(&self, path: &Path) -> Result<AudioBuffer> {
        let file = std::fs::File::open(path)?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        let metadata_options = MetadataOptions {
            limit_metadata_bytes: Limit::Maximum(0), // Disable metadata reading to avoid ID3v2 issues
            ..Default::default()
        };
        let probed = symphonia::default::get_probe().format(
            &hint,
            mss,
            &FormatOptions::default(),
            &metadata_options,
        )?;

        let mut format = probed.format;

        for t in format.tracks() {
            info!(
                "track id={} codec={:?} channels={:?} sample_rate={:?}",
                t.id,
                t.codec_params.codec,
                t.codec_params.channels,
                t.codec_params.sample_rate,
            );
        }
        let track = format
            .tracks()
            .iter()
            .find(|t| {
                    t.codec_params.codec != CODEC_TYPE_NULL
                        && t.codec_params.sample_rate.is_some()
                })
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no supported audio tracks"))?;

        let src_rate = track
            .codec_params
            .sample_rate
            .ok_or_else(|| anyhow::anyhow!("missing sample rate"))?;

        // Some codecs (e.g. AAC in MP4) may not report channel count in container metadata.
        // In practice, most multi-channel content is stereo (2 channels).
        let channels = track
            .codec_params
            .channels
            .map(|c| c.count())
            .unwrap_or(2);

        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())?;

        let mut per_channel: Vec<Vec<f32>> = vec![Vec::new(); channels];

        loop {
            let packet = match format.next_packet() {
                Ok(p) => p,
                Err(SymphErr::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            };

            if packet.track_id() != track.id {
                continue;
            }

            let decoded = match decoder.decode(&packet) {
                Ok(s) => s,
                Err(SymphErr::DecodeError(_)) => continue,
                Err(e) => return Err(e.into()),
            };

            let spec_val = *decoded.spec();

            let chans = decoded.spec().channels.count();
            // Adjust per_channel to match the actual decoded channel count.
            // Some containers (e.g. some AAC in MP4) may not report channels,
            // so we use a default (2) and correct it after the first decoded frame.
            if chans > per_channel.len() {
                per_channel.resize_with(chans, Vec::new);
            } else if chans < per_channel.len() {
                per_channel.truncate(chans);
            }
            match decoded {
                symphonia::core::audio::AudioBufferRef::F32(buf) => {
                    for ch in 0..chans {
                        per_channel[ch].extend(buf.chan(ch));
                    }
                }
                other => {
                    let mut buf = symphonia::core::audio::AudioBuffer::<f32>::new(
                        other.capacity() as u64,
                        spec_val,
                    );
                    other.convert(&mut buf);
                    for ch in 0..chans {
                        per_channel[ch].extend(buf.chan(ch));
                    }
                }
            }
        }

        // Downmix to mono
        let mono = downmix_to_mono(per_channel);

        // Resample to target rate (if needed)
        let mono = if src_rate != TARGET_SAMPLE_RATE {
            resample_linear(&mono, src_rate, TARGET_SAMPLE_RATE)?
        } else {
            mono
        };

        Ok(AudioBuffer::Mono(MonoBuffer::new(mono, TARGET_SAMPLE_RATE)))
    }

    /// 加载 WAV 文件（保持原有逻辑，添加 resample 支持）
    fn load_wav(&self, path: &Path) -> Result<AudioBuffer> {
        let mut reader = WavReader::open(path)?;
        let spec = reader.spec();

        let samples = self.read_samples(&mut reader, spec)?;

        // 如果不是单声道，分离左右声道并 downmix
        let mono = if spec.channels == 1 {
            samples
        } else {
            // 分离并 downmix
            let left: Vec<f32> = samples.iter().step_by(2).copied().collect();
            let right: Vec<f32> = samples.iter().skip(1).step_by(2).copied().collect();
            downmix_two_channels(&left, &right)
        };

        // Resample if needed
        let mono = if spec.sample_rate != TARGET_SAMPLE_RATE {
            resample_linear(&mono, spec.sample_rate, TARGET_SAMPLE_RATE)?
        } else {
            mono
        };

        Ok(AudioBuffer::Mono(MonoBuffer::new(mono, TARGET_SAMPLE_RATE)))
    }

    /// 读取音频样本
    fn read_samples<R: std::io::Read>(
        &self,
        reader: &mut WavReader<R>,
        spec: hound::WavSpec,
    ) -> Result<Vec<f32>> {
        match spec.sample_format {
            SampleFormat::Int => self.read_int_samples(reader, spec.bits_per_sample),
            SampleFormat::Float => self.read_float_samples(reader),
        }
    }

    /// 读取整数样本
    fn read_int_samples<R: std::io::Read>(
        &self,
        reader: &mut WavReader<R>,
        bits: u16,
    ) -> Result<Vec<f32>> {
        match bits {
            8 => Ok(reader
                .samples::<i8>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / i8::MAX as f32)
                .collect()),
            16 => Ok(reader
                .samples::<i16>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / i16::MAX as f32)
                .collect()),
            24 | 32 => Ok(reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| match bits {
                    24 => s as f32 / 8_388_607.0, // 2^23 - 1
                    _ => s as f32 / i32::MAX as f32,
                })
                .collect()),
            _ => bail!("Unsupported bit depth: {}", bits),
        }
    }

    /// 读取浮点样本
    fn read_float_samples<R: std::io::Read>(&self, reader: &mut WavReader<R>) -> Result<Vec<f32>> {
        Ok(reader.samples::<f32>().filter_map(|s| s.ok()).collect())
    }

    /// 获取音频格式信息（仅支持 WAV）
    pub fn get_format(path: impl AsRef<Path>) -> Result<AudioFormat> {
        let reader = WavReader::open(path.as_ref())?;
        let spec = reader.spec();
        let len = reader.len() as f64;
        let duration = len / spec.sample_rate as f64;

        Ok(AudioFormat {
            sample_rate: spec.sample_rate,
            channels: spec.channels,
            bits_per_sample: spec.bits_per_sample,
            sample_format: spec.sample_format,
            duration_secs: duration,
        })
    }
}

impl Default for AudioLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// 将多声道 downmix 为单声道
fn downmix_to_mono(samples_per_channel: Vec<Vec<f32>>) -> Vec<f32> {
    if samples_per_channel.is_empty() {
        return Vec::new();
    }
    if samples_per_channel.len() == 1 {
        return samples_per_channel.into_iter().next().unwrap();
    }
    let num_channels = samples_per_channel.len();
    let num_samples = samples_per_channel
        .iter()
        .map(|c| c.len())
        .max()
        .unwrap_or(0);
    let mut mono = vec![0.0f32; num_samples];
    for ch in &samples_per_channel {
        for (i, sample) in ch.iter().enumerate() {
            mono[i] += *sample;
        }
    }
    let inv_channels = 1.0 / num_channels as f32;
    for sample in mono.iter_mut() {
        *sample *= inv_channels;
    }
    mono
}

/// 将两个声道 downmix 为单声道
fn downmix_two_channels(left: &[f32], right: &[f32]) -> Vec<f32> {
    let len = left.len().min(right.len());
    let mut mono = Vec::with_capacity(len);
    for i in 0..len {
        mono.push((left[i] + right[i]) * 0.5);
    }
    mono
}

/// 线性插值重采样
fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    if src_rate == dst_rate {
        return Ok(input.to_vec());
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let output_len = ((input.len() as f64) * ratio).ceil().max(1.0) as usize;
    let mut out = Vec::with_capacity(output_len);
    let last = input.len() - 1;

    for n in 0..output_len {
        let pos = (n as f64) / ratio;
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;
        let i0 = idx.min(last);
        let i1 = (idx + 1).min(last);
        let s0 = input[i0];
        let s1 = input[i1];
        out.push(s0 + (s1 - s0) * frac);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_format() {
        let test_file = "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4_16k.wav";
        if std::path::Path::new(test_file).exists() {
            let format = AudioLoader::get_format(test_file).unwrap();
            assert_eq!(format.sample_rate, 16000);
            assert_eq!(format.channels, 1);
        }
    }

    #[test]
    fn test_resample_linear() {
        let input = vec![1.0f32, 0.5, 0.0, -0.5, -1.0];
        // 16000 -> 8000 should halve the length approximately
        let output = resample_linear(&input, 16000, 8000).unwrap();
        assert!(output.len() > 0);
        assert!(output.len() < input.len());
    }
}
