//! 音频文件加载器
//! 
//! 支持多种音频格式和位深的加载

use anyhow::{Result, bail};
use hound::{WavReader, SampleFormat};
use std::path::Path;

use super::types::{AudioBuffer, MonoBuffer, StereoBuffer};

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

    /// 从文件加载音频
    pub fn load(&self, path: impl AsRef<Path>) -> Result<AudioBuffer> {
        let path = path.as_ref();
        
        if !path.exists() {
            bail!("Audio file not found: {}", path.display());
        }

        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "wav" => self.load_wav(path),
            _ => bail!("Unsupported audio format: {}", extension),
        }
    }

    /// 加载 WAV 文件
    fn load_wav(&self, path: &Path) -> Result<AudioBuffer> {
        let mut reader = WavReader::open(path)?;
        let spec = reader.spec();

        let samples = self.read_samples(&mut reader, spec)?;

        if spec.channels == 1 {
            Ok(AudioBuffer::Mono(MonoBuffer::new(samples, spec.sample_rate)))
        } else {
            // 分离左右声道
            let left: Vec<f32> = samples.iter().step_by(2).copied().collect();
            let right: Vec<f32> = samples.iter().skip(1).step_by(2).copied().collect();
            Ok(AudioBuffer::Stereo(StereoBuffer::new(left, right, spec.sample_rate)))
        }
    }

    /// 读取音频样本
    fn read_samples<R: std::io::Read>(&self, reader: &mut WavReader<R>, spec: hound::WavSpec) -> Result<Vec<f32>> {
        match spec.sample_format {
            SampleFormat::Int => self.read_int_samples(reader, spec.bits_per_sample),
            SampleFormat::Float => self.read_float_samples(reader),
        }
    }

    /// 读取整数样本
    fn read_int_samples<R: std::io::Read>(&self, reader: &mut WavReader<R>, bits: u16) -> Result<Vec<f32>> {
        match bits {
            8 => Ok(reader.samples::<i8>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / i8::MAX as f32)
                .collect()),
            16 => Ok(reader.samples::<i16>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / i16::MAX as f32)
                .collect()),
            24 | 32 => Ok(reader.samples::<i32>()
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
        Ok(reader.samples::<f32>()
            .filter_map(|s| s.ok())
            .collect())
    }

    /// 获取音频格式信息
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_format() {
        // This test requires actual audio files
        // Skip if file doesn't exist
        let test_file = "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4_16k.wav";
        if std::path::Path::new(test_file).exists() {
            let format = AudioLoader::get_format(test_file).unwrap();
            assert_eq!(format.sample_rate, 16000);
            assert_eq!(format.channels, 1);
        }
    }
}
