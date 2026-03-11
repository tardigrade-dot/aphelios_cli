//! 音频文件保存器

use anyhow::Result;
use hound::{WavSpec, WavWriter, SampleFormat};
use std::path::Path;

use super::types::{MonoBuffer, StereoBuffer};

/// 音频保存器
pub struct AudioSaver {
    bits_per_sample: u16,
}

impl AudioSaver {
    pub fn new() -> Self {
        Self {
            bits_per_sample: 16,
        }
    }

    pub fn with_bits_per_sample(mut self, bits: u16) -> Self {
        self.bits_per_sample = bits;
        self
    }

    /// 保存单声道音频到 WAV 文件
    pub fn save_mono(&self, audio: &MonoBuffer, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let spec = WavSpec {
            channels: 1,
            sample_rate: audio.sample_rate,
            bits_per_sample: self.bits_per_sample,
            sample_format: SampleFormat::Int,
        };

        let mut writer = WavWriter::create(path, spec)?;

        for &sample in &audio.samples {
            let int_sample = (sample * i16::MAX as f32) as i16;
            writer.write_sample(int_sample)?;
        }

        writer.finalize()?;
        Ok(())
    }

    /// 保存立体声音频到 WAV 文件
    pub fn save_stereo(&self, audio: &StereoBuffer, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let spec = WavSpec {
            channels: 2,
            sample_rate: audio.sample_rate,
            bits_per_sample: self.bits_per_sample,
            sample_format: SampleFormat::Int,
        };

        let mut writer = WavWriter::create(path, spec)?;

        let len = audio.left.len().min(audio.right.len());
        for i in 0..len {
            let left_sample = (audio.left[i] * i16::MAX as f32) as i16;
            let right_sample = (audio.right[i] * i16::MAX as f32) as i16;
            writer.write_sample(left_sample)?;
            writer.write_sample(right_sample)?;
        }

        writer.finalize()?;
        Ok(())
    }

    /// 保存音频片段（带时间戳前缀）
    pub fn save_with_prefix(
        &self,
        audio: &StereoBuffer,
        base_path: impl AsRef<Path>,
        prefix: &str,
    ) -> Result<String> {
        let base_path = base_path.as_ref();
        let path = base_path.with_file_name(format!(
            "{}_{}",
            prefix,
            base_path.file_name().unwrap().to_str().unwrap()
        ));
        
        self.save_stereo(audio, &path)?;
        Ok(path.to_string_lossy().into_owned())
    }
}

impl Default for AudioSaver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_save_mono() {
        let saver = AudioSaver::new();
        let audio = MonoBuffer::new(vec![0.5, 0.0, -0.5], 16000);
        let temp_path = "/tmp/test_save_mono.wav";
        
        let result = saver.save_mono(&audio, temp_path);
        assert!(result.is_ok());
        
        // Cleanup
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_save_stereo() {
        let saver = AudioSaver::new();
        let audio = StereoBuffer::new(
            vec![0.5, 0.0, -0.5],
            vec![0.3, 0.0, -0.3],
            16000,
        );
        let temp_path = "/tmp/test_save_stereo.wav";
        
        let result = saver.save_stereo(&audio, temp_path);
        assert!(result.is_ok());
        
        // Cleanup
        let _ = fs::remove_file(temp_path);
    }
}
