pub mod constants;
pub mod fft;
pub mod processor;

pub static MODEL_FILE: &str =
    "/Users/larry/coderesp/aphelios_cli/aphelios_core/onnx_models/demucs/htdemucs_embedded.onnx";

#[cfg(test)]
mod tests {
    use crate::demucs::MODEL_FILE;

    use super::processor::{DemucsProcessor, StereoAudio};
    use hound;

    #[test]
    fn test_demucs_separation() {
        let test_wav = "/Users/larry/coderesp/aphelios_cli/test_data/RYdrPg6xdYo.wav";
        // 读取测试音频文件
        let wav_reader = hound::WavReader::open(test_wav).expect("Failed to open WAV file");

        let mut left_channel = Vec::new();
        let mut right_channel = Vec::new();

        // 假设音频是立体声，采样率为44100Hz
        for sample in wav_reader.into_samples::<i16>() {
            match sample {
                Ok(sample) => {
                    // 将样本转换为f32格式
                    let normalized_sample = sample as f32 / i16::MAX as f32;

                    // 假设左右声道交替存储
                    if left_channel.len() == right_channel.len() {
                        left_channel.push(normalized_sample);
                    } else {
                        right_channel.push(normalized_sample);
                    }
                }
                Err(e) => panic!("Error reading sample: {}", e),
            }
        }

        // 如果只有一个声道，则复制到两个声道
        if left_channel.len() != right_channel.len() {
            right_channel = left_channel.clone();
        }

        // 确保声道长度一致
        if left_channel.len() > right_channel.len() {
            left_channel.truncate(right_channel.len());
        } else if right_channel.len() > left_channel.len() {
            right_channel.truncate(left_channel.len());
        }

        // 创建处理器实例，指定模型路径
        let mut processor =
            DemucsProcessor::new(MODEL_FILE.to_string()).expect("Failed to create DemucsProcessor");

        // 加载模型
        processor.load_model().expect("Failed to load model");

        // 执行音频分离
        let result = processor.separate(&left_channel, &right_channel);

        match result {
            Ok(separated_tracks) => {
                println!("Audio separation completed successfully!");

                // 验证输出轨道数量
                assert_eq!(separated_tracks.drums.left.len(), left_channel.len());
                assert_eq!(separated_tracks.bass.left.len(), left_channel.len());
                assert_eq!(separated_tracks.other.left.len(), left_channel.len());
                assert_eq!(separated_tracks.vocals.left.len(), left_channel.len());

                // 保存分离后的音轨到文件（可选）
                save_audio_track(&separated_tracks.drums, "drums_output.wav");
                save_audio_track(&separated_tracks.bass, "bass_output.wav");
                save_audio_track(&separated_tracks.other, "other_output.wav");
                save_audio_track(&separated_tracks.vocals, "vocals_output.wav");

                println!("Separated tracks saved to files.");
            }
            Err(e) => {
                eprintln!("Error during audio separation: {}", e);
                panic!("Audio separation failed: {}", e);
            }
        }
    }

    // 辅助函数：保存音频轨道到WAV文件
    fn save_audio_track(audio: &StereoAudio, filename: &str) {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer =
            hound::WavWriter::create(filename, spec).expect("Failed to create WAV writer");

        for i in 0..std::cmp::min(audio.left.len(), audio.right.len()) {
            let left_sample = (audio.left[i] * i16::MAX as f32) as i16;
            let right_sample = (audio.right[i] * i16::MAX as f32) as i16;

            writer
                .write_sample(left_sample)
                .expect("Failed to write left sample");
            writer
                .write_sample(right_sample)
                .expect("Failed to write right sample");
        }

        writer.finalize().expect("Failed to finalize WAV writer");
    }
}

#[test]
fn test_vocal_only_separation() {
    use crate::demucs::processor::{DemucsProcessor, StereoAudio};
    use hound;

    // 读取测试音频文件 - 使用环境变量或默认值
    let test_wav_path = std::env::var("DEMUX_TEST_WAV")
        .unwrap_or("/Users/larry/coderesp/aphelios_cli/test_data/RYdrPg6xdYo_16k.wav".to_string());

    let mut wav_reader = hound::WavReader::open(&test_wav_path).expect("Failed to open WAV file");
    let spec = wav_reader.spec();

    let mut samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => wav_reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .expect("Failed to read float samples"),
        hound::SampleFormat::Int => {
            match spec.bits_per_sample {
                16 => wav_reader
                    .samples::<i16>()
                    .map(|s| s.map(|s| s as f32 / i16::MAX as f32))
                    .collect::<Result<Vec<_>, _>>()
                    .expect("Failed to read 16-bit integer samples"),
                24 => {
                    // For 24-bit samples, we need to handle them differently
                    // Since hound doesn't directly support 24-bit, we'll treat as 32-bit
                    // and normalize appropriately
                    let mut reader =
                        hound::WavReader::open(&test_wav_path).expect("Failed to open WAV file");
                    reader
                        .samples::<i32>()
                        .map(|s| s.map(|s| s as f32 / 8388607.0)) // Max 24-bit value is 2^23 - 1 = 8388607
                        .collect::<Result<Vec<_>, _>>()
                        .expect("Failed to read 24-bit integer samples")
                }
                32 => wav_reader
                    .samples::<i32>()
                    .map(|s| s.map(|s| s as f32 / i32::MAX as f32))
                    .collect::<Result<Vec<_>, _>>()
                    .expect("Failed to read 32-bit integer samples"),
                _ => panic!("Unsupported bit depth: {}", spec.bits_per_sample),
            }
        }
    };

    // Convert to stereo regardless of original channel count
    let (left_channel, right_channel) = convert_to_stereo_from_vec(&samples, spec.channels);

    // 创建处理器实例，指定模型路径
    let mut processor =
        DemucsProcessor::new(MODEL_FILE.to_string()).expect("Failed to create DemucsProcessor");

    // 加载模型
    processor.load_model().expect("Failed to load model");

    // 执行音频分离
    let result = processor.separate(&left_channel, &right_channel);

    match result {
        Ok(separated_tracks) => {
            println!("Vocal-only separation completed successfully!");

            // 仅保留人声轨道
            let vocal_track = &separated_tracks.vocals;

            // 合并非人声轨道（drums, bass, other）用于重新配音
            let mut other_track = StereoAudio {
                left: vec![0.0; left_channel.len()],
                right: vec![0.0; left_channel.len()],
            };

            for i in 0..left_channel.len() {
                other_track.left[i] = separated_tracks.drums.left[i]
                    + separated_tracks.bass.left[i]
                    + separated_tracks.other.left[i];
                other_track.right[i] = separated_tracks.drums.right[i]
                    + separated_tracks.bass.right[i]
                    + separated_tracks.other.right[i];
            }

            // 保存人声和非人声轨道
            save_audio_track_for_respeech_with_spec(
                vocal_track,
                "vocal_for_respeech.wav",
                spec.sample_rate,
            );
            save_audio_track_for_respeech_with_spec(
                &other_track,
                "background_for_respeech.wav",
                spec.sample_rate,
            );

            println!("Vocal and background tracks saved for respeaking.");
        }
        Err(e) => {
            eprintln!("Error during vocal-only separation: {}", e);
            panic!("Vocal-only separation failed: {}", e);
        }
    }
}

// 辅助函数：将任意声道数的音频转换为立体声
fn convert_to_stereo_from_vec(samples: &Vec<f32>, channels: u16) -> (Vec<f32>, Vec<f32>) {
    match channels {
        1 => {
            // 单声道转立体声：复制到左右声道
            (samples.clone(), samples.clone())
        }
        2 => {
            // 立体声：直接分离左右声道
            let mut left_channel = Vec::new();
            let mut right_channel = Vec::new();

            for (i, &sample) in samples.iter().enumerate() {
                if i % 2 == 0 {
                    left_channel.push(sample);
                } else {
                    right_channel.push(sample);
                }
            }

            (left_channel, right_channel)
        }
        _ => {
            // 多声道：只使用前两个声道作为左右声道，或者混合所有声道
            let mut left_channel = Vec::new();
            let mut right_channel = Vec::new();

            for (i, &sample) in samples.iter().enumerate() {
                let channel_idx = i % channels as usize;

                if channel_idx == 0 {
                    left_channel.push(sample);
                } else if channel_idx == 1 {
                    right_channel.push(sample);
                }
            }

            // 如果左右声道长度不一致，调整为相同长度
            if left_channel.len() != right_channel.len() {
                let min_len = std::cmp::min(left_channel.len(), right_channel.len());
                left_channel.truncate(min_len);
                right_channel.truncate(min_len);
            }

            (left_channel, right_channel)
        }
    }
}

// 辅助函数：保存音频轨道到WAV文件（用于重新配音功能，带指定采样率）
fn save_audio_track_for_respeech_with_spec(
    audio: &crate::demucs::processor::StereoAudio,
    filename: &str,
    sample_rate: u32,
) {
    use hound;

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(filename, spec).expect("Failed to create WAV writer");

    for i in 0..std::cmp::min(audio.left.len(), audio.right.len()) {
        let left_sample = (audio.left[i] * i16::MAX as f32) as i16;
        let right_sample = (audio.right[i] * i16::MAX as f32) as i16;

        writer
            .write_sample(left_sample)
            .expect("Failed to write left sample");
        writer
            .write_sample(right_sample)
            .expect("Failed to write right sample");
    }

    writer.finalize().expect("Failed to finalize WAV writer");
}

// 辅助函数：保存音频轨道到WAV文件（用于重新配音功能，保持原有函数兼容性）
fn save_audio_track_for_respeech(audio: &crate::demucs::processor::StereoAudio, filename: &str) {
    save_audio_track_for_respeech_with_spec(audio, filename, 44100);
}
