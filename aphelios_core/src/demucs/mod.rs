pub mod constants;
pub mod fft;
pub mod processor;

#[cfg(test)]
mod tests {
    use super::processor::{DemucsProcessor, StereoAudio};
    use hound;

    #[test]
    fn test_demucs_separation() {
        // 读取测试音频文件
        let wav_reader = hound::WavReader::open("/Users/larry/coderesp/aphelios_cli/aphelios_core/assets/youyi-15s.wav")
            .expect("Failed to open WAV file");

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
        let mut processor = DemucsProcessor::new(Some("/Users/larry/coderesp/aphelios_cli/aphelios_core/onnx_models/htdemucs_embedded.onnx".to_string()))
            .expect("Failed to create DemucsProcessor");

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

        let mut writer = hound::WavWriter::create(filename, spec).expect("Failed to create WAV writer");

        for i in 0..std::cmp::min(audio.left.len(), audio.right.len()) {
            let left_sample = (audio.left[i] * i16::MAX as f32) as i16;
            let right_sample = (audio.right[i] * i16::MAX as f32) as i16;

            writer.write_sample(left_sample).expect("Failed to write left sample");
            writer.write_sample(right_sample).expect("Failed to write right sample");
        }

        writer.finalize().expect("Failed to finalize WAV writer");
    }
}

#[test]
fn test_vocal_only_separation() {
    use crate::demucs::processor::{DemucsProcessor, StereoAudio};
    use hound;

    // 读取测试音频文件
    let wav_reader = hound::WavReader::open("/Users/larry/coderesp/aphelios_cli/aphelios_core/assets/youyi-15s.wav")
        .expect("Failed to open WAV file");

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
    let mut processor = DemucsProcessor::new(Some("/Users/larry/coderesp/aphelios_cli/aphelios_core/onnx_models/htdemucs_embedded.onnx".to_string()))
        .expect("Failed to create DemucsProcessor");

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
                other_track.left[i] = separated_tracks.drums.left[i] + 
                                     separated_tracks.bass.left[i] + 
                                     separated_tracks.other.left[i];
                other_track.right[i] = separated_tracks.drums.right[i] + 
                                      separated_tracks.bass.right[i] + 
                                      separated_tracks.other.right[i];
            }

            // 保存人声和非人声轨道
            save_audio_track_for_respeech(vocal_track, "vocal_for_respeech.wav");
            save_audio_track_for_respeech(&other_track, "background_for_respeech.wav");

            println!("Vocal and background tracks saved for respeaking.");
        }
        Err(e) => {
            eprintln!("Error during vocal-only separation: {}", e);
            panic!("Vocal-only separation failed: {}", e);
        }
    }
}

// 辅助函数：保存音频轨道到WAV文件（用于重新配音功能）
fn save_audio_track_for_respeech(audio: &crate::demucs::processor::StereoAudio, filename: &str) {
    use hound;

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(filename, spec).expect("Failed to create WAV writer");

    for i in 0..std::cmp::min(audio.left.len(), audio.right.len()) {
        let left_sample = (audio.left[i] * i16::MAX as f32) as i16;
        let right_sample = (audio.right[i] * i16::MAX as f32) as i16;

        writer.write_sample(left_sample).expect("Failed to write left sample");
        writer.write_sample(right_sample).expect("Failed to write right sample");
    }

    writer.finalize().expect("Failed to finalize WAV writer");
}