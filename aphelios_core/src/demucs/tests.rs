#[cfg(test)]
mod tests {
    use super::*;
    use crate::demucs::processor::{DemucsProcessor, StereoAudio};
    use hound;
    use std::fs::File;
    use std::io::BufReader;

    #[test]
    fn test_demucs_separation() {
        // 读取测试音频文件
        let wav_reader = hound::WavReader::open("assets/youyi-15s.wav").expect("Failed to open WAV file");
        
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
        
        // 创建处理器实例
        let mut processor = DemucsProcessor::new(Some("assets/htdemucs_embedded.onnx".to_string()))
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