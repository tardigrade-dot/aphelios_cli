use aphelios_core::demucs::processor::{DemucsProcessor, StereoAudio};
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

#[test]
fn test_vocal_separation() {
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
            println!("Vocal separation completed successfully!");

            // 合并人声轨道
            let vocal_track = &separated_tracks.vocals;
            
            // 合并非人声轨道（drums, bass, other）
            let instrumental_track = combine_stereo_tracks(&[
                &separated_tracks.drums,
                &separated_tracks.bass,
                &separated_tracks.other
            ]);

            // 保存人声和非人声轨道
            save_audio_track(vocal_track, "vocals_only.wav");
            save_audio_track(&instrumental_track, "instrumental_only.wav");

            println!("Vocal and instrumental tracks saved to files.");
        }
        Err(e) => {
            eprintln!("Error during vocal separation: {}", e);
            panic!("Vocal separation failed: {}", e);
        }
    }
}

// 辅助函数：合并多个立体声音轨
fn combine_stereo_tracks(tracks: &[&StereoAudio]) -> StereoAudio {
    if tracks.is_empty() {
        return StereoAudio {
            left: vec![0.0; 0],
            right: vec![0.0; 0],
        };
    }

    let sample_count = tracks[0].left.len();
    let mut combined_left = vec![0.0; sample_count];
    let mut combined_right = vec![0.0; sample_count];

    for track in tracks {
        for i in 0..std::cmp::min(sample_count, track.left.len()) {
            combined_left[i] += track.left[i];
            combined_right[i] += track.right[i];
        }
    }

    StereoAudio {
        left: combined_left,
        right: combined_right,
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