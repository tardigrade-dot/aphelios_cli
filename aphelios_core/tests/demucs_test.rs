use aphelios_core::{
    common::core_utils::{
        combine_stereo_tracks, get_append_filename, load_and_resample_audio,
        save_audio_track_with_spec,
    },
    demucs::{MODEL_FILE, processor::DemucsProcessor},
};

#[test]
fn test_demucs_separation() {
    // 使用一站式加载函数 (处理了位深转换、声道对齐、重采样)
    let test_wav = "/Users/larry/coderesp/aphelios_cli/test_data/RYdrPg6xdYo.wav";
    let target_sample_rate = 44100;
    let audio_data = load_and_resample_audio(test_wav, target_sample_rate).expect("音频预处理失败");

    // audio_data 结构为 [Left_Channel, Right_Channel]
    let left_channel = audio_data[0].clone();
    let right_channel = audio_data[1].clone();

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
            save_audio_track_with_spec(
                &separated_tracks.drums,
                "output/drums_output.wav",
                target_sample_rate,
            );
            save_audio_track_with_spec(
                &separated_tracks.bass,
                "output/bass_output.wav",
                target_sample_rate,
            );
            save_audio_track_with_spec(
                &separated_tracks.other,
                "output/other_output.wav",
                target_sample_rate,
            );
            save_audio_track_with_spec(
                &separated_tracks.vocals,
                "output/vocals_output.wav",
                target_sample_rate,
            );

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
    let test_wav = "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4_16k.wav";
    // 使用一站式加载函数 (处理了位深转换、声道对齐、重采样)
    let target_sample_rate = 44100;
    let mut audio_data =
        load_and_resample_audio(test_wav, target_sample_rate).expect("音频预处理失败");

    if audio_data.len() == 1 {
        // 单声道 -> 双声道，拷贝一份
        audio_data.push(audio_data[0].clone());
    }
    // audio_data 结构为 [Left_Channel, Right_Channel]
    let left_channel = audio_data[0].clone();
    let right_channel: Vec<f32> = audio_data[1].clone();

    println!(
        "音频加载完成，采样率：{}Hz, 长度：{} 采样点",
        target_sample_rate,
        left_channel.len()
    );

    // 2. 初始化 Demucs 并推理
    let mut processor =
        DemucsProcessor::new(MODEL_FILE.to_string()).expect("Failed to create DemucsProcessor");
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
                &separated_tracks.other,
            ]);

            // 保存人声和非人声轨道
            let vacals = get_append_filename(test_wav, "_vocals_only");
            save_audio_track_with_spec(vocal_track, vacals.as_str(), target_sample_rate);

            let instrumental = get_append_filename(test_wav, "_instrumental_only");
            save_audio_track_with_spec(
                &instrumental_track,
                instrumental.as_str(),
                target_sample_rate,
            );

            println!("Vocal and instrumental tracks saved to files.");
        }
        Err(e) => {
            eprintln!("Error during vocal separation: {}", e);
            panic!("Vocal separation failed: {}", e);
        }
    }
}
