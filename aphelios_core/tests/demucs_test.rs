use aphelios_core::{
    demucs::{MODEL_FILE, processor::DemucsProcessor},
    utils::core_utils::{
        combine_stereo_tracks, load_and_resample_audio, save_audio_track_with_spec,
    },
};

const AUDIO_16K: &str = "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4_16k.wav";
const AUDIO_44K: &str = "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4.wav";
const TARGET_SAMPLE_RATE: u32 = 44100;

/// 运行 Demucs 音频分离测试
fn run_demucs_test(test_wav: &str, output_suffix: &str) -> Result<(), String> {
    // 使用一站式加载函数 (处理了位深转换、声道对齐、重采样)
    let audio_data = load_and_resample_audio(test_wav, TARGET_SAMPLE_RATE)
        .map_err(|e| format!("音频预处理失败：{}", e))?;

    // audio_data 结构为 [Left_Channel, Right_Channel]
    let left_channel = audio_data[0].clone();
    let right_channel = audio_data[1].clone();

    println!(
        "音频加载完成，采样率：{}Hz, 长度：{} 采样点",
        TARGET_SAMPLE_RATE,
        left_channel.len()
    );

    // 创建处理器实例，指定模型路径
    let mut processor =
        DemucsProcessor::new(MODEL_FILE.to_string()).expect("Failed to create DemucsProcessor");

    // 加载模型
    processor
        .load_model()
        .map_err(|e| format!("模型加载失败：{}", e))?;

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

            // 保存分离后的音轨到文件
            let base_name = test_wav.trim_end_matches(".wav");
            save_audio_track_with_spec(
                &separated_tracks.drums,
                &format!("{}_drums{}.wav", base_name, output_suffix),
                TARGET_SAMPLE_RATE,
            );
            save_audio_track_with_spec(
                &separated_tracks.bass,
                &format!("{}_bass{}.wav", base_name, output_suffix),
                TARGET_SAMPLE_RATE,
            );
            save_audio_track_with_spec(
                &separated_tracks.other,
                &format!("{}_other{}.wav", base_name, output_suffix),
                TARGET_SAMPLE_RATE,
            );
            save_audio_track_with_spec(
                &separated_tracks.vocals,
                &format!("{}_vocals{}.wav", base_name, output_suffix),
                TARGET_SAMPLE_RATE,
            );

            println!("Separated tracks saved to files.");
            Ok(())
        }
        Err(e) => {
            eprintln!("Error during audio separation: {}", e);
            Err(format!("Audio separation failed: {}", e))
        }
    }
}

/// 运行人声分离测试
fn run_vocal_separation_test(test_wav: &str, output_suffix: &str) -> Result<(), String> {
    // 使用一站式加载函数 (处理了位深转换、声道对齐、重采样)
    let mut audio_data = load_and_resample_audio(test_wav, TARGET_SAMPLE_RATE)
        .map_err(|e| format!("音频预处理失败：{}", e))?;

    if audio_data.len() == 1 {
        // 单声道 -> 双声道，拷贝一份
        audio_data.push(audio_data[0].clone());
    }
    // audio_data 结构为 [Left_Channel, Right_Channel]
    let left_channel = audio_data[0].clone();
    let right_channel: Vec<f32> = audio_data[1].clone();

    println!(
        "音频加载完成，采样率：{}Hz, 长度：{} 采样点",
        TARGET_SAMPLE_RATE,
        left_channel.len()
    );

    // 初始化 Demucs 并推理
    let mut processor =
        DemucsProcessor::new(MODEL_FILE.to_string()).expect("Failed to create DemucsProcessor");
    processor
        .load_model()
        .map_err(|e| format!("模型加载失败：{}", e))?;

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
            let base_name = test_wav.trim_end_matches(".wav");
            save_audio_track_with_spec(
                vocal_track,
                &format!("{}_vocals_only{}.wav", base_name, output_suffix),
                TARGET_SAMPLE_RATE,
            );
            save_audio_track_with_spec(
                &instrumental_track,
                &format!("{}_instrumental_only{}.wav", base_name, output_suffix),
                TARGET_SAMPLE_RATE,
            );

            println!("Vocal and instrumental tracks saved to files.");
            Ok(())
        }
        Err(e) => {
            eprintln!("Error during vocal separation: {}", e);
            Err(format!("Vocal separation failed: {}", e))
        }
    }
}

#[test]
fn test_demucs_separation_16k_to_44k() {
    // 测试 16kHz 音频重采样到 44.1kHz 后进行分离
    println!("=== Demucs Test with 16kHz audio (resampled to 44.1kHz) ===");
    run_demucs_test(AUDIO_16K, "_16k_to_44k").expect("Demucs separation failed");
}

#[test]
fn test_demucs_separation_44k() {
    // 测试原生 44.1kHz 音频分离
    println!("=== Demucs Test with 44.1kHz audio ===");
    run_demucs_test(AUDIO_44K, "_44k").expect("Demucs separation failed");
}

#[test]
fn test_vocal_separation_16k_to_44k() {
    // 测试 16kHz 音频重采样到 44.1kHz 后进行人声分离
    println!("=== Vocal Separation Test with 16kHz audio (resampled to 44.1kHz) ===");
    run_vocal_separation_test(AUDIO_16K, "_16k_to_44k").expect("Vocal separation failed");
}

#[test]
fn test_vocal_separation_44k() {
    // 测试原生 44.1kHz 音频人声分离
    println!("=== Vocal Separation Test with 44.1kHz audio ===");
    run_vocal_separation_test(AUDIO_44K, "_44k").expect("Vocal separation failed");
}
