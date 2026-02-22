//! 测试工具模块

use aphelios_core::audio::{AudioLoader, Resampler, ResampleQuality};
use aphelios_core::utils::init_logging;

/// 测试音频文件路径
pub const TEST_AUDIO_16K: &str = "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4_16k.wav";
pub const TEST_AUDIO_44K: &str = "/Users/larry/coderesp/aphelios_cli/test_data/mQlxALUw3h4.wav";

/// 初始化测试环境
pub fn setup() {
    init_logging();
}

/// 加载测试音频文件（自动重采样到目标采样率）
pub fn load_test_audio(path: &str, target_sample_rate: u32) -> aphelios_core::audio::StereoBuffer {
    let audio = AudioLoader::new()
        .load(path)
        .expect("Failed to load audio file");
    
    let stereo = audio.to_stereo();
    
    if stereo.sample_rate != target_sample_rate {
        let resampler = Resampler::new().with_quality(ResampleQuality::Fast);
        resampler.resample_stereo(&stereo, target_sample_rate)
            .expect("Failed to resample audio")
    } else {
        stereo
    }
}

/// 测试配置
#[derive(Debug, Clone)]
pub struct TestConfig {
    pub model_dir: String,
    pub output_dir: String,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            model_dir: "/Volumes/sw/pretrained_models".to_string(),
            output_dir: "./test_output".to_string(),
        }
    }
}

impl TestConfig {
    pub fn whisper_model(&self) -> String {
        format!("{}/distil-large-v3.5", self.model_dir)
    }

    pub fn vad_model(&self) -> String {
        format!("{}/silero-vad/onnx/model.onnx", self.model_dir)
    }

    pub fn demucs_model(&self) -> String {
        "/Users/larry/coderesp/aphelios_cli/aphelios_core/onnx_models/demucs/htdemucs_embedded.onnx".to_string()
    }

    pub fn dia_model(&self) -> String {
        "/Users/larry/coderesp/aphelios_cli/aphelios_core/onnx_models/dia/diar_streaming_sortformer_4spk-v2.onnx".to_string()
    }
}
