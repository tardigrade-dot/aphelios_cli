// use std::time::Instant;

// use aphelios_core::utils::logger;
// use tracing::info;
// use vibevoice::{Device, VibeVoice};

// fn main() -> Result<(), vibevoice::VibeVoiceError> {
//     logger::init_logging();
//     // Create with default 1.5B model
//     // let mut vv = VibeVoice::new(ModelVariant::Batch1_5B, Device::Metal)?;

//     let mut vv = VibeVoice::builder()
//         .model_path("/Volumes/sw/pretrained_models/VibeVoice-1.5B")
//         .device(Device::Metal)
//         .build()?;
//     // Synthesize speech
//     let ref_audio = "/Users/larry/coderesp/aphelios_cli/test_data/newvoice.wav";
//     let text_to_speech = "近来研究表明，那种庞大、复杂、联合式的宗族在中国并不普遍，可能只存在于华南及江南的某些地区。";
//     let start = Instant::now();
//     let audio = vv.synthesize(text_to_speech, Some(ref_audio))?;

//     audio.save_wav("output/VibeVoice-output.wav")?;

//     info!("合成耗时: {}", start.elapsed().as_secs_f64());
//     Ok(())
// }
