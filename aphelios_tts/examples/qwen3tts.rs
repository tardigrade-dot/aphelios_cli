use aphelios_core::{
    measure_time,
    utils::{common, logger},
};
use aphelios_tts::qwen_tts::qwen_tts_infer::{Language, Qwen3TTS, Speaker};

fn main() -> anyhow::Result<()> {
    logger::init_logging();
    let device = common::get_default_device(false)?;

    let model_path = "/Volumes/sw/pretrained_models/Qwen3-TTS-12Hz-0.6B-CustomVoice";
    let text_to_speech = "你好世界,这是第一条语音.";
    let text_to_speech2 = "今天天气真的很不错";
    let text_to_speech3 = "代表了当前语音合成与识别领域的前沿水平";

    let model = Qwen3TTS::from_pretrained(model_path, device)?;

    // Batch synthesis demonstration
    println!("\n--- 批量合成演示 ---");
    let batch_texts = vec![
        text_to_speech.to_string(),
        text_to_speech2.to_string(),
        text_to_speech3.to_string(),
    ];

    // Ensure output directory exists
    std::fs::create_dir_all("output")?;

    let batch_audios = measure_time!(
        "语音合成 (批量 3 条)",
        model.synthesize_batch(&batch_texts, Speaker::Ryan, Language::Chinese, None)?
    );

    for (i, audio) in batch_audios.into_iter().enumerate() {
        let filename = format!("output/batch_output_{}.wav", i);
        audio.save(&filename)?;
        println!("已保存第 {} 条音频到 {}", i + 1, filename);
    }

    Ok(())
}
