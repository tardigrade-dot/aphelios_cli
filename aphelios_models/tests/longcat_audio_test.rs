use anyhow::{Context, Result};
use aphelios_tts::longcat_audiodit::{
    GuidanceMethod, LongCatAudioDiT, LongCatInferenceConfig, LongCatSynthesisRequest,
};

///! long text candle 24s/114s 音频/执行
///! long text python 24s/41s 音频/执行
///! shot text candle 10s/95s 音频/执行
///! shot text python 10s/35s 音频/执行
const SHORT_TEXT: &str =
    "在二十世纪上半叶的中国乡村，有两个巨大的历史进程值得注意，它们使此一时期的中国有别于前一时代";
const LONG_TEXT: &str = "本书旨在探讨中国国家政权与乡村社会之间的互动关系，比如，旧的封建帝国的权力和法令是如何行之于乡村的，它们与地方组织和领袖是怎样的关系，国家权力的扩张是如何改造乡村旧有领导机构以建立新型领导层并推行新的政策的。";

///! load model time: 40.50s
///! generate audio time: 92.21s
///! total time: 133.47s
#[test]
fn sinmple_test() -> Result<()> {
    let model_dir = "/Volumes/sw/pretrained_models/LongCat-AudioDiT-1B";
    let output_audio = "/Users/larry/coderesp/aphelios_cli/output/longcat-output-rust-youyi.wav";

    let prompt_audio = "/Volumes/sw/video/youyi-5s.wav";
    let prompt_text = "贝克莱的努力并未产生有形的结果";
    let text = "本书旨在探讨中国国家政权与乡村社会之间的互动关系，比如，旧的封建帝国的权力和法令是如何行之于乡村的，它们与地方组织和领袖是怎样的关系，国家权力的扩张是如何改造乡村旧有领导机构以建立新型领导层并推行新的政策的。";

    let nfe = 16;
    let guidance_strength = 4.0;
    let seed = 1024;
    let cpu = false;
    let tokenizer_path = "/Volumes/sw/pretrained_models/umt5-base/tokenizer.json";

    let guidance_method = GuidanceMethod::from_cli("cfg")?;

    #[cfg(feature = "profiling")]
    let start = std::time::Instant::now();
    let model = LongCatAudioDiT::from_pretrained(
        &model_dir,
        LongCatInferenceConfig {
            cpu: cpu,
            tokenizer_path: Some(tokenizer_path.into()),
            dtype: None,
        },
    )?;
    #[cfg(feature = "profiling")]
    println!("load model time: {:.2}s", start.elapsed().as_secs_f64());

    #[cfg(feature = "profiling")]
    let start2 = std::time::Instant::now();
    let request = LongCatSynthesisRequest {
        text: text.to_string(),
        prompt_text: Some(prompt_text.to_string()),
        prompt_audio: Some(prompt_audio.into()),
        duration: None,
        steps: nfe,
        cfg_strength: guidance_strength,
        guidance_method,
        seed: seed,
    };

    let waveform = model.synthesize(&request)?;
    // Write WAV file using hound
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&output_audio, spec)
        .with_context(|| format!("failed to create output WAV file: {}", output_audio))?;
    for sample in &waveform {
        writer
            .write_sample(*sample)
            .with_context(|| format!("failed to write sample to {}", output_audio))?;
    }
    writer
        .finalize()
        .with_context(|| format!("failed to finalize WAV file: {}", output_audio))?;
    let duration_sec = waveform.len() as f64 / 24000.0;
    println!(
        "Saved: {} ({:.2}s, {} samples)",
        output_audio,
        duration_sec,
        waveform.len()
    );
    #[cfg(feature = "profiling")]
    println!(
        "generate audio time: {:.2}s",
        start2.elapsed().as_secs_f64()
    );
    Ok(())
}
