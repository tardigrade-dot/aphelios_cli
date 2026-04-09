use anyhow::{Context, Result};
use aphelios_core::utils::{base, logger};
use aphelios_tts::audio::AudioBuffer;
use aphelios_tts::models::talker::Language;
use aphelios_tts::qwen_tts::qwen_tts::{device_info, Qwen3TTS, SynthesisOptions};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;

const MODEL_PATH: &str = "/Volumes/sw/pretrained_models/Qwen3-TTS-12Hz-1.7B-Base";
// const REF_AUDIO: &str = "/Users/larry/Documents/resources/qinsheng-4s-isolated.wav";
const REF_AUDIO: &str = "/Volumes/sw/video/youyi-5s.wav";
// const REF_TEXT: &str = "写这本书的目的在于通过我的走访和观察";
const REF_TEXT: &str = "贝克莱的努力并未产生有形的结果";

fn output_paths(base_path: &Path, count: usize) -> Vec<PathBuf> {
    let stemmed = base_path.to_string_lossy();
    (0..count)
        .map(|index| {
            if index == 0 {
                base_path.to_path_buf()
            } else {
                PathBuf::from(format!(
                    "{}_{}.wav",
                    stemmed.trim_end_matches(".wav"),
                    index
                ))
            }
        })
        .collect()
}

///! 1.7B
///! load model time: 7.89s
///! generate audio time: 75.52s
///! total time: 83.93s
#[test]
fn qwen_tts_single_test() -> Result<()> {
    logger::init_logging();

    for required_path in [MODEL_PATH, REF_AUDIO] {
        assert!(
            Path::new(required_path).exists(),
            "required test asset is missing: {required_path}"
        );
    }

    let text = "本书旨在探讨中国国家政权与乡村社会之间的互动关系，比如，旧的封建帝国的权力和法令是如何行之于乡村的，它们与地方组织和领袖是怎样的关系，国家权力的扩张是如何改造乡村旧有领导机构以建立新型领导层并推行新的政策的。";

    let device = base::get_default_device(false)?;
    info!(
        "running qwen3tts batch inference on {}",
        device_info(&device)
    );

    #[cfg(feature = "profiling")]
    let start = std::time::Instant::now();
    let model = Qwen3TTS::from_pretrained(MODEL_PATH, device)?;
    #[cfg(feature = "profiling")]
    println!("load model time: {:.2}s", start.elapsed().as_secs_f64());

    #[cfg(feature = "profiling")]
    let start2 = std::time::Instant::now();
    let ref_audio = AudioBuffer::load(REF_AUDIO)?;
    let prompt = model.create_voice_clone_prompt(&ref_audio, Some(REF_TEXT))?;

    let options = SynthesisOptions {
        seed: Some(42),
        max_length: 320,
        ..Default::default()
    };

    let audios =
        model.synthesize_voice_clone(text, &prompt, Language::Chinese, Some(options), None)?;

    #[cfg(feature = "profiling")]
    println!(
        "generate audio time: {:.2}s",
        start2.elapsed().as_secs_f64()
    );

    audios.save("/Users/larry/coderesp/aphelios_cli/output/qwen_tts_single_test-0.6B.wav")?;
    Ok(())
}

#[test]
fn qwen_tts_batch_test() -> Result<()> {
    logger::init_logging();

    for required_path in [MODEL_PATH, REF_AUDIO] {
        assert!(
            Path::new(required_path).exists(),
            "required test asset is missing: {required_path}"
        );
    }

    let texts = vec![
        "为了行文简便，以相同的概率处于社会中的任何位置也被称为等概率假定，从而前述所有根据此假设的决策模型都被称为道德价值判断的等概率模型。",
        "本书中许多篇论文都出自经济学家之手，这点毫不令人意外，因为他们看起来比其他领域的社会科学家更多地应用了功利主义方法。",
        "我想讨论的是人文科学中一个明显的并行进程，类似于人类行为学习者的实践往往无效。",
    ];

    let device = base::get_default_device(false)?;
    info!(
        "running qwen3tts batch inference on {}",
        device_info(&device)
    );

    let model = Qwen3TTS::from_pretrained(MODEL_PATH, device)?;
    let ref_audio = AudioBuffer::load(REF_AUDIO)?;
    let prompt = model.create_voice_clone_prompt(&ref_audio, Some(REF_TEXT))?;

    let options = SynthesisOptions {
        seed: Some(42),
        max_length: 320,
        ..Default::default()
    };

    let batch_inputs: Vec<String> = texts.iter().map(|text| (*text).to_string()).collect();
    let audios = model.synthesize_voice_clone_batch(
        &batch_inputs,
        &prompt,
        Language::Chinese,
        Some(options),
    )?;

    assert_eq!(
        audios.len(),
        texts.len(),
        "batch inference should produce one audio buffer per input text"
    );

    for (index, audio) in audios.iter().enumerate() {
        assert!(
            !audio.is_empty(),
            "batch item {index} returned an empty audio buffer"
        );
        assert!(
            audio.duration() > 0.1,
            "batch item {index} is unexpectedly short: {:.3}s",
            audio.duration()
        );
        assert_eq!(
            audio.sample_rate, 24_000,
            "batch item {index} should keep the model sample rate"
        );
    }

    let output_dir = PathBuf::from("/Users/larry/coderesp/aphelios_cli/output");
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create output dir {}", output_dir.display()))?;
    let output_path = output_dir.join("batch_test-0.6B.wav");
    let written_paths = output_paths(&output_path, texts.len());

    for path in &written_paths {
        if path.exists() {
            fs::remove_file(path)
                .with_context(|| format!("failed to remove stale output {}", path.display()))?;
        }
    }

    let output_prefix = output_path
        .to_str()
        .context("output path contains invalid UTF-8")?;
    for (index, audio) in audios.iter().enumerate() {
        let path = if index == 0 {
            PathBuf::from(output_prefix)
        } else {
            PathBuf::from(format!(
                "{}_{}.wav",
                output_prefix.trim_end_matches(".wav"),
                index
            ))
        };
        audio
            .save(&path)
            .with_context(|| format!("failed to save batch output {}", path.display()))?;
    }

    for (index, path) in written_paths.iter().enumerate() {
        assert!(
            path.exists(),
            "expected batch output file for item {index}: {}",
            path.display()
        );
        let metadata = fs::metadata(path)
            .with_context(|| format!("failed to stat output {}", path.display()))?;
        assert!(
            metadata.len() > 44,
            "output wav for item {index} looks empty: {}",
            path.display()
        );
    }

    Ok(())
}
