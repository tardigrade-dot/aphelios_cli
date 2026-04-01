use anyhow::Result;
use aphelios_asr::qwenasr::audio::{self};
use aphelios_asr::qwenasr::transcribe::Pipeline;
use aphelios_asr::qwenasr::{aligner::ForcedAligner, transcribe};
use aphelios_asr::silerovad::VadProcessor;
use aphelios_core::init_logging;
use aphelios_core::utils::base::get_device;
use candle_core::Tensor;
use std::path::{Path, PathBuf};
use tracing::info;

#[tokio::test]
async fn vad_and_qwenasr_test() -> Result<()> {
    init_logging();

    let audio_path = "/Volumes/sw/video/mQlxALUw3h4.wav";
    let language = "English";

    // let audio_path = "/Volumes/sw/video/zh-voice-example.wav";
    // let language = "Chinese";

    const ASR_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/Qwen3-ASR-0.6B";

    // 1. VAD process
    let mut vad = VadProcessor::new_default()?;
    let segments = vad.process_from_file(audio_path)?;
    info!("Detected {} speech segments", segments.len());

    assert!(
        !segments.is_empty(),
        "Should detect at least one speech segment"
    );
    let batches = vad.aggregate_segments(&segments, 30.0, 0.3);
    info!("Aggregated {} batches", batches.len());

    let device = get_device();

    info!(
        "Loading QwenASR model from {} on {:?}",
        ASR_MODEL_DIR, device
    );
    let mut pipeline = Pipeline::load_with_device(Path::new(ASR_MODEL_DIR), device.clone())?;

    // 3. Load entire audio as float samples
    let samples = audio::load_wav(Path::new(audio_path), &pipeline.audio_cfg)?;
    let sample_rate = pipeline.audio_cfg.sample_rate as f64;
    let padding_duration = 0.2; // 200ms padding

    // 4. Load Aligner before the loop
    const ALIGNER_MODEL_DIR: &str = "/Volumes/sw/pretrained_models/Qwen3-ForcedAligner-0.6B";
    info!(
        "Loading ForcedAligner model from {} on {:?}",
        ALIGNER_MODEL_DIR, device
    );
    let aligner = aphelios_asr::qwenasr::aligner::ForcedAligner::load_with_device(
        Path::new(ALIGNER_MODEL_DIR),
        device.clone(),
    )?;

    // 5. Iterate over VAD batches, transcribe & align sequentially
    let mut total_aligned_items = Vec::new();

    let mut final_text = String::new();
    for (i, batch) in batches.iter().enumerate() {
        // Include padding context
        let padded_start = f64::max(0.0, batch.start - padding_duration);
        let padded_end = batch.end + padding_duration;

        let start_sample = (padded_start * sample_rate) as usize;
        let end_sample = (padded_end * sample_rate) as usize;

        if start_sample < samples.len() {
            let end = std::cmp::min(end_sample, samples.len());
            let batch_pcm = &samples[start_sample..end];

            if !batch_pcm.is_empty() {
                let audio_ms = batch_pcm.len() as f64 / sample_rate * 1000.0;
                let (flat, n_frames) = audio::mel_spectrogram(batch_pcm, &pipeline.audio_cfg);
                let mel_bins = pipeline.audio_cfg.mel_bins;
                let mel = Tensor::from_vec(flat, (mel_bins, n_frames), &device)?;

                info!(
                    "  Batch {}: running chunk duration {:.2}s",
                    i + 1,
                    audio_ms / 1000.0
                );

                let (text, timings) = pipeline.transcribe_mel(&mel, audio_ms)?;
                info!(
                    "  -> Batch {} Result: RT={:.2}x, text: {}",
                    i + 1,
                    (timings.encode_ms + timings.decode_ms) / audio_ms,
                    text
                );

                let text_str = text.trim();
                if text_str.is_empty() {
                    continue;
                }

                final_text.push_str(text_str);

                info!("  Batch {} Running ForcedAligner...", i + 1);
                // Pass the padded audio samples and get locally relative timestamps
                let items = aligner.align_samples(batch_pcm, text_str, language)?;

                // Convert to absolute timestamps by adding the segment start time
                for mut item in items {
                    item.start_time += padded_start;
                    item.end_time += padded_start;
                    total_aligned_items.push(item);
                }
            }
        }
    }

    info!("Final text: {}", final_text);
    info!(
        "Total Aligned {} items globally:",
        total_aligned_items.len()
    );
    let mut items_info = Vec::new();
    for (_, item) in total_aligned_items.iter().enumerate() {
        items_info.push(format!(
            "[{:.3} - {:.3}] {}",
            item.start_time, item.end_time, item.text
        ));
    }
    info!("{}", items_info.join(""));

    Ok(())
}

#[test]
fn qwen3asr_simple_test() -> Result<()> {
    init_logging();
    let qwen3asr_model = PathBuf::from("/Volumes/sw/pretrained_models/Qwen3-ASR-0.6B");
    let aligner_model = PathBuf::from("/Volumes/sw/pretrained_models/Qwen3-ForcedAligner-0.6B");

    // let input = PathBuf::from("/Volumes/sw/video/mQlxALUw3h4.wav");
    // let language = "English";

    // no vad, for small audio file
    let input = PathBuf::from("/Volumes/sw/video/qinsheng.wav");
    let language = "Chinese";

    let device = get_device();
    let mut pipeline = transcribe::Pipeline::load_with_device(&qwen3asr_model, device.clone())
        .unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(1)
        });

    let text = pipeline.transcribe(&input).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1)
    });

    info!("{text}");

    if !aligner_model.exists() {
        eprintln!(
            "aligner directory not found: {}, skipping alignment",
            aligner_model.display()
        );
        return Ok(());
    }

    let aligner = ForcedAligner::load_with_device(&aligner_model, device).unwrap_or_else(|e| {
        eprintln!("error loading aligner: {e}");
        std::process::exit(1)
    });

    let items = aligner.align(&input, &text, language).unwrap_or_else(|e| {
        eprintln!("error during alignment: {e}");
        std::process::exit(1)
    });

    info!("\nAlignment results:");
    let mut items_info = Vec::new();
    for item in &items {
        items_info.push(format!(
            "[{:.3} - {:.3}] {}",
            item.start_time, item.end_time, item.text
        ));
    }
    info!("{}", items_info.join(""));
    Ok(())
}
