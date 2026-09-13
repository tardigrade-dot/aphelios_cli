use anyhow::Result;
use aphelios_asr::{
    audio::{MonoBuffer, ResampleQuality, Resampler},
    pipeline::{AsrEngineChoice, AsrPipeline, VadModel},
    srt::{format_srt_time, generate_srt, generate_vad_srt},
    AsrSegment, DecodingResult, SubSegment, VadSegment,
};

#[test]
fn test_srt_time_formatting() {
    assert_eq!(format_srt_time(0.0), "00:00:00,000");
    assert_eq!(format_srt_time(61.5), "00:01:01,500");
    assert_eq!(format_srt_time(3661.123), "01:01:01,123");
}

#[test]
fn test_resampler_mono_buffer() -> Result<()> {
    let resampler = Resampler::new().with_quality(ResampleQuality::Fast);
    let mono = MonoBuffer::new(vec![0.0; 16000], 16000);
    let resampled = resampler.resample_mono(&mono, 8000)?;
    assert_eq!(resampled.sample_rate, 8000);
    assert_eq!(resampled.len(), 8000);
    Ok(())
}

#[tokio::test]
async fn test_generate_srt_file() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let srt_file = temp_dir.path().join("test_out.srt");
    let srt_path_str = srt_file.to_str().unwrap();

    let segments = vec![AsrSegment {
        start: 0.0,
        duration: 2.0,
        dr: DecodingResult {
            tokens: vec![],
            text: "Hello world".to_string(),
            avg_logprob: -0.5,
            no_speech_prob: 0.01,
            temperature: 0.0,
            compression_ratio: 1.0,
        },
        sub_segments: vec![SubSegment {
            start: 0.0,
            end: 2.0,
            text: "Hello world".to_string(),
        }],
    }];

    generate_srt(&segments, srt_path_str).await?;
    assert!(srt_file.exists());
    let content = std::fs::read_to_string(&srt_file)?;
    assert!(content.contains("Hello world"));
    assert!(content.contains("00:00:00,000 --> 00:00:02,000"));

    Ok(())
}

#[tokio::test]
async fn test_generate_vad_srt_file() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let srt_file = temp_dir.path().join("test_vad.srt");
    let srt_path_str = srt_file.to_str().unwrap();

    let vad_segments = vec![
        VadSegment::new(0, 1000, 0.9),
        VadSegment::new(1500, 3500, 0.95),
    ];

    generate_vad_srt(&vad_segments, srt_path_str).await?;
    assert!(srt_file.exists());
    let content = std::fs::read_to_string(&srt_file)?;
    assert!(content.contains("00:00:00,000 --> 00:00:01,000"));
    assert!(content.contains("00:00:01,500 --> 00:00:03,500"));

    Ok(())
}

#[test]
fn test_asr_pipeline_builder() {
    let pipeline = AsrPipeline::new("test.mp4")
        .with_vad(VadModel::SileroVad("/path/to/vad".to_string()))
        .with_asr(AsrEngineChoice::QwenRsCrate {
            model_dir: "/path/to/model".to_string(),
        })
        .with_language("English")
        .with_context("Test news")
        .with_output_srt("out.srt");

    assert_eq!(pipeline.input_path, "test.mp4");
    assert_eq!(pipeline.language, "English");
    assert_eq!(pipeline.output_srt_path.as_deref(), Some("out.srt"));
    assert_eq!(pipeline.context.as_deref(), Some("Test news"));
}
