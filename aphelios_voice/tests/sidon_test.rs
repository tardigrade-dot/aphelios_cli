use std::{path::PathBuf, time::Instant};

use anyhow::Result;
use aphelios_core::init_logging;
use aphelios_voice::voiceclear::{preprocess, sidon::SidonPipeline, wav};
use tracing::info;

#[test]
fn process_test() -> Result<()> {
    init_logging();

    let i = "/Users/larry/coderesp/aphelios_cli/aphelios_voice/test_data/mQlxALUw3h4-12s-16k.wav";

    let i_path = &PathBuf::from(i);
    let stem = i_path.file_stem().unwrap().to_str().unwrap();
    let ext = i_path.extension().and_then(|s| s.to_str()).unwrap_or("wav");
    let o = i_path.with_file_name(format!("{stem}.enhanced.{ext}"));

    let (audio_48k, _sr) = wav::read_wav(&PathBuf::from(i))?;
    let audio_16k = preprocess::resample(&audio_48k, wav::SR, 16_000);
    let dur = audio_16k.len() as f32 / 16_000.0;
    info!("Input: {:.1}s, resampled to 16 kHz", dur);

    let mut pipeline = SidonPipeline::new(None::<String>)?;

    let t0 = Instant::now();
    let out_audio = pipeline.process(&audio_16k)?;

    let elapsed = t0.elapsed();
    let out_dur = out_audio.len() as f32 / 48_000.0;
    info!(
        "Enhanced[Only model infer] {:.1}s in {:.1}s ({:.0}x realtime)",
        out_dur, elapsed.as_secs_f64(),
        out_dur as f64 / elapsed.as_secs_f64()
    );

    wav::write_wav(&o, &out_audio, 48_000)?;

    info!("Saved to {}", o.display());
    Ok(())
}
