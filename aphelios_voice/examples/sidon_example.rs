use anyhow::Result;
use aphelios_core::init_logging;
use aphelios_voice::voiceclear::sidon::SidonPipeline;
use aphelios_voice::voiceclear::{preprocess, wav};
use clap::Parser;
use serde::Serialize;
use tracing::info;
use std::path::PathBuf;
use std::time::Instant;

// cargo run -p aphelios_voice --features metal,profiling --example sidon_example -- -i /Users/larry/coderesp/aphelios_cli/aphelios_voice/test_data/mQlxALUw3h4-12s-16k.wav -m /Volumes/sw/pretrained_models/voice-clear
// cargo run -p aphelios_voice --example sidon_example -- -i /Users/larry/coderesp/aphelios_cli/aphelios_voice/test_data/mQlxALUw3h4-12s-16k.wav -m /Volumes/sw/pretrained_models/voice-clear
#[derive(Parser, Debug, Serialize)]
#[command(name = "voice-clear", about = "Speech enhancement via ONNX Runtime")]
struct Cli {
    /// Input WAV file
    #[arg(short, long, help_heading = "Required")]
    input: PathBuf,

    /// Output WAV file [default: <input>.<task>.enhanced.wav]
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Mastering preset (clear tasks only): applePodcasts, spotify, youtube, broadcast, bypass
    #[arg(long, default_value = "applePodcasts")]
    mastering: String,

    #[arg(short, long)]
    model: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging();

    // let json_str = serde_json::to_string_pretty(&cli).unwrap();
    info!("当前运行参数配置：\n{:#?}", cli);
    // println!("当前运行参数配置 (JSON)：\n{}", json_str);

    let output = cli.output.unwrap_or_else(|| {
        let stem = cli.input.file_stem().unwrap().to_str().unwrap();
        let ext = cli.input.extension().and_then(|s| s.to_str()).unwrap_or("wav");
        cli.input.with_file_name(format!("{stem}.enhanced.{ext}"))
    });

    info!("Enhancing [{}]", cli.input.display());

    // ── Sidon pipeline ──────────────────────────────────────────
    let (audio_48k, _sr) = wav::read_wav(&cli.input)?;
    let audio_16k = preprocess::resample(&audio_48k, wav::SR, 16_000);
    let dur = audio_16k.len() as f32 / 16_000.0;
    info!("Input: {:.1}s, resampled to 16 kHz", dur);

    let mut pipeline = SidonPipeline::new(cli.model)?;

    let t0 = Instant::now();
    let out_audio = pipeline.process(&audio_16k)?;

    let elapsed = t0.elapsed();
    let out_dur = out_audio.len() as f32 / 48_000.0;
    info!(
        "Enhanced[Only model infer] {:.1}s in {:.1}s ({:.0}x realtime)",
        out_dur, elapsed.as_secs_f64(),
        out_dur as f64 / elapsed.as_secs_f64()
    );

    wav::write_wav(&output, &out_audio, 48_000)?;

    info!("Saved to {}", output.display());
    Ok(())
}
