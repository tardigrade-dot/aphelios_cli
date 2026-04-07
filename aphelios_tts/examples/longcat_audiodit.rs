use anyhow::{Context, Result};
use aphelios_tts::longcat_audiodit::{
    run_python_reference, GuidanceMethod, LongCatAudioDiT, LongCatInferenceConfig,
    LongCatPythonReference, LongCatSynthesisRequest,
};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Backend {
    Auto,
    Candle,
    Python,
}

///! long text candle 24s/114s 音频/执行
///! long text python 24s/41s 音频/执行
///! shot text candle 10s/95s 音频/执行
///! shot text python 10s/35s 音频/执行
const SHORT_TEXT: &str =
    "在二十世纪上半叶的中国乡村，有两个巨大的历史进程值得注意，它们使此一时期的中国有别于前一时代";
const LONG_TEXT: &str = "本书旨在探讨中国国家政权与乡村社会之间的互动关系，比如，旧的封建帝国的权力和法令是如何行之于乡村的，它们与地方组织和领袖是怎样的关系，国家权力的扩张是如何改造乡村旧有领导机构以建立新型领导层并推行新的政策的。";

#[derive(Debug, Parser)]
#[command(name = "longcat-audiodit")]
struct Args {
    #[arg(
        long,
        default_value = LONG_TEXT,
    )]
    text: String,
    #[arg(
        long = "prompt_text",
        alias = "prompt-text",
        default_value = "贝克莱的努力并未产生有形的结果"
    )]
    prompt_text: Option<String>,
    #[arg(
        long = "prompt_audio",
        alias = "prompt-audio",
        default_value = "/Volumes/sw/video/youyi-5s.wav"
    )]
    prompt_audio: Option<PathBuf>,
    #[arg(
        long = "output_audio",
        alias = "output-audio",
        default_value = "/Users/larry/coderesp/aphelios_cli/output/longcat-output-rust.wav"
    )]
    output_audio: PathBuf,
    #[arg(
        long = "model_dir",
        alias = "model-dir",
        default_value = "/Volumes/sw/pretrained_models/LongCat-AudioDiT-1B"
    )]
    model_dir: PathBuf,
    #[arg(long, default_value_t = 16)]
    nfe: usize,
    #[arg(
        long = "guidance_strength",
        alias = "guidance-strength",
        default_value_t = 4.0
    )]
    guidance_strength: f64,
    #[arg(
        long = "guidance_method",
        alias = "guidance-method",
        default_value = "cfg"
    )]
    guidance_method: String,
    #[arg(long, default_value_t = 1024)]
    seed: u64,
    #[arg(long)]
    duration: Option<usize>,
    #[arg(long)]
    cpu: bool,
    #[arg(
        long = "tokenizer_path",
        alias = "tokenizer-path",
        default_value = "/Volumes/sw/pretrained_models/umt5-base/tokenizer.json"
    )]
    tokenizer_path: Option<PathBuf>,
    #[arg(long, value_enum, default_value_t = Backend::Candle)]
    backend: Backend,
    #[arg(
        long = "python_bin",
        alias = "python-bin",
        default_value = "/Volumes/sw/conda_envs/lcataudio/bin/python"
    )]
    python_bin: PathBuf,
    #[arg(
        long = "python_script",
        alias = "python-script",
        default_value = "/Users/larry/coderesp/aphelios_cli/aphelios_tts/scripts/longcat_reference.py"
    )]
    python_script: PathBuf,
}

fn main() -> Result<()> {
    let start = std::time::Instant::now();
    let args = Args::parse();
    let guidance_method = GuidanceMethod::from_cli(&args.guidance_method)?;

    let model = LongCatAudioDiT::from_pretrained(
        &args.model_dir,
        LongCatInferenceConfig {
            cpu: args.cpu,
            tokenizer_path: args.tokenizer_path.clone(),
            dtype: None,
        },
    )?;

    let request = LongCatSynthesisRequest {
        text: args.text,
        prompt_text: args.prompt_text,
        prompt_audio: args.prompt_audio,
        duration: args.duration,
        steps: args.nfe,
        cfg_strength: args.guidance_strength,
        guidance_method,
        seed: args.seed,
    };

    match args.backend {
        Backend::Candle => {
            let waveform = model.synthesize(&request)?;
            // Write WAV file using hound
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 24000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut writer =
                hound::WavWriter::create(&args.output_audio, spec).with_context(|| {
                    format!(
                        "failed to create output WAV file: {}",
                        args.output_audio.display()
                    )
                })?;
            for sample in &waveform {
                writer.write_sample(*sample).with_context(|| {
                    format!("failed to write sample to {}", args.output_audio.display())
                })?;
            }
            writer.finalize().with_context(|| {
                format!(
                    "failed to finalize WAV file: {}",
                    args.output_audio.display()
                )
            })?;
            let duration_sec = waveform.len() as f64 / 24000.0;
            println!(
                "Saved: {} ({:.2}s, {} samples)",
                args.output_audio.display(),
                duration_sec,
                waveform.len()
            );
            println!("Time: {:.2}s", start.elapsed().as_secs_f64());
            Ok(())
        }
        Backend::Python => run_python_reference(
            &LongCatPythonReference::new(args.python_bin, args.python_script, args.model_dir),
            &request,
            &args.output_audio,
        ),
        Backend::Auto => match model.synthesize(&request) {
            Ok(waveform) => {
                let spec = hound::WavSpec {
                    channels: 1,
                    sample_rate: 24000,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                };
                let mut writer =
                    hound::WavWriter::create(&args.output_audio, spec).with_context(|| {
                        format!(
                            "failed to create output WAV file: {}",
                            args.output_audio.display()
                        )
                    })?;
                for sample in &waveform {
                    writer.write_sample(*sample).with_context(|| {
                        format!("failed to write sample to {}", args.output_audio.display())
                    })?;
                }
                writer.finalize().with_context(|| {
                    format!(
                        "failed to finalize WAV file: {}",
                        args.output_audio.display()
                    )
                })?;
                let duration_sec = waveform.len() as f64 / 24000.0;
                println!(
                    "Candle backend saved: {} ({:.2}s, {} samples)",
                    args.output_audio.display(),
                    duration_sec,
                    waveform.len()
                );
                Ok(())
            }
            Err(err) => {
                eprintln!("Candle path unavailable, falling back to python reference: {err}");
                run_python_reference(
                    &LongCatPythonReference::new(
                        args.python_bin,
                        args.python_script,
                        &model.paths.model_dir,
                    ),
                    &request,
                    &args.output_audio,
                )
                .with_context(|| {
                    format!(
                        "auto backend failed after candle fallback for {}",
                        args.output_audio.display()
                    )
                })
            }
        },
    }
}
