use anyhow::Result;
use std::path::PathBuf;
use std::fs;
use std::mem;

use aphelios_tts::supertonic3::helper::{
    load_text_to_speech, load_voice_style, timer, write_wav_file, sanitize_filename,
};

impl Args {

    fn default() -> Self{
        Args { use_gpu: false, onnx_dir: "/Volumes/sw/onnx_models/supertonic-3/onnx".to_string(), total_step: 8,
            speed: 1.05f32, n_test: 1, voice_style: vec!["/Volumes/sw/onnx_models/supertonic-3/voice_styles/F1.json".to_string()],
            text: vec!["This morning, I took a walk in the park, and the sound of the birds and the breeze was so pleasant that I stopped for a long time just to listen.".to_string()],
            lang: vec!["en".to_string()], save_dir: "/Users/larry/coderesp/aphelios_cli/output".to_string(),
            batch: false }
    }
}

struct Args {
    /// Use GPU for inference (default: CPU)
    use_gpu: bool,

    /// Path to ONNX model directory
    onnx_dir: String,

    /// Number of denoising steps
    total_step: usize,

    /// Speech speed factor (higher = faster)
    speed: f32,

    /// Number of times to generate
    n_test: usize,

    /// Voice style file path(s)
    voice_style: Vec<String>,

    /// Text(s) to synthesize
    text: Vec<String>,

    /// Language(s) for synthesis; see the main README for all supported codes
    lang: Vec<String>,

    /// Output directory
    save_dir: String,

    /// Enable batch mode (multiple text-style pairs)
    batch: bool,
}
#[test]
fn main_test() -> Result<()> {
    println!("=== TTS Inference with ONNX Runtime (Rust) ===\n");

    // --- 1. Parse arguments --- //
    let args = Args::default();
    let total_step = args.total_step;
    let speed = args.speed;
    let n_test = args.n_test;
    let voice_style_paths = &args.voice_style;
    let text_list = &args.text;
    let lang_list = &args.lang;
    let save_dir = &args.save_dir;
    let batch = args.batch;

    if batch {
        if voice_style_paths.len() != text_list.len() {
            anyhow::bail!(
                "Number of voice styles ({}) must match number of texts ({})",
                voice_style_paths.len(),
                text_list.len()
            );
        }
        if lang_list.len() != text_list.len() {
            anyhow::bail!(
                "Number of languages ({}) must match number of texts ({})",
                lang_list.len(),
                text_list.len()
            );
        }
    }

    let bsz = voice_style_paths.len();

    // --- 2. Load TTS components --- //
    let mut text_to_speech = load_text_to_speech(&args.onnx_dir, args.use_gpu)?;

    // --- 3. Load voice styles --- //
    let style = load_voice_style(voice_style_paths, true)?;

    // --- 4. Synthesize speech --- //
    fs::create_dir_all(save_dir)?;

    for n in 0..n_test {
        println!("\n[{}/{}] Starting synthesis...", n + 1, n_test);

        let (wav, duration) = if batch {
            timer("Generating speech from text", || {
                text_to_speech.batch(text_list, lang_list, &style, total_step, speed)
            })?
        } else {
            let (w, d) = timer("Generating speech from text", || {
                text_to_speech.call(&text_list[0], &lang_list[0], &style, total_step, speed, 0.3)
            })?;
            (w, vec![d])
        };

        // Save outputs
        for i in 0..bsz {
            let fname = format!("{}_{}.wav", sanitize_filename(&text_list[i], 20), n + 1);
            let wav_slice = if batch {
                let wav_len = wav.len() / bsz;
                let actual_len = (text_to_speech.sample_rate as f32 * duration[i]) as usize;
                let wav_start = i * wav_len;
                let wav_end = wav_start + actual_len.min(wav_len);
                &wav[wav_start..wav_end]
            } else {
                // For non-batch mode, wav is a single concatenated audio
                let actual_len = (text_to_speech.sample_rate as f32 * duration[0]) as usize;
                &wav[..actual_len.min(wav.len())]
            };

            let output_path = PathBuf::from(save_dir).join(&fname);
            write_wav_file(&output_path, wav_slice, text_to_speech.sample_rate)?;
            println!("Saved: {}", output_path.display());
        }
    }

    println!("\n=== Synthesis completed successfully! ===");

    // Prevent ONNX Runtime sessions from being dropped, which causes mutex cleanup issues
    mem::forget(text_to_speech);

    Ok(())
}
