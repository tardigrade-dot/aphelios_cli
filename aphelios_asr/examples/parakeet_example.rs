use anyhow::Result;
use aphelios_core::{
    utils::core_utils::{self, PARAKEET_TDT_MODEL_PATH},
    AudioLoader,
};
use parakeet_rs::{ExecutionConfig, ExecutionProvider, ParakeetTDT, TimestampMode, Transcriber};
fn main() -> Result<()> {
    core_utils::init_tracing();
    // let execution_config =
    //     ExecutionConfig::default().with_execution_provider(ExecutionProvider::CoreML);
    let mut parakeet =
        ParakeetTDT::from_pretrained(PARAKEET_TDT_MODEL_PATH, None).expect("load model failed");

    let audio = AudioLoader::new()
        .load("/Volumes/sw/video/mQlxALUw3h4_16k.wav")?
        .to_vec_f32();
    let result = parakeet.transcribe_samples(audio, 16000, 1, Some(TimestampMode::Sentences))?;
    println!("{}", result.text);

    // Token-level timestamps
    for token in result.tokens {
        println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
    }

    Ok(())
}
