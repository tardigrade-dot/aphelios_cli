use anyhow::Result;
use aphelios_core::{
    utils::core_utils::{self, PARAKEET_TDT_MODEL_PATH},
    AudioLoader,
};
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};

const AUDIO_SHORT: &str = "/Volumes/sw/video/mQlxALUw3h4.wav";
const AUDIO_LONG: &str = "/Volumes/sw/video/RYdrPg6xdYo.wav";
const AUDIO_LONG_2: &str = "/Volumes/sw/video/Why Does Joseph Stalin Matter？.wav";

#[test]
fn test_parakeet() -> Result<()> {
    core_utils::init_tracing();
    // let execution_config =
    //     ExecutionConfig::default().with_execution_provider(ExecutionProvider::CoreML);
    let mut parakeet =
        ParakeetTDT::from_pretrained(PARAKEET_TDT_MODEL_PATH, None).expect("load model failed");

    let audio_list = vec![AUDIO_LONG_2];

    for audio_item in audio_list {
        let audio = AudioLoader::new().load(audio_item)?.to_vec_f32();
        let result = parakeet.transcribe_samples(audio, 16000, 1, Some(TimestampMode::Words))?;
        println!("{}", result.text);

        // Token-level timestamps
        for token in result.tokens {
            println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
        }
    }

    Ok(())
}
