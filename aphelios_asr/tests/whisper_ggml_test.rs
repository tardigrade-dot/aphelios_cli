// use anyhow::Result;
// use aphelios_core::utils::core_utils;
// use tracing::info;
// use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

// const AUDIO_SHORT: &str = "/Volumes/sw/video/mQlxALUw3h4.wav";
// const AUDIO_LONG: &str = "/Volumes/sw/video/RYdrPg6xdYo.wav";
// const AUDIO_STALIN: &str = "/Volumes/sw/video/Why Does Joseph Stalin Matter？.wav";

// #[test]
// fn asr_test_short() -> Result<()> {
//     core_utils::init_tracing();
//     let model_path = "/Volumes/sw/ggml_models/ggml-large-v3-turbo.bin";
//     let coreml_model_path = "/Volumes/sw/ggml_models/ggml-large-v3-turbo-encoder.mlmodelc";

//     // load a context and model
//     let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
//         .expect("failed to load model");

//     // create a params object
//     let params = FullParams::new(SamplingStrategy::BeamSearch {
//         beam_size: 5,
//         patience: -1.0,
//     });

//     // assume we have a buffer of audio data
//     // here we'll make a fake one, floating point samples, 32 bit, 16KHz, mono
//     let (_, audio_data) = core_utils::normalize_audio(AUDIO_STALIN, 16000)?;

//     // now we can run the model
//     let mut state = ctx.create_state().expect("failed to create state");
//     state
//         .full(params, &audio_data[..])
//         .expect("failed to run model");

//     info!("segment length : {}", state.n_len());
//     // fetch the results
//     for segment in state.as_iter() {
//         info!(
//             "[{} - {}]: {}",
//             // note start and end timestamps are in centiseconds
//             // (10s of milliseconds)
//             segment.start_timestamp(),
//             segment.end_timestamp(),
//             // the Display impl for WhisperSegment will replace invalid UTF-8 with the Unicode replacement character
//             segment
//         );
//     }
//     Ok(())
// }

// /// AUDIO_SHORT 15s
// /// AUDIO_LONG 400s
// #[test]
// fn asr_test_long() -> Result<()> {
//     core_utils::init_tracing();

//     let model_path = "/Volumes/sw/ggml_models/ggml-large-v3-turbo.bin";
//     let coreml_model_path = Some("/Volumes/sw/ggml_models/ggml-large-v3-turbo-encoder.mlmodelc");

//     // load a context and model, enable Core ML encoder
//     let mut ctx_params = WhisperContextParameters::default();
//     ctx_params.use_gpu(true);
//     // ctx_params.coreml_encoder_path(coreml_model_path);

//     let ctx =
//         WhisperContext::new_with_params(model_path, ctx_params).expect("failed to load model");

//     let params = FullParams::new(SamplingStrategy::BeamSearch {
//         beam_size: 5,
//         patience: -1.0,
//     });

//     let (_, audio_data) = core_utils::normalize_audio(AUDIO_LONG, 16000)?;

//     let mut state = ctx.create_state().expect("failed to create state");
//     state
//         .full(params, &audio_data[..])
//         .expect("failed to run model");

//     info!("segment length : {}", state.n_len());
//     for segment in state.as_iter() {
//         info!(
//             "[{} - {}]: {}",
//             segment.start_timestamp(),
//             segment.end_timestamp(),
//             segment
//         );
//     }

//     Ok(())
// }
