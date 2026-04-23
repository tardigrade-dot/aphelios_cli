use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Array3, ArrayD, Axis, Ix3, IxD};
use ort::{InferenceSession, SessionOptions, GraphOptimizationLevel, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use rand::{SeedableRng, rngs::StdRng};
use serde_json;
use crate::moss_tts_nano::config::{Manifest, TtsMeta, CodecMeta};
use crate::moss_tts_nano::sampling;
use tokenizers::Tokenizer;

pub struct MossTtsRuntime {
    pub model_dir: PathBuf,
    pub manifest: Manifest,
    pub tts_meta: TtsMeta,
    pub codec_meta: CodecMeta,
    pub sessions: HashMap<String, InferenceSession>,
    pub tokenizer: Tokenizer,
    pub rng: StdRng,
    pub codec_streaming_state: Option<CodecStreamingState>,
}

pub struct CodecStreamingState {
    pub feeds: HashMap<String, Value>,
}

impl MossTtsRuntime {
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let manifest_path = model_dir.join("browser_poc_manifest.json");
        let manifest_content = std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("Failed to read manifest at {:?}", manifest_path))?;
        let manifest: Manifest = serde_json::from_str(&manifest_content)?;

        let tts_meta_path = model_dir.join(&manifest.model_files.tts_meta);
        let tts_meta_content = std::fs::read_to_string(&tts_meta_path)?;
        let tts_meta: TtsMeta = serde_json::from_str(&tts_meta_content)?;

        let codec_meta_path = model_dir.join(&manifest.model_files.codec_meta);
        let codec_meta_content = std::fs::read_to_string(&codec_meta_path)?;
        let codec_meta: CodecMeta = serde_json::from_str(&codec_meta_content)?;

        let mut sessions = HashMap::new();
        let session_options = SessionOptions::new()?;
        session_options.set_graph_optimization_level(GraphOptimizationLevel::Level3)?;
        session_options.set_intra_op_num_threads(4)?;

        let tts_dir = tts_meta_path.parent().unwrap();
        let codec_dir = codec_meta_path.parent().unwrap();

        sessions.insert("prefill".to_string(), InferenceSession::builder()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(tts_dir.join(&tts_meta.files.prefill))?);

        sessions.insert("decode".to_string(), InferenceSession::builder()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(tts_dir.join(&tts_meta.files.decode_step))?);

        sessions.insert("local_decoder".to_string(), InferenceSession::builder()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(tts_dir.join(&tts_meta.files.local_decoder))?);

        if let Some(ref path) = tts_meta.files.local_greedy_frame {
            sessions.insert("local_greedy_frame".to_string(), InferenceSession::builder()
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .commit_from_file(tts_dir.join(path))?);
        }

        if let Some(ref path) = tts_meta.files.local_fixed_sampled_frame {
            sessions.insert("local_fixed_sampled_frame".to_string(), InferenceSession::builder()
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .commit_from_file(tts_dir.join(path))?);
        }

        if let Some(ref path) = tts_meta.files.local_cached_step {
            sessions.insert("local_cached_step".to_string(), InferenceSession::builder()
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .commit_from_file(tts_dir.join(path))?);
        }

        sessions.insert("codec_encode".to_string(), InferenceSession::builder()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(codec_dir.join(&codec_meta.files.encode))?);

        sessions.insert("codec_decode".to_string(), InferenceSession::builder()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(codec_dir.join(&codec_meta.files.decode_full))?);

        sessions.insert("codec_decode_step".to_string(), InferenceSession::builder()
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(codec_dir.join(&codec_meta.files.decode_step))?);

        let tokenizer_path = model_dir.join(&manifest.model_files.tokenizer_model);
        // SentencePiece models in tokenizers are usually loaded from tokenizer.json
        // which contains the SP model, or we need to build it.
        // If tokenizer.model is a raw SentencePiece model, we might need a specific loader.
        // Assuming we can load it if it's in a compatible format or we use from_file.
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer at {:?}: {}", tokenizer_path, e))?;

        let rng = StdRng::seed_from_u64(1234);

        let mut runtime = Self {
            model_dir,
            manifest,
            tts_meta,
            codec_meta,
            sessions,
            tokenizer,
            rng,
            codec_streaming_state: None,
        };
        runtime.reset_codec_streaming_session()?;
        Ok(runtime)
    }

    pub fn reset_codec_streaming_session(&mut self) -> Result<()> {
        let mut feeds = HashMap::new();
        if let Some(ref streaming_decode) = self.codec_meta.streaming_decode {
            for spec in &streaming_decode.transformer_offsets {
                let shape: Vec<usize> = spec.shape.iter().map(|&s| s as usize).collect();
                let arr = ArrayD::<i32>::zeros(shape);
                feeds.insert(spec.input_name.clone(), Value::from_array(arr)?);
            }
            for spec in &streaming_decode.attention_caches {
                let offset_shape: Vec<usize> = spec.offset_shape.iter().map(|&s| s as usize).collect();
                feeds.insert(spec.offset_input_name.clone(), Value::from_array(ArrayD::<i32>::zeros(offset_shape))?);

                let cache_shape: Vec<usize> = spec.cache_shape.iter().map(|&s| s as usize).collect();
                feeds.insert(spec.cached_keys_input_name.clone(), Value::from_array(ArrayD::<f32>::zeros(cache_shape.clone()))?);
                feeds.insert(spec.cached_values_input_name.clone(), Value::from_array(ArrayD::<f32>::zeros(cache_shape))?);

                let positions_shape: Vec<usize> = spec.positions_shape.iter().map(|&s| s as usize).collect();
                feeds.insert(spec.cached_positions_input_name.clone(), Value::from_array(ArrayD::<i32>::from_elem(positions_shape, -1))?);
            }
        }
        self.codec_streaming_state = Some(CodecStreamingState { feeds });
        Ok(())
    }

    pub fn build_voice_clone_request_rows(&self, prompt_audio_codes: &[Vec<i32>], text_token_ids: &[i32]) -> Array3<i32> {
        let n_vq = self.manifest.tts_config.n_vq as usize;
        let pad_id = self.manifest.tts_config.audio_pad_token_id;
        let start_id = self.manifest.tts_config.audio_start_token_id;
        let end_id = self.manifest.tts_config.audio_end_token_id;
        let user_slot_id = self.manifest.tts_config.audio_user_slot_token_id;

        let mut rows = Vec::new();

        // Prefix templates
        for &tid in &self.manifest.prompt_templates.user_prompt_prefix_token_ids {
            let mut row = vec![pad_id; n_vq + 1];
            row[0] = tid;
            rows.push(row);
        }

        {
            let mut row = vec![pad_id; n_vq + 1];
            row[0] = start_id;
            rows.push(row);
        }

        // Reference audio codes
        for code_row in prompt_audio_codes {
            let mut row = vec![pad_id; n_vq + 1];
            row[0] = user_slot_id;
            for i in 0..n_vq.min(code_row.len()) {
                row[i + 1] = code_row[i];
            }
            rows.push(row);
        }

        // Suffix templates
        {
            let mut row = vec![pad_id; n_vq + 1];
            row[0] = end_id;
            rows.push(row);
        }

        for &tid in &self.manifest.prompt_templates.user_prompt_after_reference_token_ids {
            let mut row = vec![pad_id; n_vq + 1];
            row[0] = tid;
            rows.push(row);
        }

        for &tid in text_token_ids {
            let mut row = vec![pad_id; n_vq + 1];
            row[0] = tid;
            rows.push(row);
        }

        for &tid in &self.manifest.prompt_templates.assistant_prompt_prefix_token_ids {
            let mut row = vec![pad_id; n_vq + 1];
            row[0] = tid;
            rows.push(row);
        }

        {
            let mut row = vec![pad_id; n_vq + 1];
            row[0] = start_id;
            rows.push(row);
        }

        let num_rows = rows.len();
        let mut arr = Array3::zeros((1, num_rows, n_vq + 1));
        for (i, row) in rows.into_iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                arr[[0, i, j]] = val;
            }
        }
        arr
    }

    pub fn generate_audio_frames<F>(
        &mut self,
        prompt_audio_codes: &[Vec<i32>],
        text_token_ids: &[i32],
        mut on_frame: Option<F>,
    ) -> Result<Vec<Vec<i32>>>
    where
        F: FnMut(&[i32], usize),
    {
        let input_ids = self.build_voice_clone_request_rows(prompt_audio_codes, text_token_ids);
        let seq_len = input_ids.dim().1;
        let attention_mask = Array2::<i32>::ones((1, seq_len));

        let prefill_session = self.sessions.get("prefill")
            .ok_or_else(|| anyhow::anyhow!("Prefill session not found"))?;
        let outputs = prefill_session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
        ]?)?;

        let mut global_hidden = outputs["global_hidden"].try_extract_tensor::<f32>()?
            .slice(ort::s![0, -1, ..])
            .to_owned()
            .insert_axis(Axis(0));

        let mut past_valid_length = seq_len as i32;
        let mut past_kv = HashMap::new();
        for name in &self.tts_meta.onnx.prefill_output_names {
            if name != "global_hidden" {
                let past_name = name.replace("present_", "past_");
                past_kv.insert(past_name, outputs[name.as_str()].try_extract_tensor::<f32>()?.to_owned());
            }
        }

        let mut generated_frames = Vec::new();
        let max_new_frames = self.manifest.generation_defaults.max_new_frames;
        let n_vq = self.manifest.tts_config.n_vq as usize;
        let assistant_slot_id = self.manifest.tts_config.audio_assistant_slot_token_id;

        let mut previous_tokens_by_channel = vec![Vec::new(); n_vq];

        for _ in 0..max_new_frames {
            // Local decoder step to get next tokens
            let local_decoder = self.sessions.get("local_decoder")
                .ok_or_else(|| anyhow::anyhow!("Local decoder session not found"))?;
            let mut frame = vec![0; n_vq];

            // 1. Sample assistant text token
            let outputs = local_decoder.run(ort::inputs![
                "global_hidden" => global_hidden.view(),
                "text_token_id" => Array1::from_vec(vec![0i32]),
                "audio_prefix_token_ids" => Array2::<i32>::from_elem((1, n_vq - 1), self.manifest.tts_config.audio_pad_token_id),
            ]?)?;

            let text_logits = outputs["text_logits"].try_extract_tensor::<f32>()?;
            let text_logits_1d = text_logits.view().into_shape(text_logits.len())?.to_owned();

            let next_text_token = sampling::sample_assistant_text_token(
                &text_logits_1d,
                &self.manifest,
                self.manifest.generation_defaults.do_sample,
                self.manifest.generation_defaults.text_temperature,
                self.manifest.generation_defaults.text_top_k,
                self.manifest.generation_defaults.text_top_p,
                &mut self.rng,
            );

            if next_text_token != assistant_slot_id {
                break;
            }

            // 2. Sample audio tokens channel by channel
            let mut current_prefix = Vec::new();
            for channel in 0..n_vq {
                let mut audio_prefix = Array2::<i32>::from_elem((1, n_vq - 1), self.manifest.tts_config.audio_pad_token_id);
                for (i, &v) in current_prefix.iter().enumerate() {
                    audio_prefix[[0, i]] = v;
                }

                let outputs = local_decoder.run(ort::inputs![
                    "global_hidden" => global_hidden.view(),
                    "text_token_id" => Array1::from_vec(vec![next_text_token]),
                    "audio_prefix_token_ids" => audio_prefix,
                ]?)?;

                let audio_logits = outputs["audio_logits"].try_extract_tensor::<f32>()?;
                // Extract logits for current channel
                let channel_logits = audio_logits.slice(ort::s![0, channel, ..]).to_owned().into_shape(audio_logits.dim().2)?;

                let sampled_audio_token = sampling::sample_audio_token(
                    &channel_logits,
                    &previous_tokens_by_channel[channel],
                    self.manifest.generation_defaults.do_sample,
                    self.manifest.generation_defaults.audio_temperature,
                    self.manifest.generation_defaults.audio_top_k,
                    self.manifest.generation_defaults.audio_top_p,
                    self.manifest.generation_defaults.audio_repetition_penalty,
                    &mut self.rng,
                );
                frame[channel] = sampled_audio_token;
                current_prefix.push(sampled_audio_token);
                previous_tokens_by_channel[channel].push(sampled_audio_token);
            }
            generated_frames.push(frame.clone());
            if let Some(ref mut callback) = on_frame {
                callback(&frame, generated_frames.len() - 1);
            }

            // 3. Update global hidden with decode step
            let mut decode_input_ids = Array3::from_elem((1, 1, n_vq + 1), self.manifest.tts_config.audio_pad_token_id);
            decode_input_ids[[0, 0, 0]] = assistant_slot_id;
            for i in 0..n_vq {
                decode_input_ids[[0, 0, i + 1]] = frame[i];
            }

            let decode_session = self.sessions.get("decode")
                .ok_or_else(|| anyhow::anyhow!("Decode session not found"))?;
            let mut inputs = HashMap::new();
            inputs.insert("input_ids".to_string(), Value::from_array(decode_input_ids)?);
            inputs.insert("past_valid_lengths".to_string(), Value::from_array(Array1::from_vec(vec![past_valid_length]))?);
            for (name, val) in &past_kv {
                inputs.insert(name.clone(), Value::from_array(val.view())?);
            }

            let outputs = decode_session.run(inputs)?;
            global_hidden = outputs["global_hidden"].try_extract_tensor::<f32>()?
                .slice(ort::s![0, -1, ..])
                .to_owned()
                .insert_axis(Axis(0));

            past_valid_length += 1;
            for name in &self.tts_meta.onnx.decode_output_names {
                if name != "global_hidden" {
                    let past_name = name.replace("present_", "past_");
                    past_kv.insert(past_name, outputs[name.as_str()].try_extract_tensor::<f32>()?.to_owned());
                }
            }
        }

        Ok(generated_frames)
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<i32>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {}", e))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i32).collect())
    }

    pub fn encode_reference_audio<P: AsRef<Path>>(&self, path: P) -> Result<Vec<Vec<i32>>> {
        // This requires loading audio, resampling to codec sample rate, and running codec_encode
        // For now, let's assume we use crate::audio::io::load_wav
        use crate::audio::io::load_wav;
        let audio = load_wav(path)?;

        // MOSS-TTS-Nano codec expects specific sample rate (usually 24000 or 16000)
        let target_sr = self.codec_meta.codec_config.sample_rate as u32;
        let samples = if audio.sample_rate != target_sr {
            use rubato::{Resampler, FastFixedIn, InterpolationType, InterpolationParameters};
            let params = InterpolationParameters {
                sinc_len: 256,
                over_sampling_factor: 128,
                interpolation: InterpolationType::Linear,
                weighting: rubato::WindowFunction::BlackmanHarris2,
            };
            let mut resampler = FastFixedIn::<f32>::new(
                target_sr as f64 / audio.sample_rate as f64,
                1.0,
                params,
                audio.samples.len(),
                1,
            )?;
            let resampled = resampler.process(&[&audio.samples], None)?;
            resampled[0].clone()
        } else {
            audio.samples
        };

        let num_samples = samples.len();
        let mut audio_tensor = Array3::<f32>::zeros((1, 1, num_samples));
        for (i, &s) in samples.iter().enumerate() {
            audio_tensor[[0, 0, i]] = s;
        }

        let codec_encode = self.sessions.get("codec_encode")
            .ok_or_else(|| anyhow::anyhow!("Codec encode session not found"))?;
        let outputs = codec_encode.run(ort::inputs![
            "waveform" => audio_tensor,
            "input_lengths" => Array1::from_vec(vec![num_samples as i32]),
        ]?)?;

        let audio_codes = outputs["audio_codes"].try_extract_tensor::<i32>()?;
        let audio_code_lengths = outputs["audio_code_lengths"].try_extract_tensor::<i32>()?;
        let code_len = audio_code_lengths[[0]] as usize;
        let num_quantizers = self.codec_meta.codec_config.num_quantizers as usize;

        let mut prompt_audio_codes = Vec::with_capacity(code_len);
        for i in 0..code_len {
            let mut frame = Vec::with_capacity(num_quantizers);
            for q in 0..num_quantizers {
                frame.push(audio_codes[[0, i, q]]);
            }
            prompt_audio_codes.push(frame);
        }

        Ok(prompt_audio_codes)
    }

    pub fn decode_audio_step(&mut self, frame: &[i32]) -> Result<Vec<f32>> {
        let n_vq = self.manifest.tts_config.n_vq as usize;
        let mut audio_codes = Array3::<i32>::zeros((1, 1, n_vq));
        for (j, &val) in frame.iter().enumerate() {
            audio_codes[[0, 0, j]] = val;
        }

        let codec_decode_step = self.sessions.get("codec_decode_step")
            .ok_or_else(|| anyhow::anyhow!("Codec decode step session not found"))?;

        let mut inputs = HashMap::new();
        inputs.insert("audio_codes".to_string(), Value::from_array(audio_codes)?);
        inputs.insert("audio_code_lengths".to_string(), Value::from_array(Array1::from_vec(vec![1i32]))?);

        if let Some(ref state) = self.codec_streaming_state {
            for (k, v) in &state.feeds {
                // We need to clone the Value or at least its underlying data.
                // In 'ort', Value is often just a wrapper.
                // However, we need to be careful about ownership.
                // For now, let's try to extract and re-wrap if necessary, or use Arc.
                // Actually, 'Value' from 'ort' 2.0-rc.11 is Arc-like.
                inputs.insert(k.clone(), v.try_clone()?);
            }
        }

        let outputs = codec_decode_step.run(inputs)?;

        // Update state
        if let Some(ref mut state) = self.codec_streaming_state {
            if let Some(ref streaming_decode) = self.codec_meta.streaming_decode {
                for spec in &streaming_decode.transformer_offsets {
                    state.feeds.insert(spec.input_name.clone(), outputs[spec.output_name.as_str()].try_clone()?);
                }
                for spec in &streaming_decode.attention_caches {
                    state.feeds.insert(spec.offset_input_name.clone(), outputs[spec.offset_output_name.as_str()].try_clone()?);
                    state.feeds.insert(spec.cached_keys_input_name.clone(), outputs[spec.cached_keys_output_name.as_str()].try_clone()?);
                    state.feeds.insert(spec.cached_values_input_name.clone(), outputs[spec.cached_values_output_name.as_str()].try_clone()?);
                    state.feeds.insert(spec.cached_positions_input_name.clone(), outputs[spec.cached_positions_output_name.as_str()].try_clone()?);
                }
            }
        }

        let audio = outputs["audio"].try_extract_tensor::<f32>()?;
        let audio_lengths = outputs["audio_lengths"].try_extract_tensor::<i32>()?;
        let length = audio_lengths[[0]] as usize;

        let channels = audio.dim().1;
        let mut mono_audio = vec![0.0; length];
        for i in 0..length {
            let mut sum = 0.0;
            for c in 0..channels {
                sum += audio[[0, c, i]];
            }
            mono_audio[i] = sum / channels as f32;
        }

        Ok(mono_audio)
    }

    pub fn decode_audio(&self, generated_frames: &[Vec<i32>]) -> Result<Vec<f32>> {
        if generated_frames.is_empty() {
            return Ok(Vec::new());
        }
        let n_vq = self.manifest.tts_config.n_vq as usize;
        let num_frames = generated_frames.len();
        let mut audio_codes = Array3::<i32>::zeros((1, num_frames, n_vq));
        for (i, frame) in generated_frames.iter().enumerate() {
            for (j, &val) in frame.iter().enumerate() {
                audio_codes[[0, i, j]] = val;
            }
        }

        let codec_decode = self.sessions.get("codec_decode")
            .ok_or_else(|| anyhow::anyhow!("Codec decode session not found"))?;
        let outputs = codec_decode.run(ort::inputs![
            "audio_codes" => audio_codes,
            "audio_code_lengths" => Array1::from_vec(vec![num_frames as i32]),
        ]?)?;

        let audio = outputs["audio"].try_extract_tensor::<f32>()?;
        let audio_lengths = outputs["audio_lengths"].try_extract_tensor::<i32>()?;
        let length = audio_lengths[[0]] as usize;

        // Assume mono or average channels
        let channels = audio.dim().1;
        let mut mono_audio = vec![0.0; length];
        for i in 0..length {
            let mut sum = 0.0;
            for c in 0..channels {
                sum += audio[[0, c, i]];
            }
            mono_audio[i] = sum / channels as f32;
        }

        Ok(mono_audio)
    }
}
