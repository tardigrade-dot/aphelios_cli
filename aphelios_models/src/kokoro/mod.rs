use anyhow::{Context, Result};
use ndarray::{Array, Array1, Array2, ArrayView2};
use ort::session::Session;
use ort::value::Value;
use std::path::Path;

pub mod voice;
pub mod tokenizer;
pub mod phonemize;

pub use voice::KokoroVoices;
pub use tokenizer::KokoroTokenizer;
pub use phonemize::KokoroPhonemizer;

/// Supported languages for Kokoro TTS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KokoroLanguage {
    /// American English
    EnUs,
    /// British English
    EnGb,
    /// Chinese (Planned for v1.1-zh)
    Zh,
}

/// Kokoro-82M-v1.0-ONNX Inference Session.
pub struct KokoroModel {
    session: Session,
    tokenizer: KokoroTokenizer,
    phonemizer: KokoroPhonemizer,
}

impl KokoroModel {
    /// Load model and tokenizer from the specified directory.
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        
        // Load ONNX model
        // We look for model.onnx in the model_dir/onnx/ or directly in model_dir
        let mut model_path = model_dir.join("onnx").join("model.onnx");
        if !model_path.exists() {
            model_path = model_dir.join("model.onnx");
        }
        
        let session = Session::builder()?
            .commit_from_file(model_path.clone())
            .with_context(|| format!("Failed to load ONNX model from: {:?}", model_path))?;
            
        // Load Tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = KokoroTokenizer::from_file(tokenizer_path)?;
            
        Ok(Self {
            session,
            tokenizer,
            phonemizer: KokoroPhonemizer::new(),
        })
    }

    /// Generate audio for a given raw text and language.
    /// Input: 
    /// - `text`: Raw text (e.g., "Hello world").
    /// - `lang`: The language/dialect to use.
    /// - `style`: Style vector view of shape (1, 256).
    /// - `speed`: Speaking speed (default 1.0).
    pub fn generate(
        &mut self,
        text: &str,
        lang: KokoroLanguage,
        style: ArrayView2<f32>,
        speed: f32,
    ) -> Result<Array1<f32>> {
        // 1. Phomenize raw text to IPA
        let ipa_text = self.phonemizer.phonemize(text, lang)?;
        
        // 2. Delegate to lower-level generator
        self.generate_from_ipa(&ipa_text, style, speed)
    }

    /// Lower-level method if you already have the IPA phoneme string.
    /// Input: 
    /// - `ipa_text`: IPA phonemes string.
    /// - `style`: Style vector view of shape (1, 256).
    /// - `speed`: Speaking speed (default 1.0).
    pub fn generate_from_ipa(
        &mut self,
        ipa_text: &str,
        style: ArrayView2<f32>,
        speed: f32,
    ) -> Result<Array1<f32>> {
        // 1. Phomenize (clean IPA string)
        let cleaned_ipa = self.phonemizer.clean_ipa(ipa_text)?;
        
        // 2. Tokenize (adds 0 at start and end)
        let token_ids = self.tokenizer.encode(&cleaned_ipa)?;
        let n = token_ids.len();
        
        if n > 512 {
            anyhow::bail!("Input too long: {} tokens (max 512)", n);
        }
        
        // 3. Prepare inputs
        let input_ids_array = Array2::from_shape_vec(
            (1, n),
            token_ids.into_iter().map(|id| id as i64).collect()
        )?;
        
        let style_owned = style.to_owned();
        let speed_array = Array::from_elem((1,), speed);

        let input_ids_value = Value::from_array(input_ids_array)
            .map_err(|e| anyhow::anyhow!("Failed to create input_ids tensor: {}", e))?;
        let style_value = Value::from_array(style_owned)
            .map_err(|e| anyhow::anyhow!("Failed to create style tensor: {}", e))?;
        let speed_value = Value::from_array(speed_array)
            .map_err(|e| anyhow::anyhow!("Failed to create speed tensor: {}", e))?;

        // 4. Run session
        let inputs = ort::inputs![
            "input_ids" => input_ids_value,
            "style" => style_value,
            "speed" => speed_value,
        ];

        let outputs = self.session.run(inputs)
            .map_err(|e| anyhow::anyhow!("Failed to run ONNX session: {}", e))?;
        
        // 5. Extract output audio
        // The Kokoro model usually specifies "audio" as the output name, 
        // but some versions use "logits" or first available.
        // We clone the data into an owned Vec inside each branch to avoid lifetime issues.
        let audio_vec = if let Some(audio) = outputs.get("audio") {
            let (_shape, data) = audio.try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("Failed to extract 'audio' tensor: {}", e))?;
            data.to_vec()
        } else if let Some(logits) = outputs.get("logits") {
            let (_shape, data) = logits.try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("Failed to extract 'logits' tensor: {}", e))?;
            data.to_vec()
        } else if let Some(first) = outputs.values().next() {
            let (_shape, data) = first.try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("Failed to extract first available tensor: {}", e))?;
            data.to_vec()
        } else {
            anyhow::bail!("No output found in ONNX session");
        };
            
        Ok(Array1::from_vec(audio_vec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kokoro_load() {
        let model_dir = "/Volumes/sw/onnx_models/Kokoro-82M-v1.0-ONNX";
        if Path::new(model_dir).exists() {
            let model = KokoroModel::new(model_dir);
            assert!(model.is_ok(), "Failed to load Kokoro model: {:?}", model.err());
        }
    }

    #[test]
    fn test_voice_load() {
        let voice_path = "/Volumes/sw/onnx_models/Kokoro-82M-v1.0-ONNX/voices/af_bella.bin";
        if Path::new(voice_path).exists() {
            let voices = KokoroVoices::from_file(voice_path);
            assert!(voices.is_ok(), "Failed to load voices: {:?}", voices.err());
            let voices = voices.unwrap();
            assert!(voices.num_styles() > 0);
            assert!(voices.get_style(0).is_some());
        }
    }
}
