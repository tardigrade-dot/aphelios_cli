use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

/// Tokenizer for Kokoro TTS.
/// Wraps the `tokenizers` crate but provides a manual fallback if tokenizer.json
/// contains incompatible structures (like certain TemplateProcessing formats).
pub struct KokoroTokenizer {
    tokenizer: Option<Tokenizer>,
    manual_vocab: Option<HashMap<String, u32>>,
}

impl KokoroTokenizer {
    /// Load tokenizer from a local tokenizer.json file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        
        // Try standard loading first
        match Tokenizer::from_file(path) {
            Ok(tokenizer) => Ok(Self {
                tokenizer: Some(tokenizer),
                manual_vocab: None,
            }),
            Err(e) => {
                tracing::warn!("Failed to load tokenizer with tokenizers crate: {}. Falling back to manual vocab loading.", e);
                
                // Manual fallback: Load vocab from JSON
                let content = std::fs::read_to_string(path)
                    .with_context(|| format!("Failed to read tokenizer file: {:?}", path))?;
                let json: serde_json::Value = serde_json::from_str(&content)
                    .context("Failed to parse tokenizer.json")?;
                
                let vocab_value = json.get("model")
                    .and_then(|m| m.get("vocab"))
                    .context("Could not find 'model.vocab' in tokenizer.json")?;
                
                let vocab: HashMap<String, u32> = serde_json::from_value(vocab_value.clone())
                    .context("Failed to deserialize vocab map")?;
                
                Ok(Self {
                    tokenizer: None,
                    manual_vocab: Some(vocab),
                })
            }
        }
    }

    /// Encode IPA phonemes to token IDs.
    /// Kokoro expects: 0 (pad) + tokens + 0 (pad).
    pub fn encode(&self, phonemes: &str) -> Result<Vec<u32>> {
        let ids = if let Some(ref tokenizer) = self.tokenizer {
            let encoding = tokenizer.encode(phonemes, false)
                .map_err(|e| anyhow::anyhow!("Failed to encode phonemes: {}", e))?;
            encoding.get_ids().to_vec()
        } else if let Some(ref vocab) = self.manual_vocab {
            // Simple character-based tokenization as used by Kokoro
            phonemes.chars().map(|c| {
                let s = c.to_string();
                vocab.get(&s).cloned().unwrap_or(0) // Default to pad if unknown, though cleaning should prevent this
            }).collect()
        } else {
            anyhow::bail!("Tokenizer not initialized");
        };
        
        // Wrap with 0 (pad token) as per the Python implementation
        let mut wrapped_ids = Vec::with_capacity(ids.len() + 2);
        wrapped_ids.push(0);
        wrapped_ids.extend(ids);
        wrapped_ids.push(0);
        
        Ok(wrapped_ids)
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        if let Some(ref tokenizer) = self.tokenizer {
            tokenizer.get_vocab_size(true)
        } else if let Some(ref vocab) = self.manual_vocab {
            vocab.len()
        } else {
            0
        }
    }
}
