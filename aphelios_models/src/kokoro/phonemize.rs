use anyhow::Result;
use misaki_rs::{G2P, Language};
use once_cell::sync::Lazy;

use super::KokoroLanguage;

static G2P_US: Lazy<G2P> = Lazy::new(|| G2P::new(Language::EnglishUS));
static G2P_GB: Lazy<G2P> = Lazy::new(|| G2P::new(Language::EnglishGB));

/// A phonemizer for Kokoro using misaki-rs for English.
pub struct KokoroPhonemizer;

impl KokoroPhonemizer {
    pub fn new() -> Self {
        Self
    }

    /// Clean IPA text and prepare it for the model.
    pub fn clean_ipa(&self, text: &str) -> Result<String> {
        Ok(text.trim().to_string())
    }

    /// Convert raw text to IPA phonemes based on the language.
    pub fn phonemize(&self, text: &str, lang: KokoroLanguage) -> Result<String> {
        match lang {
            KokoroLanguage::EnUs => {
                let (phonemes, _tokens) = G2P_US.g2p(text)
                    .map_err(|e| anyhow::anyhow!("Misaki G2P error: {:?}", e))?;
                Ok(phonemes)
            },
            KokoroLanguage::EnGb => {
                let (phonemes, _tokens) = G2P_GB.g2p(text)
                    .map_err(|e| anyhow::anyhow!("Misaki G2P error: {:?}", e))?;
                Ok(phonemes)
            },
            KokoroLanguage::Zh => {
                anyhow::bail!("Chinese support (Kokoro-82M-v1.1-zh) is not yet implemented in the phonemizer.")
            }
        }
    }
}
