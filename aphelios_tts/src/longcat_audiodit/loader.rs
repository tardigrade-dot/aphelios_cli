use crate::longcat_audiodit::config::AudioDiTConfig;
use anyhow::{anyhow, bail, Context, Result};
use safetensors::SafeTensors;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub model_dir: PathBuf,
    pub config: PathBuf,
    pub weights: PathBuf,
    pub tokenizer: Option<PathBuf>,
}

impl ModelPaths {
    pub fn discover(
        model_dir: impl AsRef<Path>,
        tokenizer: Option<impl AsRef<Path>>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let config = model_dir.join("config.json");
        let weights = model_dir.join("model.safetensors");
        let tokenizer = tokenizer
            .map(|path| path.as_ref().to_path_buf())
            .or_else(|| {
                let local = model_dir.join("tokenizer.json");
                local.exists().then_some(local)
            });

        for required in [&config, &weights] {
            if !required.exists() {
                bail!("required LongCat file missing: {}", required.display());
            }
        }

        Ok(Self {
            model_dir,
            config,
            weights,
            tokenizer,
        })
    }

    pub fn load_config(&self) -> Result<AudioDiTConfig> {
        AudioDiTConfig::from_path(&self.config)
    }
}

#[derive(Debug, Clone)]
pub struct TensorShapeInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct WeightIndex {
    tensors: BTreeMap<String, TensorShapeInfo>,
}

impl WeightIndex {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read safetensors file: {}", path.display()))?;
        let safe_tensors = SafeTensors::deserialize(&bytes)
            .map_err(|err| anyhow!("failed to parse safetensors {}: {err}", path.display()))?;
        let mut tensors = BTreeMap::new();
        for name in safe_tensors.names() {
            let view = safe_tensors
                .tensor(name)
                .map_err(|err| anyhow!("failed to inspect tensor {name}: {err}"))?;
            tensors.insert(
                name.to_string(),
                TensorShapeInfo {
                    dtype: format!("{:?}", view.dtype()),
                    shape: view.shape().to_vec(),
                },
            );
        }
        Ok(Self { tensors })
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|entry| entry.shape.as_slice())
    }

    pub fn prefix_count(&self, prefix: &str) -> usize {
        self.tensors
            .keys()
            .filter(|name| name.starts_with(prefix))
            .count()
    }

    pub fn prefix_entries(&self, prefix: &str) -> Vec<(&str, &TensorShapeInfo)> {
        self.tensors
            .iter()
            .filter(|(name, _)| name.starts_with(prefix))
            .map(|(name, shape)| (name.as_str(), shape))
            .collect()
    }

    pub fn require_prefix(&self, prefix: &str) -> Result<()> {
        if self.prefix_count(prefix) == 0 {
            bail!("missing tensor prefix `{prefix}` in LongCat safetensors");
        }
        Ok(())
    }

    pub fn summary(&self) -> WeightSummary {
        WeightSummary {
            total_tensors: self.len(),
            text_encoder_tensors: self.prefix_count("text_encoder."),
            transformer_tensors: self.prefix_count("transformer."),
            vae_tensors: self.prefix_count("vae."),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightSummary {
    pub total_tensors: usize,
    pub text_encoder_tensors: usize,
    pub transformer_tensors: usize,
    pub vae_tensors: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_is_prefix_based() {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "text_encoder.embed.weight".to_string(),
            TensorShapeInfo {
                dtype: "F32".to_string(),
                shape: vec![4, 4],
            },
        );
        tensors.insert(
            "transformer.proj_out.weight".to_string(),
            TensorShapeInfo {
                dtype: "F32".to_string(),
                shape: vec![4, 4],
            },
        );
        tensors.insert(
            "vae.decoder.weight".to_string(),
            TensorShapeInfo {
                dtype: "F32".to_string(),
                shape: vec![4, 4],
            },
        );
        let index = WeightIndex { tensors };
        assert_eq!(
            index.summary(),
            WeightSummary {
                total_tensors: 3,
                text_encoder_tensors: 1,
                transformer_tensors: 1,
                vae_tensors: 1,
            }
        );
    }
}
