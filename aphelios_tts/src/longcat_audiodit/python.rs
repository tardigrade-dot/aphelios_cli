use crate::longcat_audiodit::{GuidanceMethod, LongCatSynthesisRequest};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct LongCatPythonReference {
    pub python_bin: PathBuf,
    pub inference_script: PathBuf,
    pub model_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct LongCatPythonAudioLoader {
    pub python_bin: PathBuf,
    pub prompt_audio_script: PathBuf,
}

impl LongCatPythonReference {
    pub fn new(
        python_bin: impl AsRef<Path>,
        inference_script: impl AsRef<Path>,
        model_dir: impl AsRef<Path>,
    ) -> Self {
        Self {
            python_bin: python_bin.as_ref().to_path_buf(),
            inference_script: inference_script.as_ref().to_path_buf(),
            model_dir: model_dir.as_ref().to_path_buf(),
        }
    }

    pub fn default_for_local_repo(model_dir: impl AsRef<Path>) -> Self {
        Self::new(
            "/Volumes/sw/conda_envs/lcataudio/bin/python",
            "/Users/larry/coderesp/aphelios_cli/aphelios_tts/scripts/longcat_reference.py",
            model_dir,
        )
    }

    pub fn validate(&self) -> Result<()> {
        for path in [&self.python_bin, &self.inference_script, &self.model_dir] {
            if !path.exists() {
                bail!("LongCat python reference path missing: {}", path.display());
            }
        }
        Ok(())
    }

    pub fn synthesize_to_file(
        &self,
        request: &LongCatSynthesisRequest,
        output_audio: impl AsRef<Path>,
    ) -> Result<()> {
        self.validate()?;
        run_python_reference(self, request, output_audio)
    }
}

impl LongCatPythonAudioLoader {
    pub fn new(
        python_bin: impl AsRef<Path>,
        prompt_audio_script: impl AsRef<Path>,
    ) -> Self {
        Self {
            python_bin: python_bin.as_ref().to_path_buf(),
            prompt_audio_script: prompt_audio_script.as_ref().to_path_buf(),
        }
    }

    pub fn default_for_local_repo() -> Self {
        Self::new(
            "/Volumes/sw/conda_envs/lcataudio/bin/python",
            "/Users/larry/coderesp/aphelios_cli/aphelios_tts/scripts/longcat_prompt_audio.py",
        )
    }

    pub fn validate(&self) -> Result<()> {
        for path in [&self.python_bin, &self.prompt_audio_script] {
            if !path.exists() {
                bail!("LongCat python prompt-audio path missing: {}", path.display());
            }
        }
        Ok(())
    }

    pub fn load_audio_f32(
        &self,
        audio_path: impl AsRef<Path>,
        sample_rate: u32,
    ) -> Result<Vec<f32>> {
        self.validate()?;
        let audio_path = audio_path.as_ref();
        let output = Command::new(&self.python_bin)
            .env("NUMBA_CACHE_DIR", "/tmp/numba-cache")
            .arg(&self.prompt_audio_script)
            .arg("--audio_path")
            .arg(audio_path)
            .arg("--sample_rate")
            .arg(sample_rate.to_string())
            .output()
            .with_context(|| {
                format!(
                    "failed to launch python prompt-audio loader: {}",
                    self.python_bin.display()
                )
            })?;
        if !output.status.success() {
            bail!(
                "LongCat python prompt-audio loader failed: {}\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        if output.stdout.len() % 4 != 0 {
            bail!(
                "python prompt-audio loader returned invalid byte count {}",
                output.stdout.len()
            );
        }
        let samples = output
            .stdout
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        Ok(samples)
    }
}

pub fn run_python_reference(
    reference: &LongCatPythonReference,
    request: &LongCatSynthesisRequest,
    output_audio: impl AsRef<Path>,
) -> Result<()> {
    let output_audio = output_audio.as_ref();
    let mut command = Command::new(&reference.python_bin);
    command
        .env("NUMBA_CACHE_DIR", "/tmp/numba-cache")
        .env("HF_HUB_OFFLINE", "1")
        .arg(&reference.inference_script)
        .arg("--text")
        .arg(&request.text)
        .arg("--output_audio")
        .arg(output_audio)
        .arg("--model_dir")
        .arg(&reference.model_dir)
        .arg("--nfe")
        .arg(request.steps.to_string())
        .arg("--guidance_strength")
        .arg(request.cfg_strength.to_string())
        .arg("--guidance_method")
        .arg(request.guidance_method.as_str())
        .arg("--seed")
        .arg(request.seed.to_string());

    if let Some(prompt_text) = request.prompt_text.as_deref() {
        command.arg("--prompt_text").arg(prompt_text);
    }
    if let Some(prompt_audio) = request.prompt_audio.as_ref() {
        command.arg("--prompt_audio").arg(prompt_audio);
    }

    let output = command.output().with_context(|| {
        format!(
            "failed to launch python reference: {}",
            reference.python_bin.display()
        )
    })?;
    if !output.status.success() {
        bail!(
            "LongCat python reference failed: {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}
