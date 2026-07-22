use anyhow::{Context, Result};
use hound::{WavReader, WavSpec, WavWriter};
use std::path::{Path, PathBuf};

pub const SR: u32 = 48_000;

/// Read a WAV file, convert to mono f32 at 48 kHz.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = WavReader::open(path).with_context(|| format!("Failed to open WAV: {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let _max = (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / 32768.0))
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("Failed to read i16 samples")?
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to read f32 samples")?,
    };

    // Mix to mono
    let mono = if channels == 1 {
        samples
    } else {
        let n = samples.len() / channels;
        let mut m = vec![0.0f32; n];
        for (i, chunk) in samples.chunks(channels).enumerate() {
            m[i] = chunk.iter().sum::<f32>() / channels as f32;
        }
        m
    };

    // Resample to SR if needed
    let resampled = if sample_rate == SR {
        mono
    } else {
        resample(&mono, sample_rate, SR)
    };

    Ok((resampled, SR))
}

/// Simple linear-interpolation resampler for mono audio.
fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() {
        return vec![];
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = (input.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = (src_pos - src_idx as f64) as f32;

        if src_idx + 1 < input.len() {
            output.push(input[src_idx] * (1.0 - frac) + input[src_idx + 1] * frac);
        } else {
            output.push(*input.last().unwrap());
        }
    }
    output
}

/// Write mono f32 PCM as 16-bit WAV.
pub fn write_wav(output: &PathBuf, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = WavWriter::create(output, spec).with_context(|| format!("Failed to create WAV: {}", output.display()))?;

    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let v = if clamped < 0.0 {
            (clamped * 32768.0) as i16
        } else {
            (clamped * 32767.0) as i16
        };
        writer
            .write_sample(v)
            .context("Failed to write sample")?;
    }
    writer
        .finalize()
        .context("Failed to finalize WAV")?;
    Ok(())
}
