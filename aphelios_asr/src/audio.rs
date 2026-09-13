use anyhow::{bail, Result};
use hound::{SampleFormat, WavReader};
use std::path::Path;

use symphonia::core::audio::{Audio, GenericAudioBufferRef};
use symphonia::core::codecs::audio::{AudioDecoderOptions, CODEC_ID_NULL_AUDIO};
use symphonia::core::common::Limit;
use symphonia::core::errors::Error as SymphErr;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

pub const TARGET_SAMPLE_RATE: u32 = 16000;

#[derive(Debug, Clone)]
pub struct MonoBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl MonoBuffer {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self { samples, sample_rate }
    }

    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn samples(&self) -> Vec<f32> {
        self.samples.clone()
    }
}

#[derive(Debug, Clone)]
pub struct StereoBuffer {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub sample_rate: u32,
}

impl StereoBuffer {
    pub fn new(left: Vec<f32>, right: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            left,
            right,
            sample_rate,
        }
    }

    pub fn from_mono(mono: &MonoBuffer) -> Self {
        Self {
            left: mono.samples.clone(),
            right: mono.samples.clone(),
            sample_rate: mono.sample_rate,
        }
    }

    pub fn to_mono(&self) -> MonoBuffer {
        let samples: Vec<f32> = self
            .left
            .iter()
            .zip(self.right.iter())
            .map(|(&l, &r)| (l + r) / 2.0)
            .collect();
        MonoBuffer::new(samples, self.sample_rate)
    }

    pub fn duration_secs(&self) -> f64 {
        self.left.len() as f64 / self.sample_rate as f64
    }

    pub fn len(&self) -> usize {
        self.left.len()
    }

    pub fn is_empty(&self) -> bool {
        self.left.is_empty()
    }
}

#[derive(Debug, Clone)]
pub enum AudioBuffer {
    Mono(MonoBuffer),
    Stereo(StereoBuffer),
}

impl AudioBuffer {
    pub fn sample_rate(&self) -> u32 {
        match self {
            Self::Mono(m) => m.sample_rate,
            Self::Stereo(s) => s.sample_rate,
        }
    }

    pub fn is_stereo(&self) -> bool {
        matches!(self, Self::Stereo(_))
    }

    pub fn is_mono(&self) -> bool {
        matches!(self, Self::Mono(_))
    }

    pub fn into_mono(self) -> MonoBuffer {
        match self {
            Self::Mono(m) => m,
            Self::Stereo(s) => s.to_mono(),
        }
    }

    pub fn duration_secs(&self) -> f64 {
        match self {
            Self::Mono(m) => m.duration_secs(),
            Self::Stereo(s) => s.duration_secs(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Mono(m) => m.len(),
            Self::Stereo(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn to_vec_f32(&self) -> Vec<f32> {
        match self {
            Self::Mono(m) => m.samples.clone(),
            Self::Stereo(s) => s
                .left
                .iter()
                .zip(&s.right)
                .map(|(l, r)| (l + r) * 0.5)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResampleQuality {
    #[default]
    Fast,
    High,
}

pub struct Resampler {
    quality: ResampleQuality,
}

impl Resampler {
    pub fn new() -> Self {
        Self {
            quality: ResampleQuality::default(),
        }
    }

    pub fn with_quality(mut self, quality: ResampleQuality) -> Self {
        self.quality = quality;
        self
    }

    pub fn resample_mono(&self, input: &MonoBuffer, target_rate: u32) -> Result<MonoBuffer> {
        if input.sample_rate == target_rate {
            return Ok(input.clone());
        }

        let samples = match self.quality {
            ResampleQuality::Fast => self.linear_resample(&input.samples, input.sample_rate, target_rate),
            ResampleQuality::High => self.sinc_resample(&input.samples, input.sample_rate, target_rate),
        };

        Ok(MonoBuffer::new(samples, target_rate))
    }

    fn linear_resample(&self, samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
        let ratio = dst_rate as f64 / src_rate as f64;
        let new_len = (samples.len() as f64 * ratio).floor() as usize;

        if new_len == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_idx = i as f64 / ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(samples.len() - 1);
            let frac = src_idx - idx0 as f64;

            let sample = samples[idx0] as f64 * (1.0 - frac) + samples[idx1] as f64 * frac;
            result.push(sample as f32);
        }

        result
    }

    fn sinc_resample(&self, samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
        use rubato::audioadapter_buffers::direct::InterleavedSlice;
        use rubato::{Async, FixedAsync, Indexing, Resampler as RubatoResampler, SincInterpolationParameters, SincInterpolationType, WindowFunction};

        let ratio = dst_rate as f64 / src_rate as f64;

        let params = SincInterpolationParameters::new(256, WindowFunction::BlackmanHarris2)
            .f_cutoff(0.95)
            .oversampling_factor(256)
            .interpolation(SincInterpolationType::Linear);

        let chunk_size = 4096;
        let mut resampler = Async::<f32>::new_sinc(ratio, 2.0, &params, chunk_size, 1, FixedAsync::Input)
            .expect("Failed to create resampler");

        let input_frames = samples.len();
        if input_frames == 0 {
            return Vec::new();
        }

        let input_adapter = InterleavedSlice::new(samples, 1, input_frames).expect("Failed to create input adapter");
        let estimated_frames = 2 * ((input_frames as f64) * ratio).ceil() as usize + resampler.output_delay() + chunk_size;
        let mut output_data = vec![0.0f32; estimated_frames.max(1024)];
        let output_adapter_frames = output_data.len();
        let mut output_adapter = InterleavedSlice::new_mut(&mut output_data, 1, output_adapter_frames).expect("Failed to create output adapter");

        let mut indexing = Indexing::new();
        let mut input_frames_left = input_frames;
        let mut input_frames_next = resampler.input_frames_next();

        while input_frames_left >= input_frames_next {
            let (nbr_in, nbr_out) = resampler
                .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
                .unwrap_or((0, 0));
            if nbr_in == 0 {
                break;
            }
            indexing.input_offset += nbr_in;
            indexing.output_offset += nbr_out;
            input_frames_left -= nbr_in;
            input_frames_next = resampler.input_frames_next();
        }

        if input_frames_left > 0 {
            indexing.partial_len = Some(input_frames_left);
            if let Ok((_nbr_in, nbr_out)) = resampler.process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing)) {
                indexing.output_offset += nbr_out;
            }
        }

        output_data.truncate(indexing.output_offset);
        let target_len = (input_frames as f64 * ratio).floor() as usize;
        if output_data.len() > target_len {
            output_data.truncate(target_len);
        }

        output_data
    }
}

impl Default for Resampler {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AudioLoader {
    normalize: bool,
}

impl AudioLoader {
    pub fn new() -> Self {
        Self { normalize: true }
    }

    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    pub fn load(&self, path: impl AsRef<Path>) -> Result<AudioBuffer> {
        let path = path.as_ref();

        if !path.exists() {
            bail!("Audio file not found: {}", path.display());
        }

        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "wav" => self.load_wav(path),
            _ => self.load_with_symphonia(path),
        }
    }

    fn load_with_symphonia(&self, path: &Path) -> Result<AudioBuffer> {
        let file = std::fs::File::open(path)?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        let metadata_options = MetadataOptions::default().limit_tag_bytes(Limit::Maximum(0));
        let metadata = std::fs::metadata(path)?;
        if metadata.len() == 0 {
            bail!("The audio file is empty (0 bytes): {:?}", path);
        }

        let mut format = symphonia::default::get_probe().probe(&hint, mss, FormatOptions::default(), metadata_options)?;

        let track = format
            .tracks()
            .iter()
            .find(|t| {
                t.codec_params
                    .as_ref()
                    .and_then(|p| p.audio())
                    .map(|a| a.codec != CODEC_ID_NULL_AUDIO && a.sample_rate.is_some())
                    .unwrap_or(false)
            })
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no supported audio tracks"))?;

        let audio_params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .ok_or_else(|| anyhow::anyhow!("no audio codec params"))?;

        let src_rate = audio_params
            .sample_rate
            .ok_or_else(|| anyhow::anyhow!("missing sample rate"))?;

        let channels = audio_params
            .channels
            .as_ref()
            .map(|c| c.count())
            .unwrap_or(2);

        let mut decoder = symphonia::default::get_codecs().make_audio_decoder(audio_params, &AudioDecoderOptions::default())?;
        let mut per_channel: Vec<Vec<f32>> = vec![Vec::new(); channels];

        loop {
            let packet = match format.next_packet() {
                Ok(Some(p)) => p,
                Ok(None) => break,
                Err(SymphErr::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            };

            if packet.track_id != track.id {
                continue;
            }

            let decoded = match decoder.decode(&packet) {
                Ok(s) => s,
                Err(SymphErr::DecodeError(_)) => continue,
                Err(e) => return Err(e.into()),
            };

            let chans = decoded.spec().channels().count();
            if chans > per_channel.len() {
                per_channel.resize_with(chans, Vec::new);
            } else if chans < per_channel.len() {
                per_channel.truncate(chans);
            }

            match &decoded {
                GenericAudioBufferRef::F32(buf) => {
                    for ch in 0..chans {
                        if let Some(plane) = buf.plane(ch) {
                            per_channel[ch].extend(plane);
                        }
                    }
                }
                _ => {
                    let mut temp: Vec<Vec<f32>> = Vec::new();
                    decoded.copy_to_vecs_planar(&mut temp);
                    for ch in 0..chans {
                        if let Some(plane) = temp.get(ch) {
                            per_channel[ch].extend(plane);
                        }
                    }
                }
            }
        }

        let mono = downmix_to_mono(per_channel);
        let mono = if src_rate != TARGET_SAMPLE_RATE {
            resample_linear(&mono, src_rate, TARGET_SAMPLE_RATE)?
        } else {
            mono
        };

        Ok(AudioBuffer::Mono(MonoBuffer::new(mono, TARGET_SAMPLE_RATE)))
    }

    fn load_wav(&self, path: &Path) -> Result<AudioBuffer> {
        let mut reader = WavReader::open(path)?;
        let spec = reader.spec();
        let samples = self.read_samples(&mut reader, spec)?;

        let mono = if spec.channels == 1 {
            samples
        } else {
            let left: Vec<f32> = samples.iter().step_by(2).copied().collect();
            let right: Vec<f32> = samples.iter().skip(1).step_by(2).copied().collect();
            downmix_two_channels(&left, &right)
        };

        let mono = if spec.sample_rate != TARGET_SAMPLE_RATE {
            resample_linear(&mono, spec.sample_rate, TARGET_SAMPLE_RATE)?
        } else {
            mono
        };

        Ok(AudioBuffer::Mono(MonoBuffer::new(mono, TARGET_SAMPLE_RATE)))
    }

    fn read_samples<R: std::io::Read>(&self, reader: &mut WavReader<R>, spec: hound::WavSpec) -> Result<Vec<f32>> {
        match spec.sample_format {
            SampleFormat::Int => self.read_int_samples(reader, spec.bits_per_sample),
            SampleFormat::Float => self.read_float_samples(reader),
        }
    }

    fn read_int_samples<R: std::io::Read>(&self, reader: &mut WavReader<R>, bits: u16) -> Result<Vec<f32>> {
        match bits {
            8 => Ok(reader
                .samples::<i8>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / i8::MAX as f32)
                .collect()),
            16 => Ok(reader
                .samples::<i16>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / i16::MAX as f32)
                .collect()),
            24 | 32 => Ok(reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| match bits {
                    24 => s as f32 / 8_388_607.0,
                    _ => s as f32 / i32::MAX as f32,
                })
                .collect()),
            _ => bail!("Unsupported bit depth: {}", bits),
        }
    }

    fn read_float_samples<R: std::io::Read>(&self, reader: &mut WavReader<R>) -> Result<Vec<f32>> {
        Ok(reader
            .samples::<f32>()
            .filter_map(|s| s.ok())
            .collect())
    }
}

impl Default for AudioLoader {
    fn default() -> Self {
        Self::new()
    }
}

fn downmix_to_mono(samples_per_channel: Vec<Vec<f32>>) -> Vec<f32> {
    if samples_per_channel.is_empty() {
        return Vec::new();
    }
    if samples_per_channel.len() == 1 {
        return samples_per_channel.into_iter().next().unwrap();
    }
    let num_channels = samples_per_channel.len();
    let num_samples = samples_per_channel.iter().map(|c| c.len()).max().unwrap_or(0);
    let mut mono = vec![0.0f32; num_samples];
    for ch in &samples_per_channel {
        for (i, sample) in ch.iter().enumerate() {
            mono[i] += *sample;
        }
    }
    let inv_channels = 1.0 / num_channels as f32;
    for sample in mono.iter_mut() {
        *sample *= inv_channels;
    }
    mono
}

fn downmix_two_channels(left: &[f32], right: &[f32]) -> Vec<f32> {
    let len = left.len().min(right.len());
    let mut mono = Vec::with_capacity(len);
    for i in 0..len {
        mono.push((left[i] + right[i]) * 0.5);
    }
    mono
}

fn resample_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    if src_rate == dst_rate {
        return Ok(input.to_vec());
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let output_len = ((input.len() as f64) * ratio).ceil().max(1.0) as usize;
    let mut out = Vec::with_capacity(output_len);
    let last = input.len() - 1;

    for n in 0..output_len {
        let pos = (n as f64) / ratio;
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;
        let i0 = idx.min(last);
        let i1 = (idx + 1).min(last);
        let s0 = input[i0];
        let s1 = input[i1];
        out.push(s0 + (s1 - s0) * frac);
    }

    Ok(out)
}
