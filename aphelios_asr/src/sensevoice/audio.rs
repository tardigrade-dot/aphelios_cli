use anyhow::{ensure, Context, Result};
use std::path::Path;

use opusic_sys as opus_sys;
use symphonia::core::audio::{Audio, GenericAudioBufferRef};
use symphonia::core::codecs::audio::well_known::CODEC_ID_OPUS;
use symphonia::core::codecs::audio::{AudioDecoderOptions, CODEC_ID_NULL_AUDIO};
use symphonia::core::errors::Error as SymphErr;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

pub fn decode_audio_multi(path: &Path) -> Result<(u32, usize, Vec<Vec<f32>>)> {
    let file = std::fs::File::open(path).with_context(|| format!("open {:?}", path))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path
        .extension()
        .and_then(|e| e.to_str())
    {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe().probe(&hint, mss, FormatOptions::default(), MetadataOptions::default())?;

    let track = format
        .tracks()
        .iter()
        .find(|t| {
            t.codec_params
                .as_ref()
                .and_then(|p| p.audio())
                .map(|a| a.codec != CODEC_ID_NULL_AUDIO)
                .unwrap_or(false)
        })
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no supported audio tracks"))?;

    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| anyhow::anyhow!("no audio codec params"))?;

    let sample_rate = audio_params
        .sample_rate
        .ok_or_else(|| anyhow::anyhow!("missing sample rate"))?;
    let channels = audio_params
        .channels
        .as_ref()
        .map(|c| c.count())
        .unwrap_or(2);

    let audio_params = audio_params.clone();
    let codec_type = audio_params.codec;
    if codec_type == CODEC_ID_OPUS {
        decode_opus_track(format.as_mut(), track, sample_rate, channels)
    } else {
        decode_with_builtin_decoder(format.as_mut(), track, audio_params, sample_rate, channels)
    }
}

#[allow(dead_code)]
fn decode_with_builtin_decoder(
    format: &mut dyn symphonia::core::formats::FormatReader,
    track: symphonia::core::formats::Track,
    audio_params: symphonia::core::codecs::audio::AudioCodecParameters,
    sample_rate: u32,
    channels: usize,
) -> Result<(u32, usize, Vec<Vec<f32>>)> {
    let mut decoder = symphonia::default::get_codecs().make_audio_decoder(&audio_params, &AudioDecoderOptions::default())?;

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
        // Adjust per_channel to match actual decoded channel count
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

    Ok((sample_rate, channels, per_channel))
}

fn decode_opus_track(format: &mut dyn symphonia::core::formats::FormatReader, track: symphonia::core::formats::Track, sample_rate: u32, channels: usize) -> Result<(u32, usize, Vec<Vec<f32>>)> {
    use std::ffi::CStr;

    const MAX_PACKET_DURATION_MS: usize = 120;

    fn opus_error_message(code: i32) -> String {
        unsafe {
            let ptr = opus_sys::opus_strerror(code);
            if ptr.is_null() {
                return format!("code {code}");
            }
            CStr::from_ptr(ptr)
                .to_string_lossy()
                .into_owned()
        }
    }

    struct DecoderGuard(*mut opus_sys::OpusDecoder);

    impl Drop for DecoderGuard {
        fn drop(&mut self) {
            unsafe {
                opus_sys::opus_decoder_destroy(self.0);
            }
        }
    }

    ensure!(channels > 0, "Opus decoder requires at least one channel");
    ensure!(channels <= 2, "Opus decoder currently supports only mono or stereo (got {channels})");
    ensure!(sample_rate <= i32::MAX as u32, "Sample rate {sample_rate} exceeds Opus API range");

    let sample_rate_i32 = sample_rate as i32;
    let channel_i32 = channels as i32;

    let mut err: i32 = opus_sys::OPUS_OK;
    let decoder_ptr = unsafe { opus_sys::opus_decoder_create(sample_rate_i32, channel_i32, &mut err) };
    if decoder_ptr.is_null() || err != opus_sys::OPUS_OK {
        let message = opus_error_message(err);
        anyhow::bail!("create Opus decoder: {message}");
    }

    let decoder = DecoderGuard(decoder_ptr);

    let max_frame_samples = ((sample_rate as usize * MAX_PACKET_DURATION_MS) + 999) / 1000;
    let mut per_channel: Vec<Vec<f32>> = vec![Vec::new(); channels];
    let mut decode_buf = vec![0.0_f32; max_frame_samples.max(1) * channels.max(1)];

    // Skip encoder priming samples if present.
    let mut skip_samples = track.delay.unwrap_or(0) as usize;

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

        let data = &*packet.data;
        if data.is_empty() {
            continue;
        }

        let required_frames = unsafe {
            let frames = opus_sys::opus_packet_get_nb_samples(data.as_ptr(), data.len() as i32, sample_rate_i32);
            if frames <= 0 {
                max_frame_samples as i32
            } else {
                frames
            }
        } as usize;

        if required_frames * channels > decode_buf.len() {
            decode_buf.resize(required_frames * channels, 0.0);
        }

        let frames = unsafe { opus_sys::opus_decode_float(decoder.0, data.as_ptr(), data.len() as i32, decode_buf.as_mut_ptr(), required_frames as i32, 0) };

        if frames < 0 {
            anyhow::bail!("decode Opus packet: {}", opus_error_message(frames));
        }

        let frames = frames as usize;
        if frames == 0 {
            continue;
        }

        let trim_start = packet.trim_start.get() as usize;
        let trim_end = packet.trim_end.get() as usize;
        let mut start = trim_start;
        let end = frames.saturating_sub(trim_end);
        if start >= end {
            continue;
        }

        if skip_samples > 0 {
            let to_skip = skip_samples.min(end - start);
            start += to_skip;
            skip_samples -= to_skip;
            if start >= end {
                continue;
            }
        }

        for frame_idx in start..end {
            let base = frame_idx * channels;
            for ch in 0..channels {
                per_channel[ch].push(decode_buf[base + ch]);
            }
        }
    }

    // Align number of samples across channels in case of unexpected discrepancies.
    if let Some(min_len) = per_channel
        .iter()
        .map(|c| c.len())
        .min()
    {
        for ch in per_channel.iter_mut() {
            ch.truncate(min_len);
        }
    }

    Ok((sample_rate, channels, per_channel))
}

pub fn downmix_to_mono(samples_per_channel: Vec<Vec<f32>>) -> Vec<f32> {
    if samples_per_channel.is_empty() {
        return Vec::new();
    }
    if samples_per_channel.len() == 1 {
        return samples_per_channel
            .into_iter()
            .next()
            .unwrap();
    }
    let num_channels = samples_per_channel.len();
    let num_samples = samples_per_channel
        .iter()
        .map(|c| c.len())
        .max()
        .unwrap_or(0);
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

pub fn resample_channels(samples_per_channel: Vec<Vec<f32>>, src_rate: u32, dst_rate: u32) -> Result<Vec<Vec<f32>>> {
    if src_rate == dst_rate {
        return Ok(samples_per_channel);
    }
    ensure!(src_rate > 0 && dst_rate > 0, "sample rate must be positive");
    let ratio = dst_rate as f64 / src_rate as f64;
    let mut out = Vec::with_capacity(samples_per_channel.len());
    for channel in samples_per_channel.iter() {
        out.push(resample_linear(channel, ratio));
    }
    Ok(out)
}

fn resample_linear(input: &[f32], ratio: f64) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let output_len = ((input.len() as f64) * ratio)
        .ceil()
        .max(1.0) as usize;
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
    out
}
