use crate::demucs::constants::Constants;
use crate::demucs::fft::{istft, reflect_pad, stft};
use ort::ep::{CPU, CoreML};
use ort::session::Session;
use ort::value::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct TrackSpec {
    pub left_real: Vec<f32>,
    pub left_imag: Vec<f32>,
    pub right_real: Vec<f32>,
    pub right_imag: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct StereoAudio {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SeparatedTracks {
    pub drums: StereoAudio,
    pub bass: StereoAudio,
    pub other: StereoAudio,
    pub vocals: StereoAudio,
}

pub fn standalone_mask(freq_output: &[f32]) -> Vec<TrackSpec> {
    let num_tracks = 4;
    let num_channels = 4;
    let num_bins = Constants::MODEL_SPEC_BINS;
    let num_frames = Constants::MODEL_SPEC_FRAMES;
    let result = (0..num_tracks)
        .map(|t| {
            let mut track_spec = TrackSpec {
                left_real: vec![0.0; num_bins * num_frames],
                left_imag: vec![0.0; num_bins * num_frames],
                right_real: vec![0.0; num_bins * num_frames],
                right_imag: vec![0.0; num_bins * num_frames],
            };

            for f in 0..num_frames {
                for b in 0..num_bins {
                    let base_idx = t * num_channels * num_bins * num_frames;
                    let out_idx = b * num_frames + f;
                    track_spec.left_real[out_idx] =
                        freq_output[base_idx + 0 * num_bins * num_frames + b * num_frames + f];
                    track_spec.left_imag[out_idx] =
                        freq_output[base_idx + 1 * num_bins * num_frames + b * num_frames + f];
                    track_spec.right_real[out_idx] =
                        freq_output[base_idx + 2 * num_bins * num_frames + b * num_frames + f];
                    track_spec.right_imag[out_idx] =
                        freq_output[base_idx + 3 * num_bins * num_frames + b * num_frames + f];
                }
            }
            track_spec
        })
        .collect();

    result
}

pub fn standalone_ispec(track_spec: &TrackSpec, target_length: usize) -> StereoAudio {
    let num_bins = Constants::MODEL_SPEC_BINS;
    let num_frames = Constants::MODEL_SPEC_FRAMES;
    let hop_length = Constants::HOP_SIZE;
    let padded_bins = num_bins + 1;
    let padded_frames = num_frames + 4;

    let pad_channel = |real: &[f32], imag: &[f32]| -> (Vec<f32>, Vec<f32>) {
        let mut padded_real = vec![0.0; padded_frames * padded_bins];
        let mut padded_imag = vec![0.0; padded_frames * padded_bins];

        for f in 0..num_frames {
            for b in 0..num_bins {
                let src_idx = b * num_frames + f;
                let dst_frame = f + 2;
                let dst_idx = dst_frame * padded_bins + b;
                padded_real[dst_idx] = real[src_idx];
                padded_imag[dst_idx] = imag[src_idx];
            }
        }
        (padded_real, padded_imag)
    };

    let (left_padded_real, left_padded_imag) =
        pad_channel(&track_spec.left_real, &track_spec.left_imag);
    let (right_padded_real, right_padded_imag) =
        pad_channel(&track_spec.right_real, &track_spec.right_imag);

    let center_pad = Constants::FFT_SIZE / 2;
    let pad = (hop_length / 2) * 3;
    let istft_length = (padded_frames - 1) * hop_length + Constants::FFT_SIZE;

    let left_out = istft(
        &left_padded_real,
        &left_padded_imag,
        padded_frames,
        padded_bins,
        Constants::FFT_SIZE,
        hop_length,
        Some(istft_length),
    );
    let right_out = istft(
        &right_padded_real,
        &right_padded_imag,
        padded_frames,
        padded_bins,
        Constants::FFT_SIZE,
        hop_length,
        Some(istft_length),
    );

    let total_offset = center_pad + pad;
    let left_start = total_offset;
    let left_end = std::cmp::min(total_offset + target_length, left_out.len());
    let right_start = total_offset;
    let right_end = std::cmp::min(total_offset + target_length, right_out.len());

    let left = if left_start < left_out.len() {
        left_out[left_start..left_end].to_vec()
    } else {
        vec![0.0; target_length]
    };
    let right = if right_start < right_out.len() {
        right_out[right_start..right_end].to_vec()
    } else {
        vec![0.0; target_length]
    };

    StereoAudio { left, right }
}

pub fn prepare_model_input(left_channel: &[f32], right_channel: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let input_length = Constants::TRAINING_SAMPLES;

    let mut padded_left = vec![0.0; input_length];
    let mut padded_right = vec![0.0; input_length];
    let copy_len = std::cmp::min(left_channel.len(), input_length);
    if copy_len > 0 {
        padded_left[..copy_len].copy_from_slice(&left_channel[..copy_len]);
        padded_right[..copy_len].copy_from_slice(&right_channel[..copy_len]);
    }

    let le = ((input_length as f32) / (Constants::HOP_SIZE as f32)).ceil() as usize;
    let pad = (Constants::HOP_SIZE / 2) * 3;
    let pad_right = pad + le * Constants::HOP_SIZE - input_length;

    let stft_input_left = reflect_pad(&padded_left, pad, pad_right);
    let stft_input_right = reflect_pad(&padded_right, pad, pad_right);

    let center_pad = Constants::FFT_SIZE / 2;
    let centered_left = reflect_pad(&stft_input_left, center_pad, center_pad);
    let centered_right = reflect_pad(&stft_input_right, center_pad, center_pad);

    let stft_left = stft(&centered_left, Constants::FFT_SIZE, Constants::HOP_SIZE);
    let stft_right = stft(&centered_right, Constants::FFT_SIZE, Constants::HOP_SIZE);

    let num_bins = Constants::MODEL_SPEC_BINS;
    let num_frames = Constants::MODEL_SPEC_FRAMES;
    let frame_offset = 2;

    let mut mag_spec = vec![0.0; 4 * num_bins * num_frames];

    for f in 0..num_frames {
        let src_frame = f + frame_offset;
        for b in 0..num_bins {
            let src_idx = src_frame * stft_left.num_bins + b;
            if src_idx < stft_left.real.len() {
                mag_spec[0 * num_bins * num_frames + b * num_frames + f] = stft_left.real[src_idx];
                mag_spec[1 * num_bins * num_frames + b * num_frames + f] = stft_left.imag[src_idx];
                mag_spec[2 * num_bins * num_frames + b * num_frames + f] = stft_right.real[src_idx];
                mag_spec[3 * num_bins * num_frames + b * num_frames + f] = stft_right.imag[src_idx];
            }
        }
    }

    // waveform 应该是 (2, input_length) 形状，但模型期望 (batch, channels, samples)
    let mut waveform = Vec::with_capacity(2 * input_length);
    waveform.extend_from_slice(&padded_left);
    waveform.extend_from_slice(&padded_right);

    (waveform, mag_spec)
}

pub struct DemucsProcessor {
    pub model_path: String,
    pub session: Option<Arc<Mutex<Session>>>,
}

impl DemucsProcessor {
    pub fn new(model_path: String) -> anyhow::Result<Self, Box<dyn std::error::Error>> {
        Ok(DemucsProcessor {
            model_path,
            session: None,
        })
    }

    /// coreml会卡住
    pub fn load_model(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let coreml_options = CoreML::default().with_subgraphs(true);
        let cpu_provider = CPU::default().build();
        let coreml_provider = coreml_options.build();
        // let session = Session::builder()?.with_execution_providers([execution_providers])?.commit_from_file(&self.model_path)?;
        let session = Session::builder()?
            .with_execution_providers([cpu_provider])?
            .commit_from_file(&self.model_path)?;
        self.session = Some(Arc::new(Mutex::new(session)));
        Ok(())
    }

    pub fn separate(
        &self,
        left_channel: &[f32],
        right_channel: &[f32],
    ) -> Result<SeparatedTracks, Box<dyn std::error::Error>> {
        let session_mutex = self.session.as_ref().ok_or("Model not loaded")?;
        let mut session = session_mutex
            .lock()
            .map_err(|_| "Failed to acquire session lock")?;

        let total_samples = left_channel.len();
        let stride =
            (Constants::TRAINING_SAMPLES as f32 * (1.0 - Constants::SEGMENT_OVERLAP)) as usize;
        let num_segments = (((total_samples - Constants::TRAINING_SAMPLES) as f32 / stride as f32)
            .ceil() as usize)
            + 1;

        eprintln!(
            "[Demucs] Total samples: {}, stride: {}, segments: {}",
            total_samples, stride, num_segments
        );

        let mut outputs = [
            StereoAudio {
                left: vec![0.0; total_samples],
                right: vec![0.0; total_samples],
            },
            StereoAudio {
                left: vec![0.0; total_samples],
                right: vec![0.0; total_samples],
            },
            StereoAudio {
                left: vec![0.0; total_samples],
                right: vec![0.0; total_samples],
            },
            StereoAudio {
                left: vec![0.0; total_samples],
                right: vec![0.0; total_samples],
            },
        ];
        let mut weights = vec![0.0; total_samples];

        let mut segment_idx = 0;
        let total_start = std::time::Instant::now();

        let mut start = 0;
        while start < total_samples {
            let seg_start = std::time::Instant::now();
            let end = std::cmp::min(start + Constants::TRAINING_SAMPLES, total_samples);
            let segment_length = end - start;

            let mut seg_left = vec![0.0; Constants::TRAINING_SAMPLES];
            let mut seg_right = vec![0.0; Constants::TRAINING_SAMPLES];

            for i in 0..segment_length {
                if start + i < left_channel.len() {
                    seg_left[i] = left_channel[start + i];
                }
                if start + i < right_channel.len() {
                    seg_right[i] = right_channel[start + i];
                }
            }

            let (waveform, mag_spec) = prepare_model_input(&seg_left, &seg_right);

            // 创建 ONNX 输入张量
            let mut waveform_with_batch = Vec::with_capacity(1 * 2 * Constants::TRAINING_SAMPLES);
            waveform_with_batch
                .extend(std::iter::repeat(0.0).take(1 * 2 * Constants::TRAINING_SAMPLES));
            // 假设 waveform 已经是 (2, TRAINING_SAMPLES) 的形状
            let waveform_array = ndarray::Array3::<f32>::from_shape_vec(
                (1, 2, Constants::TRAINING_SAMPLES),
                waveform,
            )?;
            let waveform_tensor = Value::from_array(waveform_array)?;

            let mag_spec_array = ndarray::Array4::<f32>::from_shape_vec(
                (
                    1,
                    4,
                    Constants::MODEL_SPEC_BINS,
                    Constants::MODEL_SPEC_FRAMES,
                ),
                mag_spec,
            )?;
            let mag_spec_tensor = Value::from_array(mag_spec_array)?;

            // 准备输入映射
            let mut inputs = HashMap::new();

            // 获取模型输入名称
            let input_names: Vec<String> = session
                .inputs()
                .iter()
                .map(|input| input.name().to_string())
                .collect();

            if !input_names.is_empty() {
                inputs.insert(input_names[0].clone(), waveform_tensor);
            }
            if input_names.len() > 1 {
                inputs.insert(input_names[1].clone(), mag_spec_tensor);
            }

            // 运行推理
            let outputs_map = session.run(inputs)?;

            // 处理输出
            let mut time_data: Option<Vec<f32>> = None;
            let mut time_shape: Option<Vec<usize>> = None;
            let mut freq_data: Option<Vec<f32>> = None;

            for (_name, tensor) in &outputs_map {
                if let Ok((shape_info, data)) = tensor.try_extract_tensor::<f32>() {
                    let shape: Vec<usize> = shape_info.iter().map(|&x| x as usize).collect();
                    let data_vec: Vec<f32> = data.iter().copied().collect();

                    if shape.len() == 4 && shape[2] == 2 {
                        time_data = Some(data_vec);
                        time_shape = Some(shape);
                    } else if shape.len() == 5 && shape[2] == 4 {
                        freq_data = Some(data_vec);
                    }
                }
            }

            if time_data.is_none() {
                return Err("Could not find time-domain output tensor".into());
            }

            let mut combined_outputs: Option<Vec<StereoAudio>> = None;
            if let Some(freq_data_val) = freq_data {
                let track_specs = standalone_mask(&freq_data_val);
                let mut temp_combined_outputs = Vec::with_capacity(4);

                for t in 0..4 {
                    let freq_output =
                        standalone_ispec(&track_specs[t], Constants::TRAINING_SAMPLES);
                    let num_channels = time_shape.as_ref().unwrap()[2];
                    let samples = time_shape.as_ref().unwrap()[3];
                    let time_data_ref = time_data.as_ref().unwrap();

                    let mut time_left = vec![0.0; samples];
                    let mut time_right = vec![0.0; samples];

                    for i in 0..samples {
                        time_left[i] = time_data_ref[t * num_channels * samples + 0 * samples + i];
                        time_right[i] = time_data_ref[t * num_channels * samples + 1 * samples + i];
                    }

                    let mut combined = StereoAudio {
                        left: vec![0.0; samples],
                        right: vec![0.0; samples],
                    };
                    for i in 0..samples {
                        if i < time_left.len() && i < freq_output.left.len() {
                            combined.left[i] = time_left[i] + freq_output.left[i];
                        }
                        if i < time_right.len() && i < freq_output.right.len() {
                            combined.right[i] = time_right[i] + freq_output.right[i];
                        }
                    }
                    temp_combined_outputs.push(combined);
                }
                combined_outputs = Some(temp_combined_outputs);
            }

            let num_tracks = time_shape.as_ref().unwrap()[1];
            let num_channels = time_shape.as_ref().unwrap()[2];
            let samples = time_shape.as_ref().unwrap()[3];
            let time_data_ref = time_data.as_ref().unwrap();

            let mut overlap_window = vec![0.0; segment_length];
            for i in 0..segment_length {
                let fade_in = ((i as f32) / ((stride as f32) * 0.5)).min(1.0);
                let fade_out = (((segment_length - i) as f32) / ((stride as f32) * 0.5)).min(1.0);
                overlap_window[i] = fade_in.min(fade_out);
            }

            for t in 0..num_tracks {
                for i in 0..segment_length {
                    if start + i >= total_samples {
                        break;
                    }

                    let mut left_val = 0.0;
                    let mut right_val = 0.0;

                    if let Some(ref combined_outputs_val) = combined_outputs {
                        if t < combined_outputs_val.len() && i < combined_outputs_val[t].left.len()
                        {
                            left_val = combined_outputs_val[t].left[i];
                            right_val = combined_outputs_val[t].right[i];
                        }
                    } else {
                        let left_idx = t * num_channels * samples + 0 * samples + i;
                        let right_idx = t * num_channels * samples + 1 * samples + i;
                        if left_idx < time_data_ref.len() && right_idx < time_data_ref.len() {
                            left_val = time_data_ref[left_idx];
                            right_val = time_data_ref[right_idx];
                        }
                    }

                    outputs[t].left[start + i] += left_val * overlap_window[i];
                    outputs[t].right[start + i] += right_val * overlap_window[i];
                }
            }

            for i in 0..segment_length {
                if start + i >= total_samples {
                    break;
                }
                weights[start + i] += overlap_window[i];
            }

            // 可以在这里添加进度回调
            segment_idx += 1;
            let seg_elapsed = seg_start.elapsed().as_millis();
            eprintln!(
                "[Demucs] Segment {}/{} completed in {}ms",
                segment_idx, num_segments, seg_elapsed
            );

            start += stride;
        }

        let total_elapsed = total_start.elapsed().as_secs();
        eprintln!("[Demucs] Total separation time: {}s", total_elapsed);

        // 归一化输出
        for t in 0..Constants::TRACKS.len() {
            for i in 0..total_samples {
                if weights[i] > 0.0 {
                    outputs[t].left[i] /= weights[i];
                    outputs[t].right[i] /= weights[i];
                }
            }
        }

        Ok(SeparatedTracks {
            drums: outputs[0].clone(),
            bass: outputs[1].clone(),
            other: outputs[2].clone(),
            vocals: outputs[3].clone(),
        })
    }
}
