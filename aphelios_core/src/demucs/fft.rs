use std::collections::HashMap;
use std::sync::Mutex;
use lazy_static::lazy_static;

// 使用 lazy_static 来创建全局缓存
lazy_static! {
    static ref FFT_TWIDDLES: Mutex<HashMap<usize, (Vec<f32>, Vec<f32>)>> = Mutex::new(HashMap::new());
    static ref IFFT_TWIDDLES: Mutex<HashMap<usize, (Vec<f32>, Vec<f32>)>> = Mutex::new(HashMap::new());
    static ref HANN_WINDOWS: Mutex<HashMap<usize, Vec<f32>>> = Mutex::new(HashMap::new());
}

fn get_fft_twiddles(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cache = FFT_TWIDDLES.lock().unwrap();
    
    if let Some(twiddles) = cache.get(&n) {
        return twiddles.clone();
    }
    
    let mut real = vec![0.0; n / 2];
    let mut imag = vec![0.0; n / 2];
    
    for k in 0..n/2 {
        let angle = -2.0 * std::f32::consts::PI * k as f32 / n as f32;
        real[k] = angle.cos();
        imag[k] = angle.sin();
    }
    
    let twiddles = (real, imag);
    cache.insert(n, twiddles.clone());
    twiddles
}

fn get_ifft_twiddles(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cache = IFFT_TWIDDLES.lock().unwrap();
    
    if let Some(twiddles) = cache.get(&n) {
        return twiddles.clone();
    }
    
    let mut real = vec![0.0; n / 2];
    let mut imag = vec![0.0; n / 2];
    
    for k in 0..n/2 {
        let angle = 2.0 * std::f32::consts::PI * k as f32 / n as f32;
        real[k] = angle.cos();
        imag[k] = angle.sin();
    }
    
    let twiddles = (real, imag);
    cache.insert(n, twiddles.clone());
    twiddles
}

pub fn get_hann_window(size: usize) -> Vec<f32> {
    let mut cache = HANN_WINDOWS.lock().unwrap();
    
    if let Some(window) = cache.get(&size) {
        return window.clone();
    }
    
    let mut window = vec![0.0; size];
    for i in 0..size {
        window[i] = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos());
    }
    
    cache.insert(size, window.clone());
    window
}

fn bit_reverse(mut n: usize, bits: u32) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (n & 1);
        n >>= 1;
    }
    result
}

pub fn fft(real_out: &mut [f32], imag_out: &mut [f32], real_in: &[f32], n: usize) {
    let bits = (n as f64).log2().round() as u32;
    let (twiddles_real, twiddles_imag) = get_fft_twiddles(n);
    
    // 初始化输出数组
    for i in 0..n {
        let j = bit_reverse(i, bits);
        real_out[i] = real_in[j];
        imag_out[i] = 0.0;
    }
    
    let mut size = 2;
    while size <= n {
        let half_size = size / 2;
        let step = n / size;
        
        let mut i = 0;
        while i < n {
            for j in 0..half_size {
                let k = j * step;
                let t_real = twiddles_real[k];
                let t_imag = twiddles_imag[k];
                
                let idx1 = i + j;
                let idx2 = i + j + half_size;
                
                let e_real = real_out[idx1];
                let e_imag = imag_out[idx1];
                
                let o_real = real_out[idx2] * t_real - imag_out[idx2] * t_imag;
                let o_imag = real_out[idx2] * t_imag + imag_out[idx2] * t_real;
                
                real_out[idx1] = e_real + o_real;
                imag_out[idx1] = e_imag + o_imag;
                real_out[idx2] = e_real - o_real;
                imag_out[idx2] = e_imag - o_imag;
            }
            i += size;
        }
        size *= 2;
    }
}

pub fn ifft(real_out: &mut [f32], imag_out: &mut [f32], real_in: &[f32], imag_in: &[f32], n: usize) {
    let bits = (n as f64).log2().round() as u32;
    let (twiddles_real, twiddles_imag) = get_ifft_twiddles(n);
    
    // 初始化输出数组
    for i in 0..n {
        let j = bit_reverse(i, bits);
        real_out[i] = real_in[j];
        imag_out[i] = imag_in[j];
    }
    
    let mut size = 2;
    while size <= n {
        let half_size = size / 2;
        let step = n / size;
        
        let mut i = 0;
        while i < n {
            for j in 0..half_size {
                let k = j * step;
                let t_real = twiddles_real[k];
                let t_imag = twiddles_imag[k];
                
                let idx1 = i + j;
                let idx2 = i + j + half_size;
                
                let e_real = real_out[idx1];
                let e_imag = imag_out[idx1];
                
                let o_real = real_out[idx2] * t_real - imag_out[idx2] * t_imag;
                let o_imag = real_out[idx2] * t_imag + imag_out[idx2] * t_real;
                
                real_out[idx1] = e_real + o_real;
                imag_out[idx1] = e_imag + o_imag;
                real_out[idx2] = e_real - o_real;
                imag_out[idx2] = e_imag - o_imag;
            }
            i += size;
        }
        size *= 2;
    }
    
    // 归一化
    for i in 0..n {
        real_out[i] /= n as f32;
        imag_out[i] /= n as f32;
    }
}

pub struct Spectrogram {
    pub real: Vec<f32>,
    pub imag: Vec<f32>,
    pub num_frames: usize,
    pub num_bins: usize,
}

pub fn stft(signal: &[f32], fft_size: usize, hop_size: usize) -> Spectrogram {
    let num_frames = ((signal.len() - fft_size) / hop_size) + 1;
    let num_bins = fft_size / 2 + 1;
    let window = get_hann_window(fft_size);
    let scale = 1.0 / (fft_size as f32).sqrt();
    
    let mut spec_real = vec![0.0; num_frames * num_bins];
    let mut spec_imag = vec![0.0; num_frames * num_bins];
    let mut frame_real = vec![0.0; fft_size];
    let mut frame_imag = vec![0.0; fft_size];
    let mut windowed_frame = vec![0.0; fft_size];
    
    for frame in 0..num_frames {
        let start = frame * hop_size;
        for i in 0..fft_size {
            windowed_frame[i] = signal[start + i] * window[i];
        }
        fft(&mut frame_real, &mut frame_imag, &windowed_frame, fft_size);
        
        let out_offset = frame * num_bins;
        for k in 0..num_bins {
            spec_real[out_offset + k] = frame_real[k] * scale;
            spec_imag[out_offset + k] = frame_imag[k] * scale;
        }
    }
    
    Spectrogram {
        real: spec_real,
        imag: spec_imag,
        num_frames,
        num_bins,
    }
}

pub fn istft(
    spec_real: &[f32],
    spec_imag: &[f32],
    num_frames: usize,
    num_bins: usize,
    fft_size: usize,
    hop_size: usize,
    length: Option<usize>,
) -> Vec<f32> {
    let output_length = length.unwrap_or((num_frames - 1) * hop_size + fft_size);
    let mut output = vec![0.0; output_length];
    let mut window_sum = vec![0.0; output_length];
    let window = get_hann_window(fft_size);
    let scale = (fft_size as f32).sqrt();
    
    let mut full_real = vec![0.0; fft_size];
    let mut full_imag = vec![0.0; fft_size];
    let mut out_real = vec![0.0; fft_size];
    let mut out_imag = vec![0.0; fft_size];
    
    for frame in 0..num_frames {
        full_real.fill(0.0);
        full_imag.fill(0.0);
        
        for k in 0..num_bins {
            let idx = frame * num_bins + k;
            full_real[k] = spec_real[idx];
            full_imag[k] = spec_imag[idx];
        }
        
        for k in 1..num_bins - 1 {
            let idx = fft_size - k;
            full_real[idx] = full_real[k];
            full_imag[idx] = -full_imag[k];
        }
        
        ifft(&mut out_real, &mut out_imag, &full_real, &full_imag, fft_size);
        
        let start = frame * hop_size;
        for i in 0..fft_size {
            if start + i < output_length {
                output[start + i] += out_real[i] * window[i] * scale;
                window_sum[start + i] += window[i] * window[i];
            }
        }
    }
    
    for i in 0..output_length {
        if window_sum[i] > 1e-8 {
            output[i] /= window_sum[i];
        }
    }
    
    output
}

pub fn reflect_pad(signal: &[f32], pad_left: usize, pad_right: usize) -> Vec<f32> {
    let length = signal.len();
    let mut output = vec![0.0; pad_left + length + pad_right];
    
    for i in 0..pad_left {
        let src_idx = std::cmp::min(pad_left - i, length - 1);
        output[i] = signal[src_idx];
    }
    
    output.splice(pad_left..pad_left + length, signal.iter().cloned());
    
    for i in 0..pad_right {
        let src_idx = std::cmp::max(0, length - 2 - i);
        output[pad_left + length + i] = signal[src_idx];
    }
    
    output
}