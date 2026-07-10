"""Export SeamlessM4TFeatureExtractor params for Rust reimplementation.

Usage: python scripts/export_sidon_fe.py

Output files (in models/):
  mel_filters.npy — (257, 80) float64 mel filterbank matrix
  mel_window.npy   — (400,) float64 Hann window

Key parameters:
  sample_rate: 16000 Hz
  n_fft: 512
  hop_length: 160 samples (10ms)
  win_length: 400 samples (25ms)
  window: Hann
  num_mel_bins: 80
  stride: 2 (frame stacking → 160-dim features at 50Hz)

Pipeline:
  1. Pad audio both sides (context for edge frames)
  2. STFT: n_fft=512, hop_length=160, win_length=400, Hann window
  3. Power: |STFT|^2, keep first 257 bins
  4. Mel: power @ mel_filters (257×80) → 80 mel bins
  5. Log: log10(mel_power + 1e-10)
  6. ZMUV: zero-mean unit-variance per frame
  7. Stack: concat every 2 adjacent frames → 160-dim features
"""
import numpy as np
from transformers import AutoFeatureExtractor

def main():
    fe = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    # Save raw arrays
    np.save("models/mel_filters.npy", fe.mel_filters.astype(np.float32))
    np.save("models/mel_window.npy", fe.window.astype(np.float32))
    # Also save as raw f32 binary for Rust include_bytes!
    fe.mel_filters.astype(np.float32).tofile("models/mel_filters_f32.bin")
    fe.window.astype(np.float32).tofile("models/mel_window_f32.bin")

    # Print config summary
    print(f"sample_rate:       {fe.sampling_rate}")
    print(f"n_fft:             512")
    print(f"hop_length:        160  (10ms)")
    print(f"win_length:        400  (25ms)")
    print(f"num_mel_bins:      {fe.num_mel_bins}")
    print(f"stride:            {fe.stride}")
    print(f"feature_dim (out): {fe.feature_size * 2}  ({fe.feature_size} mel × {fe.stride} stride)")
    print(f"padding_value:     {fe.padding_value}")
    print(f"mel_filters shape: {fe.mel_filters.shape}")
    print(f"window length:     {len(fe.window)}")
    print()
    print("Saved models/mel_filters.npy")
    print("Saved models/mel_window.npy")

if __name__ == "__main__":
    main()
