#!/usr/bin/env python3
"""Analyze FunASR Nano model structure and expected behavior"""

import sys
sys.path.insert(0, '/Volumes/sw/uv_envs/funasr/lib/python3.12/site-packages')

from funasr import AutoModel
import torch

model_dir = "/Volumes/sw/pretrained_models/Fun-ASR-Nano-2512"

# Load model without inference
print("Loading model...")
model = AutoModel(model=model_dir, device="cpu")

# Check model structure
print("\n=== Model Structure ===")
print(f"Model type: {type(model)}")
print(f"Model attributes: {dir(model)}")

# Check if there's a way to get CTC probs aligned with text
print("\n=== Checking inference output ===")
wav_path = "/Volumes/sw/video/qinsheng.wav"

# Run inference with detailed output
res = model.generate(input=[wav_path], cache={}, batch_size_s=0)
print(f"Result keys: {res[0].keys()}")
print(f"Text: {res[0]['text']}")
if 'timestamp' in res[0]:
    print(f"Timestamps: {res[0]['timestamp'][:5]}...")  # First 5
if 'alignment' in res[0]:
    print(f"Alignment: {res[0]['alignment'][:5]}...")  # First 5
