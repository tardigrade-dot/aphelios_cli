import torch
import numpy as np

dim = 64
base = 100000.0

# Python
inv_freq_py = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
print(f"Py inv_freq[0]: {inv_freq_py[0].item()}")

# Rust equivalent (reconstructed)
idx = 0
inv_freq_rs = 1.0 / (base ** (idx / dim))
print(f"Rust inv_freq[0]: {inv_freq_rs}")
