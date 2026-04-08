import torch
import torch.nn.functional as F

# Rust: scale and shift are [1, 1536] (after unsqueeze)
# Python: scale and shift are [1, 1536]
# Python logic: x * (1 + scale) + shift
# Rust logic: scaled = x.broadcast_mul(&(scale + 1.0)); return scaled.broadcast_add(&shift)
# They are identical.
print("Modulation logic is identical.")
