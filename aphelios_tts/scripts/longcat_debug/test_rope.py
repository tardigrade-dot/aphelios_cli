import torch

dim = 64
max_seq_len = 10
base = 100000.0

# Python original: emb = torch.cat((freqs, freqs), dim=-1).cos()...
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
t = torch.arange(max_seq_len, dtype=torch.float32)
freqs = torch.outer(t, inv_freq)
# Python does: emb = torch.cat((freqs, freqs), dim=-1)
# Py: emb.cos() gives [cos(freqs), cos(freqs)]
# Rust: cos: freqs.cos(), sin: freqs.sin()

py_cos = torch.cat((freqs, freqs), dim=-1).cos()
rs_cos = freqs.cos()
print(f"Py cos shape: {py_cos.shape}")
print(f"Rust cos shape: {rs_cos.shape}")
print(f"First element match: {torch.allclose(py_cos[:, :dim//2], rs_cos)}")
print(f"Second element match: {torch.allclose(py_cos[:, dim//2:], rs_cos)}")
