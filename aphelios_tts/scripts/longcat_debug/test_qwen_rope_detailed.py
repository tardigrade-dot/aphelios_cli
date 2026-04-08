import torch

dim = 4
seq = 2
heads = 1
head_dim = 4
x = torch.randn(1, heads, seq, head_dim)

# Qwen params
inv_freq = 1.0 / (100000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
t = torch.arange(seq).float()
freqs = torch.outer(t, inv_freq)
emb = torch.cat((freqs, freqs), dim=-1)
cos = emb.cos().view(1, 1, seq, head_dim)
sin = emb.sin().view(1, 1, seq, head_dim)

def qwen_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos[..., :head_dim//2] - x2 * sin[..., :head_dim//2],
                      x2 * cos[..., head_dim//2:] + x1 * sin[..., head_dim//2:]], dim=-1)

# This is a slightly different implementation than rotate_half.
# Let's see if this matches.
