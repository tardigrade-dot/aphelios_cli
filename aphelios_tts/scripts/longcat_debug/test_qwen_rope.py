import torch

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def qwen_apply_rope(x, cos, sin):
    return (x * cos) + (_rotate_half(x) * sin)

dim = 4
x = torch.randn(1, 1, 1, dim)
# cos/sin shape?
# Python: _cos = emb.cos() where emb = cat((freqs, freqs))
# Python forward:
# return (x * cos) + (_rotate_half(x) * sin)
