import torch
import torch.nn.functional as F

def qwen_apply_rope(x, cos, sin):
    # cos, sin: [1, 1, seq, dim]
    # x: [1, heads, seq, head_dim]
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    return (x * cos) + (rotate_half(x) * sin)

# Candle's rope (roughly):
# x * cos + rotate_half(x) * sin
# Wait, candle_nn::rope is the SAME as Qwen rope if the cos/sin are prepared correctly!
# Let's check candle_nn::rotary_emb::rope
