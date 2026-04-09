import sys
from pathlib import Path
sys.path.insert(0, "/Users/larry/coderesp/LongCat-AudioDiT")

import torch
from einops import rearrange
from audiodit.modeling_audiodit import AudioDiTModel

device = 'cpu'
model_dir = Path("/Volumes/sw/pretrained_models/LongCat-AudioDiT-1B")
model = AudioDiTModel.from_pretrained(model_dir).to(device)

t = torch.randn(1, 1536)
adaln_global_out = model.transformer.adaln_global_mlp(t)
for i, block in enumerate(model.transformer.blocks):
    adaln_out = adaln_global_out + rearrange(block.adaln_scale_shift, "f -> 1 f")
    _, scale_sa, _, _, _, _ = torch.chunk(adaln_out, 6, dim=-1)
    print(f"Python Block {i} scale_sa mean: {scale_sa.mean().item()}")
