import torch
from pathlib import Path
from audiodit.modeling_audiodit import AudioDiTModel
model = AudioDiTModel.from_pretrained("/Volumes/sw/pretrained_models/LongCat-AudioDiT-1B")
rope = model.transformer.rotary_embed
rope._build(2048, torch.device('cpu'), torch.float32)
print(f"Py cos mean: {rope._cos.mean().item()}")
