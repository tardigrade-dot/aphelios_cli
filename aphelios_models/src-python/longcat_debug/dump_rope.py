import torch
import numpy as np
from pathlib import Path
from audiodit.modeling_audiodit import AudioDiTModel

model_dir = Path("/Volumes/sw/pretrained_models/LongCat-AudioDiT-1B")
model = AudioDiTModel.from_pretrained(model_dir)
# Force build
model.transformer.rotary_embed._build(10, torch.device('cpu'), torch.float32)
print("Py cos mean:", model.transformer.rotary_embed._cos.mean().item())
print("Py sin mean:", model.transformer.rotary_embed._sin.mean().item())
