import numpy as np
import torch
import torch.nn.functional as F

def build_mask_from_lens(lengths, device):
    # lengths: [batch]
    # return [batch, max_len]
    max_len = lengths.max().item()
    batch = lengths.shape[0]
    seq = torch.arange(max_len, device=device).unsqueeze(0).expand(batch, -1)
    return seq < lengths.unsqueeze(1)

# Testing the mask logic for full-zero neg_text
lengths = torch.tensor([53])
device = 'cpu'
mask = build_mask_from_lens(lengths, device)
print(mask.shape)
print(mask.int())
