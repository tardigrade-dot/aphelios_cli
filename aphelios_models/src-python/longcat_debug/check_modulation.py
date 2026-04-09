import torch
import torch.nn.functional as F

x = torch.randn(1, 10, 1536)
x_norm = F.layer_norm(x.float(), (1536,), eps=1e-6)

scale = torch.randn(1, 1536)
shift = torch.randn(1, 1536)

# Python modulate logic: x * (1 + scale) + shift
# Wait, check Python modulate implementation carefully
# 281:def _modulate(x, scale, shift, eps=1e-6):
# 283:    x = F.layer_norm(x.float(), (x.shape[-1],), eps=eps).type_as(x)
# 285:        return x * (1 + scale[:, None]) + shift[:, None]
