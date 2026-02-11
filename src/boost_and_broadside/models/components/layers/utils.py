import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Cast weight to x.dtype for fused kernel support (especially in BF16)
        if hasattr(F, "rms_norm"):
             return F.rms_norm(x, (x.size(-1),), self.weight.to(x.dtype), self.eps)
        else:
             # Manual fallback
             dims = x.shape[-1]
             var = x.pow(2).mean(-1, keepdim=True)
             x_normed = x * torch.rsqrt(var + self.eps)
             return self.weight.to(x.dtype) * x_normed

class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 128, expand: int = 2, layer_idx: int = 0):
        super().__init__()
        if Mamba2 is None:
            self.mamba = nn.Identity()
        else:
            try:
                self.mamba = Mamba2(d_model=d_model, d_state=d_state, expand=expand, layer_idx=layer_idx)
            except:
                self.mamba = nn.Identity()
        
    def forward(self, x, seq_idx=None, inference_params=None):
         if Mamba2 is None or x.device.type == 'cpu':
             return x
         try:
             return self.mamba(x, seq_idx=seq_idx, inference_params=inference_params)
         except Exception:
             return self.mamba(x)
