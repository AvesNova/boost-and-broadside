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
        # Flatten to (Total, Dim) to unify rank for compilation
        # Cast weight to x.dtype for fused kernel support (especially in BF16)
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        
        if hasattr(F, "rms_norm"):
             out = F.rms_norm(x_flat, (orig_shape[-1],), self.weight.to(x.dtype), self.eps)
        else:
             # Manual fallback
             var = x_flat.pow(2).mean(-1, keepdim=True)
             x_normed = x_flat * torch.rsqrt(var + self.eps)
             out = self.weight.to(x.dtype) * x_normed
             
        return out.view(*orig_shape)

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

    def step(self, x, conv_state, ssm_state):
        if hasattr(self.mamba, 'step'):
            return self.mamba.step(x, conv_state, ssm_state)
        return x, conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        if hasattr(self.mamba, 'allocate_inference_cache'):
             return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
        return (None, None)
