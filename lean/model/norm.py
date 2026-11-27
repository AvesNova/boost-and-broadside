"""RMSNorm whose output may return to the autocast dtype (LLaMA's rule): the reduction in
float32 (it sums many small squares), then, with `low_precision_output` and autocast on, the
output cast to the autocast dtype (bf16), which the next Linear reads anyway. Without it, or
outside autocast, the output is float32 as torch's RMSNorm returns under autocast. The
residual stream itself stays float32 (the norms that start it keep a float32 output)."""

import torch
from torch import Tensor, nn


class Norm(nn.RMSNorm):
    def __init__(self, width: int, eps: float | None, low_precision_output: bool):
        super().__init__(width, eps=eps)
        self.low_precision_output = low_precision_output

    def forward(self, x: Tensor) -> Tensor:
        normed = super().forward(x)
        device = x.device.type
        if self.low_precision_output and torch.is_autocast_enabled(device):
            return normed.to(torch.get_autocast_dtype(device))
        return normed
