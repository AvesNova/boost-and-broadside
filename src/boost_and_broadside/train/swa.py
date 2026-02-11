import torch.nn as nn
import logging
from copy import deepcopy

log = logging.getLogger(__name__)

class SWAModule:
    """
    Stochastic Weight Averaging (SWA) Module.
    Maintains a shadow copy of the model on CPU and performs cumulative moving average updates.
    """
    def __init__(self, model: nn.Module):
        """
        Initialize SWA module.
        
        Args:
            model: The base model to average. The averaged model will be created 
                   as a CPU copy of this model.
        """
        self.n_averaged = 0
        # Create a deep copy of the model on CPU to save GPU memory
        self.averaged_model = deepcopy(model).to('cpu')
        
    def update_parameters(self, model: nn.Module):
        """
        Update SWA parameters with the current model's parameters.
        Formula: w_swa = (w_swa * n + w_k) / (n + 1)
        
        Args:
            model: The current active model (can be on GPU).
        """
        self.n_averaged += 1
        alpha = 1.0 / self.n_averaged
        
        log.info(f"Updating SWA statistics (n={self.n_averaged})")
        
        # Iterate over parameters and buffers (like running_mean/var for BN if we had it)
        # We process manually to ensure CPU operation
        for swa_param, param in zip(self.averaged_model.parameters(), model.parameters()):
            if param.requires_grad:
                # Move param to CPU non-blocking
                param_cpu = param.detach().to('cpu', non_blocking=True)
                
                # In-place update: swa = swa * (1 - alpha) + param * alpha
                # which is equivalent to cumulative average
                swa_param.data.mul_(1.0 - alpha).add_(param_cpu, alpha=alpha)
                
        # Handle buffers (e.g. for LayerNorm tracking if enabled, though LN usually doesn't have momentum buffers)
        # If we had BatchNorm, we would need to handle running stats.
        # But LayerNorm usually doesn't need this. We copy buffers just in case.
        for swa_buffer, buffer in zip(self.averaged_model.buffers(), model.buffers()):
             buffer_cpu = buffer.detach().to('cpu', non_blocking=True)
             if swa_buffer.shape != buffer_cpu.shape:
                 log.debug(f"SWA: Resizing buffer from {swa_buffer.shape} to {buffer_cpu.shape}")
                 # Use set_ to point to the new storage (effectively resizing/replacing)
                 swa_buffer.set_(buffer_cpu)
             else:
                 swa_buffer.data.copy_(buffer_cpu)

    def state_dict(self):
        return self.averaged_model.state_dict()
        
    def load_state_dict(self, state_dict):
        self.averaged_model.load_state_dict(state_dict)
