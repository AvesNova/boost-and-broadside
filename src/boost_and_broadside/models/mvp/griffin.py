"""Griffin temporal block and Yemong combined block for the MVP policy backbone.

YemongBlock = SpatialTransformerBlock (MHA + GatedMLP) + GriffinTemporalBlock (RG-LRU + GatedMLP).

The RG-LRU (Real-Gated Linear Recurrent Unit) provides per-ship temporal memory
with learnable decay rates, replacing the GRU in the original backbone.

Hidden state shape: (n_layers, B*N, CONV_KERNEL * D) packed as:
  hidden[:, :, :D]   — RG-LRU recurrent state per layer
  hidden[:, :, D:]   — causal conv buffer flattened as (CONV_KERNEL-1) * D
The conv buffer stores the last (kernel-1) inputs to linear1, so that rollout
(T=1) and PPO re-evaluation (T=128) use identical causal context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from boost_and_broadside.config import ModelConfig
from boost_and_broadside.models.mvp.attention import GatedMLP, TransformerBlock

CONV_KERNEL: int = 4  # causal depthwise conv kernel size


class RGLRU(nn.Module):
    """Real-Gated Linear Recurrent Unit from the Griffin paper.

    Per-element decay rates aₜ = σ(Λ)^(c·rₜ) are controlled by learnable
    log-eigenvalues Λ and an input-dependent recurrence gate rₜ. An input
    gate iₜ scales the new information before mixing with the hidden state.

        rₜ = σ(Wₐxₜ + bₐ)
        iₜ = σ(Wₓxₜ + bₓ)
        aₜ = σ(Λ)^(c·rₜ)            c=8
        hₜ = aₜ⊙hₜ₋₁ + √(1−aₜ²)⊙(iₜ⊙xₜ)

    Args:
        d_model: State and input dimension D.
        c:       Exponent scaling constant (default 8, as in Griffin paper).
    """

    def __init__(self, d_model: int, c: float = 8.0) -> None:
        super().__init__()
        self.c = c
        # σ(log_lambda) ∈ (0,1) gives per-element decay rates.
        # linspace(0, 4, D) → σ values from ~0.5 (fast) to ~0.98 (slow).
        self.log_lambda = nn.Parameter(torch.linspace(0.0, 4.0, d_model))
        self.linear_a = nn.Linear(d_model, d_model, bias=True)  # r gate
        self.linear_x = nn.Linear(d_model, d_model, bias=True)  # i gate

    def _step(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """One RG-LRU step.

        Args:
            x: (B_seq, D) input at current timestep.
            h: (B_seq, D) previous hidden state.

        Returns:
            new_h: (B_seq, D) updated hidden state (also the output).
        """
        r = torch.sigmoid(self.linear_a(x))                 # (B_seq, D)
        i = torch.sigmoid(self.linear_x(x))                 # (B_seq, D)
        a = torch.sigmoid(self.log_lambda) ** (self.c * r)  # (B_seq, D)
        # Factor (1-a)(1+a) avoids catastrophic cancellation in bfloat16 when a→1.
        # Epsilon prevents sqrt gradient explosion at the boundary.
        gate = torch.sqrt(torch.clamp((1.0 - a) * (1.0 + a), min=1e-6))
        new_h = a * h + gate * (i * x)
        return new_h

    def forward_sequence(
        self, x_seq: torch.Tensor, h0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a full sequence sequentially (fused by torch.compile).

        Args:
            x_seq: (B_seq, T, D) input sequence.
            h0:    (B_seq, D) initial hidden state.

        Returns:
            outputs: (B_seq, T, D) per-step hidden states.
            h:       (B_seq, D) final hidden state.
        """
        T = x_seq.shape[1]
        h = h0
        outputs = []
        for t in range(T):
            h = self._step(x_seq[:, t], h)
            outputs.append(h)
        return torch.stack(outputs, dim=1), h


class GriffinTemporalBlock(nn.Module):
    """Griffin temporal block applied independently per ship across time.

    Matches the diagram:
        norm → (linear₁ → causal_conv → RG-LRU) × GeLU(linear₂) → linear_out
             → 1st residual → GatedMLP(norm) → 2nd residual

    The causal conv uses a stored buffer (the last kernel-1 linear₁ outputs)
    as left-padding, making rollout (T=1) and training (T=128) identical.

    Args:
        d_model: Embedding dimension D.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model, bias=False)     # branch1 input
        self.conv = nn.Conv1d(                                       # depthwise, no padding
            d_model, d_model, kernel_size=CONV_KERNEL, groups=d_model, bias=True
        )
        self.rg_lru = RGLRU(d_model)
        self.linear2 = nn.Linear(d_model, d_model, bias=False)     # branch2 (gate) input
        self.linear_out = nn.Linear(d_model, d_model, bias=False)  # combine branches
        self.norm2 = nn.RMSNorm(d_model)
        self.gated_mlp = GatedMLP(d_model)

    def forward_sequence(
        self,
        x_seq: torch.Tensor,    # (B_seq, T, D)
        h0: torch.Tensor,       # (B_seq, D)
        conv_buf: torch.Tensor, # (B_seq, CONV_KERNEL-1, D)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply temporal block over a sequence.

        Args:
            x_seq:    (B_seq, T, D) input (ships as batch, time as sequence).
            h0:       (B_seq, D) initial RG-LRU hidden state.
            conv_buf: (B_seq, CONV_KERNEL-1, D) stored linear1 outputs from
                      the end of the previous sequence — used as causal left-padding.

        Returns:
            output:       (B_seq, T, D).
            new_h:        (B_seq, D) final RG-LRU hidden state.
            new_conv_buf: (B_seq, CONV_KERNEL-1, D) updated conv buffer.
        """
        T = x_seq.shape[1]
        normed = self.norm1(x_seq)

        b1 = self.linear1(normed)                                  # (B_seq, T, D)

        # Causal conv: prepend stored buffer instead of zeros.
        # padded: (B_seq, T+CONV_KERNEL-1, D) → conv (no padding) → (B_seq, T, D)
        padded = torch.cat([conv_buf, b1], dim=1)
        b1_conv = self.conv(padded.transpose(1, 2)).transpose(1, 2)  # (B_seq, T, D)
        new_conv_buf = padded[:, -(CONV_KERNEL - 1):, :]             # (B_seq, K-1, D)

        b1_out, new_h = self.rg_lru.forward_sequence(b1_conv, h0)   # (B_seq, T, D)

        b2 = F.gelu(self.linear2(normed))                           # (B_seq, T, D)
        recurrent_out = self.linear_out(b1_out * b2)                # (B_seq, T, D)

        x1 = x_seq + recurrent_out                                  # 1st residual
        x2 = x1 + self.gated_mlp(self.norm2(x1))                   # 2nd residual
        return x2, new_h, new_conv_buf


class YemongBlock(nn.Module):
    """Yemong layer: SpatialTransformerBlock followed by GriffinTemporalBlock.

    Ships attend to each other in the spatial block (cross-ship, within timestep).
    Each ship's embedding then evolves through the temporal block (per-ship, across time).

    Args:
        model_config: Supplies d_model, n_heads.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.spatial = TransformerBlock(model_config)
        self.temporal = GriffinTemporalBlock(model_config.d_model)

    def step(
        self,
        x: torch.Tensor,        # (B, N, D)
        alive: torch.Tensor,    # (B, N) bool
        h: torch.Tensor,        # (B*N, D)
        conv_buf: torch.Tensor, # (B*N, CONV_KERNEL-1, D)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step forward for rollout inference.

        Returns:
            x:           (B, N, D) updated embeddings.
            new_h:       (B*N, D) updated RG-LRU hidden state.
            new_conv_buf:(B*N, CONV_KERNEL-1, D) updated conv buffer.
        """
        B, N, D = x.shape
        x = self.spatial(x, alive)                                     # (B, N, D)
        x_flat = x.reshape(B * N, 1, D)                               # (B*N, 1, D)
        out, new_h, new_cb = self.temporal.forward_sequence(x_flat, h, conv_buf)
        return out.squeeze(1).reshape(B, N, D), new_h, new_cb

    def sequence(
        self,
        x: torch.Tensor,           # (T, B, N, D)
        alive_mask: torch.Tensor,  # (T, B, N) bool
        h0: torch.Tensor,          # (B*N, D)
        conv_buf0: torch.Tensor,   # (B*N, CONV_KERNEL-1, D)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full-sequence forward for PPO re-evaluation.

        Returns:
            (T, B, N, D) updated embeddings, final RG-LRU h, final conv buf.
        """
        T, B, N, D = x.shape

        # Spatial: fold T into batch for parallel cross-ship attention
        x = self.spatial(
            x.reshape(T * B, N, D), alive_mask.reshape(T * B, N)
        ).reshape(T, B, N, D)

        # Temporal: fold B*N into batch, sequence over T per ship
        x_seq = x.permute(1, 2, 0, 3).reshape(B * N, T, D)           # (B*N, T, D)
        out, new_h, new_cb = self.temporal.forward_sequence(x_seq, h0, conv_buf0)
        x = out.reshape(B, N, T, D).permute(2, 0, 1, 3)              # (T, B, N, D)
        return x, new_h, new_cb
