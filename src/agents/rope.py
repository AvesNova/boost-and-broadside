"""
Rotary Position Embedding (RoPE) implementation for temporal attention.

RoPE applies rotation matrices to query and key vectors based on their position,
providing better extrapolation to longer sequences than learned embeddings.
"""

import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for transformer attention.

    Applies rotation matrices to query and key vectors based on position indices.
    The rotation is applied in the complex plane, rotating pairs of dimensions
    by angles that increase with position and decrease with dimension index.

    Attributes:
        dim: Dimension of the embedding (must be even).
        max_seq_len: Maximum sequence length to precompute.
        base: Base for computing rotation frequencies.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        """
        Initialize RoPE module.

        Args:
            dim: Embedding dimension (must be even).
            max_seq_len: Maximum sequence length for precomputation.
            base: Base for frequency computation (default: 10000.0).

        Raises:
            ValueError: If dim is not even.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute rotation frequencies
        # inv_freq shape: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin for all positions
        self._precompute_freqs(max_seq_len)

    def _precompute_freqs(self, seq_len: int) -> None:
        """
        Precompute cos and sin values for all positions up to seq_len.

        Args:
            seq_len: Sequence length to precompute.
        """
        # positions shape: (seq_len,)
        # Ensure positions is on the same device as inv_freq
        positions = torch.arange(
            seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )

        # freqs shape: (seq_len, dim // 2)
        freqs = torch.outer(positions, self.inv_freq)

        # emb shape: (seq_len, dim // 2, 2) for cos and sin
        # We'll store cos and sin separately for efficiency
        cos = freqs.cos()
        sin = freqs.sin()

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the dimensions of x.

        For RoPE, we rotate pairs of dimensions: [x0, x1, x2, x3, ...] becomes
        [-x1, x0, -x3, x2, ...].

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Rotated tensor of same shape.
        """
        # Split into two halves along last dimension
        x1, x2 = x.chunk(2, dim=-1)
        # Rotate: [-x2, x1]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, seq_len, dim).
            k: Key tensor of shape (batch, seq_len, dim).
            position_ids: Position indices of shape (seq_len,) or (batch, seq_len).

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs.
        """
        # Get sequence length
        seq_len = q.shape[1]
        max_pos = position_ids.max().item()

        # Extend cache if needed
        required_len = max(seq_len, max_pos + 1)
        if required_len > self.cos_cached.shape[0]:
            self._precompute_freqs(required_len)

        # Handle position_ids shape
        if position_ids.ndim == 1:
            # (seq_len,) -> use directly
            cos = self.cos_cached[position_ids]  # (seq_len, dim // 2)
            sin = self.sin_cached[position_ids]  # (seq_len, dim // 2)
        else:
            # (batch, seq_len) -> gather per batch
            cos = self.cos_cached[position_ids]  # (batch, seq_len, dim // 2)
            sin = self.sin_cached[position_ids]  # (batch, seq_len, dim // 2)

        # Expand cos and sin to match full dimension
        # We need to interleave: [cos0, cos0, cos1, cos1, ...] for pairs
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # (..., seq_len, dim)
        sin = torch.repeat_interleave(sin, 2, dim=-1)  # (..., seq_len, dim)

        # Add batch dimension if needed
        if cos.ndim == 2:
            cos = cos.unsqueeze(0)  # (1, seq_len, dim)
            sin = sin.unsqueeze(0)  # (1, seq_len, dim)

        # Apply rotation: q_rotated = q * cos + rotate_half(q) * sin
        q_rotated = q * cos + self._rotate_half(q) * sin
        k_rotated = k * cos + self._rotate_half(k) * sin

        return q_rotated, k_rotated


class Continuous2DRotaryEmbedding(nn.Module):
    """
    Continuous 2D Rotary Position Embedding for spatial attention.

    Encodes relative 2D positions (x, y) into the attention mechanism by rotating
    query and key vectors. This allows the model to naturally understand spatial
    relationships and distance without learned biases.
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        
        # Calculate frequencies for half the dimension (since we process X and Y separately)
        # We need dim/2 frequencies because we split dim into X-half and Y-half.
        # Each half needs (dim/4) frequencies for pairs.
        # We generate frequencies for the full dimension and will slice as needed.
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the dimensions of x."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rotary(
        self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply separate RoPE to X or Y component.

        Args:
            q: Query subset (Batch, N, Heads, Half_Dim)
            k: Key subset (Batch, N, Heads, Half_Dim)
            pos: Position coordinates (Batch, N, 1)

        Returns:
            Rotated q and k
        """
        half_dim = q.shape[-1]
        
        # Slicing inv_freq: We need (Half_Dim // 2) frequencies.
        # Our buffer inv_freq has (Dim // 2) frequencies.
        # Half_Dim is usually Dim // 2.
        # So we need (Dim // 4) frequencies.
        inv_freq = self.inv_freq[:half_dim // 2]  # (Half_Dim // 2)

        # Outer product: (Batch * N) x (Half_Dim // 2)
        flat_pos = pos.reshape(-1)
        freqs = torch.outer(flat_pos, inv_freq)

        # Reshape to (Batch, N, 1, Half_Dim // 2)
        batch_size, n_ships = pos.shape[:2]
        freqs = freqs.view(batch_size, n_ships, 1, half_dim // 2)

        # Create cos/sin and repeat for pairs -> (Batch, N, 1, Half_Dim)
        _cos = freqs.cos()
        _sin = freqs.sin()
        cos = torch.cat([_cos, _cos], dim=-1) 
        sin = torch.cat([_sin, _sin], dim=-1)

        # Apply rotation (broadcast over heads)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin

        return q_rot, k_rot

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D RoPE to query and key tensors.

        Args:
            q: Query tensor of shape (Batch, N, Heads, Dim)
            k: Key tensor of shape (Batch, N, Heads, Dim)
            coords: Normalized 2D coordinates of shape (Batch, N, 2) [x, y]

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        # 1. Split q, k into X and Y halves
        half_dim = self.dim // 2
        q_x, q_y = q[..., :half_dim], q[..., half_dim:]
        k_x, k_y = k[..., :half_dim], k[..., half_dim:]

        # 2. Extract coordinates and scale
        # Scale by 100.0 to move normalized (0-1) coords into effective frequency range
        x = coords[..., 0:1] * 100.0
        y = coords[..., 1:2] * 100.0

        # 3. Apply RoPE separately
        q_x, k_x = self._apply_rotary(q_x, k_x, x)
        q_y, k_y = self._apply_rotary(q_y, k_y, y)

        # 4. Concatenate back
        q_out = torch.cat([q_x, q_y], dim=-1)
        k_out = torch.cat([k_x, k_y], dim=-1)

        return q_out, k_out
