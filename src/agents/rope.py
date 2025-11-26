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
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        
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
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        position_ids: torch.Tensor
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
        
        # Extend cache if needed
        if seq_len > self.cos_cached.shape[0]:
            self._precompute_freqs(seq_len)
        
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
