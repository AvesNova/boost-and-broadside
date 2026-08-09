"""Griffin temporal block and Yemong combined block for the Yemong policy backbone.

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
from boost_and_broadside.models.yemong.attention import GatedMLP, TransformerBlock

CONV_KERNEL: int = 4  # causal depthwise conv kernel size


def _parallel_scan(
    a: torch.Tensor,
    b: torch.Tensor,
    h0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hillis-Steele inclusive parallel scan for h_t = a_t·h_{t-1} + b_t.

    Associative operator: (a2,b2)∘(a1,b1) = (a2·a1, a2·b1+b2).
    After log2(T) rounds, position t holds the cumulative prefix product
    from 0..t, giving h_t = A_t·h0 + B_t for all t simultaneously.

    Args:
        a:   (B, T, D) per-timestep decay coefficients, values in (0, 1).
        b:   (B, T, D) per-timestep input contributions.
        h0:  (B, D) initial hidden state.

    Returns:
        outputs: (B, T, D) all hidden states h_0 … h_{T-1}.
        final_h: (B, D) final hidden state h_{T-1}.
    """
    B, T_real, D = a.shape

    # Pad T to the next power of two — identity elements (a=1, b=0) leave state unchanged.
    T_pad = 1 << (T_real - 1).bit_length()  # next power of two >= T_real
    if T_pad > T_real:
        pad = T_pad - T_real
        a = F.pad(a, (0, 0, 0, pad), value=1.0)
        b = F.pad(b, (0, 0, 0, pad), value=0.0)

    # Hillis-Steele inclusive scan: log2(T_pad) rounds.
    # b must be updated before a because b uses the pre-round value of a.
    for step in range(T_pad.bit_length() - 1):
        stride = 1 << step
        a_lag = F.pad(a[:, : T_pad - stride], (0, 0, stride, 0), value=1.0)
        b_lag = F.pad(b[:, : T_pad - stride], (0, 0, stride, 0), value=0.0)
        b = a * b_lag + b
        a = a * a_lag

    outputs = (a * h0.unsqueeze(1) + b)[:, :T_real]
    return outputs, outputs[:, -1]


def _causal_conv_tap_validity(done_mask: torch.Tensor) -> torch.Tensor:
    """Per-output, per-lag mask marking taps that stay inside the current episode.

    Args:
        done_mask: (B_seq, T) bool — True at t means the episode ended at t, so
            t+1 begins a fresh one (same convention as the RG-LRU reset).

    Returns:
        (B_seq, T, CONV_KERNEL) bool — entry [b, t, j] is True when the input at
        t-j belongs to the same episode as t. Lag 0 is always valid.
    """
    starts_new = done_mask.roll(1, dims=1)
    starts_new[:, 0] = False  # the segment start continues the stored buffer
    no_boundary = ~starts_new  # (B_seq, T)

    valids = [torch.ones_like(no_boundary)]  # lag 0
    running = torch.ones_like(no_boundary)
    for lag in range(1, CONV_KERNEL):
        shift = lag - 1
        shifted = (
            no_boundary
            if shift == 0
            else F.pad(no_boundary[:, :-shift], (shift, 0), value=True)
        )
        running = running & shifted
        valids.append(running)
    return torch.stack(valids, dim=-1)  # (B_seq, T, CONV_KERNEL)


def _causal_depthwise_conv(
    padded: torch.Tensor,  # (B_seq, T + CONV_KERNEL - 1, D)
    conv: nn.Conv1d,
    num_steps: int,
    done_mask: torch.Tensor | None,  # (B_seq, T) bool
) -> torch.Tensor:
    """Depthwise causal conv over a left-padded sequence, cut at episode boundaries.

    Written as an explicit sum of CONV_KERNEL shifted taps rather than a
    ``conv1d`` call, because the boundary mask depends on *both* the output step
    and the tap's lag — a single input feeds CONV_KERNEL outputs and must be
    dropped for some of them and kept for others, which no input-side mask can
    express. The arithmetic is identical (a depthwise K-tap conv is K
    multiply-adds) and it drops the transpose pair the conv1d path needed.

    Without the mask, rollout and update disagree: ``reset_hidden_for_envs``
    zeroes the whole packed hidden — conv buffer included — at an episode
    boundary during rollout, so post-boundary steps convolve against zeros. The
    sequence path saw the previous episode's inputs for CONV_KERNEL-1 steps
    after every boundary, which is exactly where value estimates matter most.

    Args:
        padded: Stored conv buffer concatenated with this segment's inputs.
        conv: The depthwise Conv1d supplying weights and bias.
        num_steps: T — outputs to produce.
        done_mask: Episode-end flags, or None to convolve straight through
            (the rollout path, where the buffer is reset externally).

    Returns:
        (B_seq, T, D)
    """
    weight = conv.weight.squeeze(1)  # (D, CONV_KERNEL) — depthwise
    validity = _causal_conv_tap_validity(done_mask) if done_mask is not None else None

    out = torch.zeros_like(padded[:, :num_steps, :])
    for k in range(CONV_KERNEL):
        # conv1d is cross-correlation: out[t] = Σ_k w[k] · padded[t + k], so the
        # tap at weight index k reads lag CONV_KERNEL-1-k.
        tap = padded[:, k : k + num_steps, :] * weight[:, k]
        if validity is not None:
            tap = tap * validity[..., CONV_KERNEL - 1 - k].unsqueeze(-1)
        out = out + tap
    if conv.bias is not None:
        out = out + conv.bias
    return out


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
        """One RG-LRU step. Sequential reference; kept for testing.

        Args:
            x: (B_seq, D) input at current timestep.
            h: (B_seq, D) previous hidden state.

        Returns:
            new_h: (B_seq, D) updated hidden state (also the output).
        """
        r = torch.sigmoid(self.linear_a(x))  # (B_seq, D)
        i = torch.sigmoid(self.linear_x(x))  # (B_seq, D)
        a = torch.sigmoid(self.log_lambda) ** (self.c * r)  # (B_seq, D)
        # Factor (1-a)(1+a) avoids catastrophic cancellation in bfloat16 when a→1.
        # Epsilon prevents sqrt gradient explosion at the boundary.
        gate = torch.sqrt(torch.clamp((1.0 - a) * (1.0 + a), min=1e-6))
        new_h = a * h + gate * (i * x)
        return new_h

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        h0: torch.Tensor,
        done_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a full sequence via parallel scan.

        Args:
            x_seq:     (B_seq, T, D) input sequence.
            h0:        (B_seq, D) initial hidden state.
            done_mask: (B_seq, T) bool — True at step t means the episode ended
                       at t, so step t+1 starts a fresh episode (a[t+1] is zeroed).

        Returns:
            outputs: (B_seq, T, D) per-step hidden states.
            h:       (B_seq, D) final hidden state.
        """
        B, T, D = x_seq.shape
        x_flat = x_seq.reshape(B * T, D)
        r = torch.sigmoid(self.linear_a(x_flat)).reshape(B, T, D)
        i = torch.sigmoid(self.linear_x(x_flat)).reshape(B, T, D)
        a = torch.sigmoid(self.log_lambda) ** (self.c * r)
        gate = torch.sqrt(torch.clamp((1.0 - a) * (1.0 + a), min=1e-6))
        b = gate * (i * x_seq)
        if done_mask is not None:
            # Episode ended at t → step t+1 must not inherit h[t].
            # Shift the mask right by one so a[t+1] is zeroed.
            reset_a = done_mask.roll(1, dims=1)
            reset_a[:, 0] = False
            a = a.masked_fill(reset_a.unsqueeze(-1), 0.0)
        return _parallel_scan(a, b, h0)


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
        self.linear1 = nn.Linear(d_model, d_model, bias=False)  # branch1 input
        self.conv = nn.Conv1d(  # depthwise, no padding
            d_model, d_model, kernel_size=CONV_KERNEL, groups=d_model, bias=True
        )
        self.rg_lru = RGLRU(d_model)
        self.linear2 = nn.Linear(d_model, d_model, bias=False)  # branch2 (gate) input
        self.linear_out = nn.Linear(d_model, d_model, bias=False)  # combine branches
        self.norm2 = nn.RMSNorm(d_model)
        self.gated_mlp = GatedMLP(d_model)

    def forward_sequence(
        self,
        x_seq: torch.Tensor,  # (B_seq, T, D)
        h0: torch.Tensor,  # (B_seq, D)
        conv_buf: torch.Tensor,  # (B_seq, CONV_KERNEL-1, D)
        done_mask: torch.Tensor | None = None,  # (B_seq, T) bool
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
        normed = self.norm1(x_seq)

        b1 = self.linear1(normed)  # (B_seq, T, D)

        # Causal conv: prepend stored buffer instead of zeros.
        # padded: (B_seq, T+CONV_KERNEL-1, D) → (B_seq, T, D)
        padded = torch.cat([conv_buf, b1], dim=1)
        b1_conv = _causal_depthwise_conv(padded, self.conv, x_seq.shape[1], done_mask)
        new_conv_buf = padded[:, -(CONV_KERNEL - 1) :, :]  # (B_seq, K-1, D)

        b1_out, new_h = self.rg_lru.forward_sequence(b1_conv, h0, done_mask)  # (B_seq, T, D)

        b2 = F.gelu(self.linear2(normed))  # (B_seq, T, D)
        recurrent_out = self.linear_out(b1_out * b2)  # (B_seq, T, D)

        x1 = x_seq + recurrent_out  # 1st residual
        x2 = x1 + self.gated_mlp(self.norm2(x1))  # 2nd residual
        return x2, new_h, new_conv_buf

    def forward_nonrecurrent(self, x: torch.Tensor, sub: nn.Linear) -> torch.Tensor:
        """Same block with the temporal operator replaced by a linear map.

        Used for entities whose state is static within an episode (refractive
        fields): the causal conv and RG-LRU are dropped, while ``norm1``,
        ``linear1``, ``linear2``, ``linear_out``, ``norm2``, and ``gated_mlp``
        stay shared with the recurrent path. Sharing is the point — it keeps
        both entity types in one representation, so the next spatial layer's
        single ``W_qkv`` does not have to reconcile two diverged token spaces.

        ``sub`` supplies the one degree of freedom the shared weights cannot:
        a type-specific linear map. It is identity-initialised (see
        ``YemongBlock``), not zero-initialised — zeroing it would null the
        entire recurrent branch and leave these tokens with only ``gated_mlp``.

        Shape-agnostic in the leading dimensions: there is no scan, so the
        caller need not fold entities into a sequence layout.

        Args:
            x:   (..., D) entity embeddings.
            sub: Linear(D, D) standing in for the temporal operator.

        Returns:
            (..., D) updated embeddings. No hidden state, no conv buffer.
        """
        normed = self.norm1(x)
        b1 = sub(self.linear1(normed))  # replaces causal conv -> RG-LRU
        b2 = F.gelu(self.linear2(normed))
        x1 = x + self.linear_out(b1 * b2)  # 1st residual
        return x1 + self.gated_mlp(self.norm2(x1))  # 2nd residual


class YemongBlock(nn.Module):
    """Yemong layer: spatial transformer sublayers followed by temporal sublayers.

    Entities attend to each other in the spatial sublayers (cross-entity, within
    timestep). Each entity's embedding then evolves through the temporal sublayers
    (per-entity, across time).

    Every block in the trunk has this same structure; ``ModelConfig`` sets the two
    sublayer counts. The recurrent state is one slot per temporal sublayer, so a
    block consumes ``n_temporal_per_block`` slots of the policy's hidden tensor.

    Args:
        model_config: Supplies d_model, n_heads, and the sublayer counts.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        # Bullet reads are counted from the first spatial sublayer: a ship can only
        # reason about fire aimed at *another* ship if that ship's own bullet read
        # happens before a later entity-to-entity layer.
        self.spatial = nn.ModuleList(
            [
                TransformerBlock(
                    model_config,
                    reads_bullets=i < model_config.n_bullet_cross_per_block,
                )
                for i in range(model_config.n_spatial_per_block)
            ]
        )
        self.temporal = nn.ModuleList(
            [
                GriffinTemporalBlock(model_config.d_model)
                for _ in range(model_config.n_temporal_per_block)
            ]
        )
        # Type-specific linear standing in for the temporal operator on
        # non-recurrent (field) tokens; see GriffinTemporalBlock.forward_nonrecurrent.
        # Allocated even when a profile has no fields so one checkpoint loads into
        # both the zero-field and multi-field profiles.
        self.field_sub = nn.ModuleList(
            [
                nn.Linear(model_config.d_model, model_config.d_model, bias=False)
                for _ in range(model_config.n_temporal_per_block)
            ]
        )
        for sub in self.field_sub:
            # Identity, not zero: b1_out feeds a multiplicative gate, so zeroing it
            # would erase the whole recurrent branch for field tokens.
            nn.init.eye_(sub.weight)

    @property
    def n_temporal(self) -> int:
        """Hidden-state slots this block consumes."""

        return len(self.temporal)

    def step(
        self,
        x: torch.Tensor,  # (B, N+M, D)
        alive: torch.Tensor,  # (B, N+M) bool
        h: torch.Tensor,  # (n_temporal, B*N, D)
        conv_buf: torch.Tensor,  # (n_temporal, B*N, CONV_KERNEL-1, D)
        num_recurrent: int | None = None,
        bullets: torch.Tensor | None = None,  # (B, NB, D)
        bullet_mask: torch.Tensor | None = None,  # (B, NB) bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step forward for rollout inference.

        Args:
            num_recurrent: leading token count on the recurrent path (ships).
                Trailing tokens (fields) take the non-recurrent path instead.
                None means every token is recurrent.

        Returns:
            x:           (B, N+M, D) updated embeddings.
            new_h:       (n_temporal, B*N, D) updated RG-LRU hidden states.
            new_conv_buf:(n_temporal, B*N, CONV_KERNEL-1, D) updated conv buffers.
        """
        B, NM, D = x.shape
        n_rec = NM if num_recurrent is None else num_recurrent
        for spatial in self.spatial:
            x = spatial(x, alive, bullets, bullet_mask)  # (B, N+M, D)

        new_hs: list[torch.Tensor] = []
        new_cbs: list[torch.Tensor] = []
        for j, temporal in enumerate(self.temporal):
            ships, fields = x[:, :n_rec, :], x[:, n_rec:, :]
            out, new_h, new_cb = temporal.forward_sequence(
                ships.reshape(B * n_rec, 1, D), h[j], conv_buf[j]
            )
            ships = out.squeeze(1).reshape(B, n_rec, D)
            if fields.shape[1]:
                fields = temporal.forward_nonrecurrent(fields, self.field_sub[j])
            x = torch.cat([ships, fields], dim=1)
            new_hs.append(new_h)
            new_cbs.append(new_cb)

        return x, _stack_like(new_hs, h), _stack_like(new_cbs, conv_buf)

    def sequence(
        self,
        x: torch.Tensor,  # (T, B, N+M, D)
        alive_mask: torch.Tensor,  # (T, B, N+M) bool
        h0: torch.Tensor,  # (n_temporal, B*N, D)
        conv_buf0: torch.Tensor,  # (n_temporal, B*N, CONV_KERNEL-1, D)
        done_mask: torch.Tensor | None = None,  # (T, B) bool
        num_recurrent: int | None = None,
        bullets: torch.Tensor | None = None,  # (T*B, NB, D)
        bullet_mask: torch.Tensor | None = None,  # (T*B, NB) bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full-sequence forward for PPO re-evaluation.

        Args:
            num_recurrent: leading token count on the recurrent path (ships).
                None means every token is recurrent.

        Returns:
            (T, B, N+M, D) updated embeddings, final RG-LRU h, final conv buf.
        """
        T, B, NM, D = x.shape
        n_rec = NM if num_recurrent is None else num_recurrent

        # Spatial: fold T into batch for parallel cross-entity attention
        for spatial in self.spatial:
            x = spatial(
                x.reshape(T * B, NM, D),
                alive_mask.reshape(T * B, NM),
                bullets,
                bullet_mask,
            ).reshape(T, B, NM, D)

        done_mask_bn = (
            done_mask.permute(1, 0).repeat_interleave(n_rec, dim=0)  # (B*N, T)
            if done_mask is not None
            else None
        )

        new_hs: list[torch.Tensor] = []
        new_cbs: list[torch.Tensor] = []
        for j, temporal in enumerate(self.temporal):
            ships, fields = x[:, :, :n_rec, :], x[:, :, n_rec:, :]
            # Temporal: fold B*N into batch, sequence over T per ship
            x_seq = ships.permute(1, 2, 0, 3).reshape(B * n_rec, T, D)  # (B*N, T, D)
            out, new_h, new_cb = temporal.forward_sequence(x_seq, h0[j], conv_buf0[j], done_mask_bn)
            ships = out.reshape(B, n_rec, T, D).permute(2, 0, 1, 3)  # (T, B, N, D)
            if fields.shape[2]:
                # No scan, so fields need no sequence layout — apply in place.
                fields = temporal.forward_nonrecurrent(fields, self.field_sub[j])
            x = torch.cat([ships, fields], dim=2)
            new_hs.append(new_h)
            new_cbs.append(new_cb)

        return x, _stack_like(new_hs, h0), _stack_like(new_cbs, conv_buf0)


def _stack_like(parts: list[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    """Stack per-sublayer states, falling back to an empty slice when there are none.

    ``n_temporal_per_block=0`` is a legal (purely spatial) configuration, and
    ``torch.stack`` rejects empty lists — so borrow the reference's zero-length
    shape instead of special-casing every caller.
    """

    if not parts:
        return reference[:0]
    return torch.stack(parts, dim=0)
