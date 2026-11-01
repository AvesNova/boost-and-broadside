"""The linear recurrence h_t = a_t h_{t-1} + b_t over a leading time axis.

On CUDA (float32) it runs as one Triton kernel per direction: each program owns a block of
channels and walks time, so a (T, C) problem reads a and b once and writes h once. The
backward is the reverse recurrence g_t = dL/dh_t + a_{t+1} g_{t+1}, with db_t = g_t,
da_t = g_t h_{t-1} and d initial = a_0 g_0. Elsewhere (the float64 fixtures, the CPU) it is
a plain loop.
"""

import torch
import triton
import triton.language as tl


def sequential_scan(decay, contribution, initial):
    """decay, contribution (T, ..., D); initial (..., D) -> h (T, ..., D)."""
    hidden = initial
    outputs = []
    for t in range(decay.shape[0]):
        hidden = decay[t] * hidden + contribution[t]
        outputs.append(hidden)
    return torch.stack(outputs)


@triton.jit
def _scan_forward(decay, contribution, initial, out, steps, channels, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < channels
    hidden = tl.load(initial + offsets, mask=mask, other=0.0).to(tl.float32)
    for t in range(steps):
        index = t * channels + offsets
        a = tl.load(decay + index, mask=mask, other=0.0)
        b = tl.load(contribution + index, mask=mask, other=0.0)
        hidden = a * hidden + b
        tl.store(out + index, hidden, mask=mask)


@triton.jit
def _scan_backward(decay, initial, out, grad_out, grad_decay, grad_contribution, grad_initial,
                   steps, channels, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < channels
    carried = tl.zeros([BLOCK], dtype=tl.float32)
    next_decay = tl.zeros([BLOCK], dtype=tl.float32)
    for step in range(steps):
        t = steps - 1 - step
        index = t * channels + offsets
        carried = tl.load(grad_out + index, mask=mask, other=0.0) + next_decay * carried
        if t > 0:
            previous = tl.load(out + index - channels, mask=mask, other=0.0)
        else:
            previous = tl.load(initial + offsets, mask=mask, other=0.0)
        a = tl.load(decay + index, mask=mask, other=0.0)
        tl.store(grad_contribution + index, carried, mask=mask)
        tl.store(grad_decay + index, carried * previous, mask=mask)
        next_decay = a
    tl.store(grad_initial + offsets, next_decay * carried, mask=mask)


BLOCK = 256


@torch.library.custom_op("swift::linear_scan", mutates_args=())
def triton_scan(decay: torch.Tensor, contribution: torch.Tensor,
                initial: torch.Tensor) -> torch.Tensor:
    """decay, contribution (T, C) float32 contiguous; initial (C,) -> h (T, C)."""
    steps, channels = decay.shape
    out = torch.empty_like(decay)
    grid = (triton.cdiv(channels, BLOCK),)
    _scan_forward[grid](decay, contribution, initial, out, steps, channels, BLOCK=BLOCK)
    return out


@triton_scan.register_fake
def _(decay, contribution, initial):
    return torch.empty_like(decay)


@torch.library.custom_op("swift::linear_scan_backward", mutates_args=())
def triton_scan_backward(decay: torch.Tensor, initial: torch.Tensor, out: torch.Tensor,
                         grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                          torch.Tensor]:
    steps, channels = decay.shape
    grad_decay = torch.empty_like(decay)
    grad_contribution = torch.empty_like(decay)
    grad_initial = torch.empty_like(initial)
    grid = (triton.cdiv(channels, BLOCK),)
    _scan_backward[grid](decay, initial, out, grad_out.contiguous(), grad_decay,
                         grad_contribution, grad_initial, steps, channels, BLOCK=BLOCK)
    return grad_decay, grad_contribution, grad_initial


@triton_scan_backward.register_fake
def _(decay, initial, out, grad_out):
    return torch.empty_like(decay), torch.empty_like(decay), torch.empty_like(initial)


def _setup_context(ctx, inputs, output):
    decay, _, initial = inputs
    ctx.save_for_backward(decay, initial, output)


def _backward(ctx, grad_out):
    decay, initial, out = ctx.saved_tensors
    return triton_scan_backward(decay, initial, out, grad_out)


triton_scan.register_autograd(_backward, setup_context=_setup_context)


def linear_scan(decay, contribution, initial):
    """h_t = decay_t h_{t-1} + contribution_t, h_{-1} = initial. Time-major (T, ..., D)."""
    if decay.is_cuda and decay.dtype == torch.float32 and contribution.dtype == torch.float32:
        shape = decay.shape
        steps = shape[0]
        out = triton_scan(decay.reshape(steps, -1).contiguous(),
                          contribution.reshape(steps, -1).contiguous(),
                          initial.to(torch.float32).reshape(-1).contiguous())
        return out.view(shape)
    return sequential_scan(decay, contribution, initial)
