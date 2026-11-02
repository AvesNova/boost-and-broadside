"""The linear recurrence h_t = a_t h_(t-1) + b_t over a leading time axis, as one Triton
kernel per direction (CUDA, float32).

Each program owns a block of channels and walks time, so a (L, C) problem reads a and b
once and writes h once. The backward is the reverse recurrence g_t = dL/dh_t + a_(t+1)
g_(t+1), with db_t = g_t, da_t = g_t h_(t-1) and d initial = a_0 g_0. No atomics: every
output element has one writer, so the result does not depend on launch order. The plain
twin is `temporal.parallel_scan`.
"""

import torch
import triton
import triton.language as tl

# Channels per program: elementwise along channels, so every configuration gives the same bits.
CONFIGS = [triton.Config({"BLOCK": block}, num_warps=warps)
           for block, warps in ((128, 4), (256, 4), (512, 4), (1024, 8))]


@triton.autotune(configs=CONFIGS, key=["channels"], cache_results=True)
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


@triton.autotune(configs=CONFIGS, key=["channels"], cache_results=True)
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


def _grid(channels):
    return lambda meta: (triton.cdiv(channels, meta["BLOCK"]),)


@torch.library.triton_op("yemong::linear_scan", mutates_args=())
def _linear_scan(decay: torch.Tensor, contribution: torch.Tensor,
                 initial: torch.Tensor) -> torch.Tensor:
    """decay, contribution (L, C) float32 contiguous; initial (C,) -> h (L, C)."""
    steps, channels = decay.shape
    out = torch.empty_like(decay)
    torch.library.wrap_triton(_scan_forward)[_grid(channels)](decay, contribution, initial,
                                                              out, steps, channels)
    return out


@torch.library.triton_op("yemong::linear_scan_backward", mutates_args=())
def _linear_scan_backward(decay: torch.Tensor, initial: torch.Tensor, out: torch.Tensor,
                          grad_out: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    steps, channels = decay.shape
    grad_decay = torch.empty_like(decay)
    grad_contribution = torch.empty_like(decay)
    grad_initial = torch.empty_like(initial)
    torch.library.wrap_triton(_scan_backward)[_grid(channels)](
        decay, initial, out, grad_out.contiguous(), grad_decay, grad_contribution,
        grad_initial, steps, channels)
    return grad_decay, grad_contribution, grad_initial


def _setup_context(ctx, inputs, output):
    decay, _, initial = inputs
    ctx.save_for_backward(decay, initial, output)


def _backward(ctx, grad_out):
    decay, initial, out = ctx.saved_tensors
    return _linear_scan_backward(decay, initial, out, grad_out)


_linear_scan.register_autograd(_backward, setup_context=_setup_context)


def applies(decay: torch.Tensor, contribution: torch.Tensor, initial: torch.Tensor) -> bool:
    return decay.is_cuda and all(t.dtype == torch.float32
                                 for t in (decay, contribution, initial))


def linear_scan(decay: torch.Tensor, contribution: torch.Tensor,
                initial: torch.Tensor) -> torch.Tensor:
    """Every h (L, ..., D) of h_t = decay_t h_(t-1) + contribution_t from h_(-1) = initial
    (..., D). CUDA float32 only (`applies`)."""
    steps = decay.shape[0]
    out = _linear_scan(decay.reshape(steps, -1).contiguous(),
                       contribution.reshape(steps, -1).contiguous(),
                       initial.expand(decay.shape[1:]).reshape(-1).contiguous())
    return out.view(decay.shape)
