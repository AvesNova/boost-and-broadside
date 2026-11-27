"""The linear recurrence h_t = decay_t h_(t-1) + update_t over the leading (time) axis in one
pass: a Triton kernel on a GPU (one thread per channel, sequential over T, forward and
backward), a loop over T on the CPU. A custom operator (torch.library), so a compiled graph
calls it as one opaque node at any shape, static or dynamic.

The state is float32 (decided 2026-11-23): the inputs are cast to float32 and the states
returned in float32. Backward (with g_t = dL/dh_t from the output and the later states):
g_t = grad_t + decay_(t+1) g_(t+1); d decay_t = g_t h_(t-1); d update_t = g_t;
d initial = decay_0 g_0.
"""

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # a CPU-only install: the loop below
    triton = None

BLOCK = 256  # channels per Triton program


if triton is not None:

    @triton.jit
    def _forward_kernel(decay, update, initial, states, ticks, channels, BLOCK: tl.constexpr):
        columns = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = columns < channels
        state = tl.load(initial + columns, mask=valid, other=0.0)
        for tick in range(ticks):
            offset = tick * channels + columns
            a = tl.load(decay + offset, mask=valid, other=0.0)
            b = tl.load(update + offset, mask=valid, other=0.0)
            state = a * state + b
            tl.store(states + offset, state, mask=valid)

    @triton.jit
    def _backward_kernel(
        decay,
        states,
        initial,
        grad,
        decay_grad,
        update_grad,
        initial_grad,
        ticks,
        channels,
        BLOCK: tl.constexpr,
    ):
        columns = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = columns < channels
        carried = tl.zeros([BLOCK], dtype=tl.float32)  # g_(t+1)
        following = tl.zeros([BLOCK], dtype=tl.float32)  # decay_(t+1)
        for step in range(ticks):
            tick = ticks - 1 - step
            offset = tick * channels + columns
            g = tl.load(grad + offset, mask=valid, other=0.0) + following * carried
            if tick > 0:
                previous = tl.load(states + offset - channels, mask=valid, other=0.0)
            else:
                previous = tl.load(initial + columns, mask=valid, other=0.0)
            tl.store(decay_grad + offset, g * previous, mask=valid)
            tl.store(update_grad + offset, g, mask=valid)
            carried = g
            following = tl.load(decay + offset, mask=valid, other=0.0)
        tl.store(initial_grad + columns, following * carried, mask=valid)


def _grid(channels: int):
    return (triton.cdiv(channels, BLOCK),)


@torch.library.custom_op("lean::linear_scan", mutates_args=())
def _scan(decay: Tensor, update: Tensor, initial: Tensor) -> Tensor:
    ticks = update.shape[0]
    if update.is_cuda and triton is not None:
        states = torch.empty_like(update)
        channels = initial.numel()
        _forward_kernel[_grid(channels)](decay, update, initial, states, ticks, channels, BLOCK)
        return states
    rows, state = [], initial
    for tick in range(ticks):
        state = decay[tick] * state + update[tick]
        rows.append(state)
    return torch.stack(rows)


@_scan.register_fake
def _(decay, update, initial):
    return torch.empty_like(update)


@torch.library.custom_op("lean::linear_scan_backward", mutates_args=())
def _scan_backward(
    decay: Tensor, states: Tensor, initial: Tensor, grad: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    ticks = states.shape[0]
    if states.is_cuda and triton is not None:
        decay_grad, update_grad = torch.empty_like(states), torch.empty_like(states)
        initial_grad = torch.empty_like(initial)
        channels = initial.numel()
        _backward_kernel[_grid(channels)](
            decay,
            states,
            initial,
            grad,
            decay_grad,
            update_grad,
            initial_grad,
            ticks,
            channels,
            BLOCK,
        )
        return decay_grad, update_grad, initial_grad
    carried = torch.zeros_like(initial)
    following = torch.zeros_like(initial)
    decay_rows, update_rows = [], []
    for tick in reversed(range(ticks)):
        g = grad[tick] + following * carried
        previous = states[tick - 1] if tick > 0 else initial
        decay_rows.append(g * previous)
        update_rows.append(g)
        carried, following = g, decay[tick]
    return (
        torch.stack(decay_rows[::-1]),
        torch.stack(update_rows[::-1]),
        following * carried,
    )


@_scan_backward.register_fake
def _(decay, states, initial, grad):
    return torch.empty_like(states), torch.empty_like(states), torch.empty_like(initial)


def _setup(ctx, inputs, output):
    decay, _, initial = inputs
    ctx.save_for_backward(decay, output, initial)


def _backward(ctx, grad):
    decay, states, initial = ctx.saved_tensors
    return _scan_backward(decay, states, initial, grad.contiguous())


_scan.register_autograd(_backward, setup_context=_setup)


def linear_scan(decay: Tensor, update: Tensor, initial: Tensor) -> tuple[Tensor, Tensor]:
    """h_t = decay_t h_(t-1) + update_t over the leading axis from h_(-1) = initial, in one
    pass, float32. decay, update (T, ...) broadcasting to update's shape; initial (...).
    Returns every h (T, ...) float32 and the last."""
    update = update.float().contiguous()
    decay = decay.float().expand_as(update).contiguous()
    initial = initial.float().expand_as(update[0]).contiguous()
    states = _scan(decay, update, initial)
    return states, states[-1]
