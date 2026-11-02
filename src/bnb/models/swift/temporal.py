"""The temporal sublayer's gates, RG-LRU scan and output gate as one Triton kernel per
direction (spec 6.4).

For every channel c of a (T, S) problem (S = tokens x D, time-major), with the gate
pre-activations p_r, p_i (the recurrence and input gate GEMM, bias included), the convolved
input u and the gate branch v:

    r = sigmoid(p_r), i = sigmoid(p_i), a = sigmoid(Lambda)^(8 r) = exp(8 r logsigmoid(Lambda))
    b = sqrt(max((1 - a)(1 + a), 1e-6)) i u,   a <- a keep  (keep = 0 where an episode starts)
    h_t = a_t h_{t-1} + b_t,                    y_t = h_t gelu(v_t)

Each program owns BLOCK channels and walks time in chunks of CHUNK steps; a chunk is one
associative scan over (a, b) pairs in registers, so CHUNK rows of loads are in flight at
once (the plain one-step loop is latency bound at small batches). The forward writes h
(float32: the backward's h_{t-1} and the carry) and y (the output GEMM's input). The backward
walks the chunks in reverse, recomputes the gates from the same inputs and solves the
adjoint recurrence G_t = dh_t + gelu(v_t) dy_t + a_{t+1} G_{t+1} the same way; it writes the
gradients of p_r, p_i, u, v, the initial state, and per-channel partial sums for Lambda.
Nothing between the two GEMMs is stored but h.
"""

import torch
import triton
import triton.language as tl

CHUNK = 16
BLOCK = 64
SQRT_HALF = tl.constexpr(0.7071067811865476)
INVERSE_SQRT_TAU = tl.constexpr(0.3989422804014327)


@triton.jit
def _forward_combine(decay_earlier, value_earlier, decay_later, value_later):
    return decay_earlier * decay_later, decay_later * value_earlier + value_later


@triton.jit
def _reverse_combine(decay_later, value_later, decay_earlier, value_earlier):
    # tl.associative_scan(reverse=True) passes the accumulation of the later rows first
    return decay_later * decay_earlier, value_earlier + decay_earlier * value_later


@triton.jit
def _log_sigmoid(x):
    return tl.minimum(x, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(x)))


@triton.jit
def _gates(pre, keep, log_decay_sigmoid, row, d, mask, WIDTH: tl.constexpr,
           HAS_KEEP: tl.constexpr):
    """(r, i, a before keep, kept a) for the rows `row` (tile) at channels d."""
    recurrence = tl.sigmoid(tl.load(pre + row * (2 * WIDTH) + d, mask=mask, other=0.0)
                            .to(tl.float32))
    gate = tl.sigmoid(tl.load(pre + row * (2 * WIDTH) + WIDTH + d, mask=mask, other=0.0)
                      .to(tl.float32))
    decay = tl.exp(8.0 * recurrence * log_decay_sigmoid)
    kept = decay
    if HAS_KEEP:
        kept = decay * tl.load(keep + row, mask=mask, other=1.0)
    return recurrence, gate, decay, kept


@triton.jit
def _temporal_forward(pre, convolved, branch, keep, log_decay, initial, hidden_out, y_out,
                      steps, tokens, branch_stride, WIDTH: tl.constexpr,
                      HAS_KEEP: tl.constexpr, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    channels = tokens * WIDTH
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel_mask = c < channels
    token = c // WIDTH
    d = c % WIDTH
    log_decay_sigmoid = _log_sigmoid(tl.load(log_decay + d, mask=channel_mask, other=0.0))
    hidden = tl.load(initial + c, mask=channel_mask, other=0.0).to(tl.float32)
    rows_in_chunk = tl.arange(0, CHUNK)
    for start in range(0, steps, CHUNK):
        t = start + rows_in_chunk
        mask = (t[:, None] < steps) & channel_mask[None, :]
        row = t[:, None] * tokens + token[None, :]
        _, gate, decay, kept = _gates(pre, keep, log_decay_sigmoid[None, :], row, d[None, :],
                                      mask, WIDTH, HAS_KEEP)
        u = tl.load(convolved + t[:, None] * channels + c[None, :], mask=mask,
                    other=0.0).to(tl.float32)
        contribution = tl.sqrt(tl.maximum((1.0 - decay) * (1.0 + decay), 1e-6)) * gate * u
        kept = tl.where(mask, kept, 1.0)
        contribution = tl.where(mask, contribution, 0.0)
        decay_product, value = tl.associative_scan((kept, contribution), 0, _forward_combine)
        states = decay_product * hidden[None, :] + value
        index = t[:, None] * channels + c[None, :]
        tl.store(hidden_out + index, states, mask=mask)
        v = tl.load(branch + row * branch_stride + d[None, :], mask=mask,
                    other=0.0).to(tl.float32)
        gelu = 0.5 * v * (1.0 + tl.math.erf(v * SQRT_HALF))
        tl.store(y_out + index, (states * gelu).to(y_out.dtype.element_ty), mask=mask)
        hidden = tl.sum(tl.where(rows_in_chunk[:, None] == CHUNK - 1, states, 0.0), 0)


@triton.jit
def _temporal_backward(pre, convolved, branch, keep, log_decay, initial, hidden, grad_y,
                       grad_hidden, grad_pre, grad_convolved, grad_branch, grad_initial,
                       grad_log_decay_partial, steps, tokens, branch_stride,
                       WIDTH: tl.constexpr, HAS_KEEP: tl.constexpr,
                       HAS_GRAD_HIDDEN: tl.constexpr, CHUNK: tl.constexpr,
                       BLOCK: tl.constexpr):
    channels = tokens * WIDTH
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel_mask = c < channels
    token = c // WIDTH
    d = c % WIDTH
    log_decay_sigmoid = _log_sigmoid(tl.load(log_decay + d, mask=channel_mask, other=0.0))
    rows_in_chunk = tl.arange(0, CHUNK)
    adjoint_after = tl.zeros([BLOCK], dtype=tl.float32)  # G at the first row of the later chunk
    log_decay_sum = tl.zeros([BLOCK], dtype=tl.float32)
    chunks = tl.cdiv(steps, CHUNK)
    for back in range(chunks):
        start = (chunks - 1 - back) * CHUNK
        t = start + rows_in_chunk
        mask = (t[:, None] < steps) & channel_mask[None, :]
        row = t[:, None] * tokens + token[None, :]
        index = t[:, None] * channels + c[None, :]
        recurrence, gate, decay, kept = _gates(pre, keep, log_decay_sigmoid[None, :], row,
                                               d[None, :], mask, WIDTH, HAS_KEEP)
        next_mask = (t[:, None] + 1 < steps) & channel_mask[None, :]
        _, _, _, next_kept = _gates(pre, keep, log_decay_sigmoid[None, :], row + tokens,
                                    d[None, :], next_mask, WIDTH, HAS_KEEP)
        next_kept = tl.where(next_mask, next_kept, 0.0)
        states = tl.load(hidden + index, mask=mask, other=0.0)
        previous_mask = (t[:, None] >= 1) & mask
        previous = tl.load(hidden + index - channels, mask=previous_mask, other=0.0)
        previous = tl.where(t[:, None] == 0,
                            tl.load(initial + c, mask=channel_mask, other=0.0)[None, :],
                            previous)
        v = tl.load(branch + row * branch_stride + d[None, :], mask=mask,
                    other=0.0).to(tl.float32)
        cumulative = 0.5 * (1.0 + tl.math.erf(v * SQRT_HALF))
        gy = tl.load(grad_y + index, mask=mask, other=0.0).to(tl.float32)
        direct = gy * v * cumulative
        if HAS_GRAD_HIDDEN:
            direct += tl.load(grad_hidden + index, mask=mask, other=0.0).to(tl.float32)
        direct = tl.where(mask, direct, 0.0)
        # G_t = direct_t + a_{t+1} G_{t+1}: the later chunk enters through adjoint_after
        decay_product, value = tl.associative_scan((next_kept, direct), 0, _reverse_combine,
                                                   reverse=True)
        # decay_product_t = a_{t+1} ... a_{end+1}, where a_{end+1} links to the later chunk
        adjoint = value + decay_product * adjoint_after[None, :]
        grad_v = gy * states * (cumulative + v * tl.exp(-0.5 * v * v) * INVERSE_SQRT_TAU)
        u = tl.load(convolved + index, mask=mask, other=0.0).to(tl.float32)
        multiplier = tl.sqrt(tl.maximum((1.0 - decay) * (1.0 + decay), 1e-6))
        grad_multiplier = adjoint * gate * u
        grad_gate = adjoint * multiplier * u
        grad_u = adjoint * multiplier * gate
        grad_kept = adjoint * previous
        grad_decay = grad_kept
        if HAS_KEEP:
            grad_decay = grad_kept * tl.load(keep + row, mask=mask, other=1.0)
        unclamped = (1.0 - decay) * (1.0 + decay) > 1e-6
        grad_decay += tl.where(unclamped, -grad_multiplier * decay / multiplier, 0.0)
        exponent_grad = tl.where(mask, grad_decay * decay * 8.0, 0.0)
        log_decay_sum += tl.sum(exponent_grad * recurrence, 0)
        grad_recurrence = exponent_grad * log_decay_sigmoid[None, :]
        out = grad_pre + row * (2 * WIDTH) + d[None, :]
        tl.store(out, (grad_recurrence * recurrence * (1.0 - recurrence))
                 .to(grad_pre.dtype.element_ty), mask=mask)
        tl.store(out + WIDTH, (grad_gate * gate * (1.0 - gate)).to(grad_pre.dtype.element_ty),
                 mask=mask)
        tl.store(grad_convolved + index, grad_u.to(grad_convolved.dtype.element_ty), mask=mask)
        tl.store(grad_branch + row * WIDTH + d[None, :],
                 grad_v.to(grad_branch.dtype.element_ty), mask=mask)
        adjoint_after = tl.sum(tl.where(rows_in_chunk[:, None] == 0, adjoint, 0.0), 0)
    _, _, _, first_kept = _gates(pre, keep, log_decay_sigmoid, token, d, channel_mask, WIDTH,
                                 HAS_KEEP)
    tl.store(grad_initial + c, first_kept * adjoint_after, mask=channel_mask)
    tl.store(grad_log_decay_partial + c, log_decay_sum, mask=channel_mask)


def _grid(channels):
    return (triton.cdiv(channels, BLOCK),)


# triton_op (not custom_op): under torch.compile the launches go into Inductor's wrapper code,
# without a Python custom-op dispatch per call (the CPU side bounds small training batches)
@torch.library.triton_op("swift::temporal_mix", mutates_args=())
def temporal_mix(pre: torch.Tensor, convolved: torch.Tensor, branch: torch.Tensor,
                 keep: torch.Tensor | None, log_decay: torch.Tensor,
                 initial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """pre (T, S_tok, 2D) contiguous; convolved (T, S_tok, D) contiguous; branch (T, S_tok, D)
    with unit channel stride (a view of the projection is fine); keep (T, S_tok) float32 or
    None; log_decay (D,); initial (S_tok, D) float32.
    Returns (hidden (T, S_tok, D) float32, y (T, S_tok, D) in branch's dtype)."""
    steps, tokens, width = convolved.shape
    hidden = torch.empty(convolved.shape, dtype=torch.float32, device=convolved.device)
    y = torch.empty(convolved.shape, dtype=branch.dtype, device=convolved.device)
    torch.library.wrap_triton(_temporal_forward)[_grid(tokens * width)](
        pre, convolved, branch, pre if keep is None else keep, log_decay, initial, hidden, y,
        steps, tokens, branch.stride(1), WIDTH=width, HAS_KEEP=keep is not None, CHUNK=CHUNK,
        BLOCK=BLOCK)
    return hidden, y


@torch.library.triton_op("swift::temporal_mix_backward", mutates_args=())
def temporal_mix_backward(pre: torch.Tensor, convolved: torch.Tensor, branch: torch.Tensor,
                          keep: torch.Tensor | None, log_decay: torch.Tensor,
                          initial: torch.Tensor, hidden: torch.Tensor, grad_y: torch.Tensor,
                          grad_hidden: torch.Tensor | None
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                     torch.Tensor]:
    """grad_y contiguous. Returns the gradients of (pre, convolved, branch, initial) and the
    per-channel partial sums (S_tok, D) of dL/d logsigmoid(Lambda)."""
    steps, tokens, width = convolved.shape
    grad_pre = torch.empty_like(pre)
    grad_convolved = torch.empty_like(convolved)
    grad_branch = torch.empty(convolved.shape, dtype=branch.dtype, device=branch.device)
    grad_initial = torch.empty_like(initial)
    partial = torch.empty(tokens, width, dtype=torch.float32, device=pre.device)
    torch.library.wrap_triton(_temporal_backward)[_grid(tokens * width)](
        pre, convolved, branch, pre if keep is None else keep, log_decay, initial, hidden,
        grad_y, grad_y if grad_hidden is None else grad_hidden, grad_pre, grad_convolved,
        grad_branch, grad_initial, partial, steps, tokens, branch.stride(1), WIDTH=width,
        HAS_KEEP=keep is not None, HAS_GRAD_HIDDEN=grad_hidden is not None, CHUNK=CHUNK,
        BLOCK=BLOCK)
    return grad_pre, grad_convolved, grad_branch, grad_initial, partial


def _setup_context(ctx, inputs, output):
    pre, convolved, branch, keep, log_decay, initial = inputs
    ctx.has_keep = keep is not None
    ctx.save_for_backward(pre, convolved, branch, keep, log_decay, initial, output[0])


def _backward(ctx, grad_hidden, grad_y):
    pre, convolved, branch, keep, log_decay, initial, hidden = ctx.saved_tensors
    if not ctx.has_keep:
        keep = None
    if grad_y is None:
        grad_y = torch.zeros_like(hidden, dtype=branch.dtype)
    grads = temporal_mix_backward(pre, convolved, branch, keep, log_decay, initial, hidden,
                                  grad_y.contiguous(),
                                  None if grad_hidden is None else grad_hidden.contiguous())
    grad_pre, grad_convolved, grad_branch, grad_initial, partial = grads
    grad_log_decay = partial.sum(0) * torch.sigmoid(-log_decay.float())
    return (grad_pre, grad_convolved, grad_branch, None, grad_log_decay.to(log_decay.dtype),
            grad_initial)


temporal_mix.register_autograd(_backward, setup_context=_setup_context)


def supported(convolved, branch) -> bool:
    """The kernel handles CUDA float32 / 16-bit tensors with a unit channel stride."""
    return (convolved.is_cuda and convolved.dtype in (torch.float32, torch.bfloat16,
                                                      torch.float16)
            and branch.stride(-1) == 1)
