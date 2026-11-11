"""out = W2 gelu(RMSNorm(W1 x + b1)) + b2 as one Triton kernel per direction (16-bit). The
plain twins: a head of `HeadGroup`, the tail of `ShotTrajectoryHead.query` and the
encoder's `KindProjection` followed by its `shared` projection.

Staged (`staged_head`, the whole of `ShotTrajectoryHead.query` at ages shared by every
row): x itself is gelu(RMSNorm(latent + age)) of a latent row and an age row (row r a of
the output is latent r at age a), made in the kernel from the two (as h is), so neither x
nor its norm's gradient is written; the backward forms the input norm's gradient in
registers too and writes the gradient of latent + age, summed over the ages and the rows
afterwards.

Every row's whole hidden vector (H = 64, 128 or 256) fits in one program's registers, so the
forward computes h = W1 x + b1, the norm, the gelu and the output product without writing h
or the activation. The backward recomputes them, forms d activation = dout W2 in registers
and the norm and gelu gradients there, and writes only dh and the activation a (the inputs
of the weight and input gradient GEMMs, which stay cuBLAS; the input's is skipped when x
needs none). The norm's scale gradient comes out as partial sums over fixed groups of
rows, summed afterwards in a fixed order. No atomics.

Precision: x, W1, W2 in the autocast dtype; h, the norm and the gelu in float32 (RMSNorm
with float32's epsilon, which is what `layers.norm` uses on a bf16 input too); out in the
autocast dtype. Staged: latent + age, its norm and gelu in float32 too, x rounded to the
autocast dtype once (the plain path rounds the sum and the norm's output as well).
"""

import torch
import triton
import triton.language as tl

# Rows per program, the input and output column tiles, warps. The backward's rows per program
# also group the norm scale's partial sums. The warps can change the last bits (the layout
# orders the norm's sums). The 128-row configuration needs more shared memory than some GPUs
# have; autotuning skips it there.
CONFIGS = [triton.Config({"BLOCK_M": m, "BLOCK_K": k, "BLOCK_O": 64}, num_warps=warps,
                         num_stages=2)
           for m, k, warps in ((32, 64, 4), (64, 64, 4), (64, 64, 8), (128, 32, 8))]
KEY = ["M", "K", "N", "HIDDEN", "STAGED"]
GROUP = min(config.kwargs["BLOCK_M"] for config in CONFIGS)  # rows per norm-scale partial sum
LARGEST_BLOCK_M = max(config.kwargs["BLOCK_M"] for config in CONFIGS)
EPSILON = tl.constexpr(1.1920928955078125e-07)  # float32 machine epsilon
SQRT_HALF = tl.constexpr(0.7071067811865476)
INVERSE_SQRT_TAU = tl.constexpr(0.3989422804014327)


@triton.jit
def _hidden(x, first_weight, first_bias, rows, M, K, HIDDEN: tl.constexpr,
            BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """h = x W1^T + b1 for the rows: (BLOCK_M, HIDDEN) float32."""
    hidden_index = tl.arange(0, HIDDEN)
    accumulator = tl.zeros((BLOCK_M, HIDDEN), dtype=tl.float32)
    for start in range(0, K, BLOCK_K):
        inner = start + tl.arange(0, BLOCK_K)
        tile = tl.load(x + rows[:, None] * K + inner[None, :],
                       mask=(rows[:, None] < M) & (inner[None, :] < K), other=0.0)
        weight = tl.load(first_weight + hidden_index[None, :] * K + inner[:, None],
                         mask=inner[:, None] < K, other=0.0)
        accumulator = tl.dot(tile, weight, accumulator)
    return accumulator + tl.load(first_bias + hidden_index).to(tl.float32)[None, :]


@triton.jit
def _staged_input(latent, age, input_norm, rows, M, AGES, INPUTS: tl.constexpr):
    """x = gelu(RMSNorm(latent[row // AGES] + age[row % AGES])) for the rows, float32
    (BLOCK_M, INPUTS), with what its backward reads: the norm's output and unit vector, the
    gelu's cumulative and the inverse rms."""
    inner = tl.arange(0, INPUTS)
    mask = rows[:, None] < M
    pre = (tl.load(latent + (rows // AGES)[:, None] * INPUTS + inner[None, :], mask=mask,
                   other=0.0).to(tl.float32)
           + tl.load(age + (rows % AGES)[:, None] * INPUTS + inner[None, :], mask=mask,
                     other=0.0).to(tl.float32))
    inverse_rms = tl.rsqrt(tl.sum(pre * pre, 1) / INPUTS + EPSILON)
    unit = pre * inverse_rms[:, None]
    normed = unit * tl.load(input_norm + inner)[None, :]
    cumulative = 0.5 * (1.0 + tl.math.erf(normed * SQRT_HALF))
    return normed * cumulative, normed, cumulative, unit, inverse_rms


@triton.jit
def _staged_hidden(x, first_weight, first_bias, INPUTS: tl.constexpr, HIDDEN: tl.constexpr):
    """h = x W1^T + b1 of a staged input x (BLOCK_M, INPUTS): (BLOCK_M, HIDDEN) float32,
    the product in the weight's dtype."""
    hidden_index = tl.arange(0, HIDDEN)
    inner = tl.arange(0, INPUTS)
    weight = tl.load(first_weight + hidden_index[None, :] * INPUTS + inner[:, None])
    return (tl.dot(x.to(first_weight.dtype.element_ty), weight)
            + tl.load(first_bias + hidden_index).to(tl.float32)[None, :])


@triton.autotune(configs=CONFIGS, key=KEY, cache_results=True)
@triton.jit
def _head_forward(x, first_weight, first_bias, norm_weight, last_weight, last_bias, out, M, K,
                  N, age, input_norm, AGES, HIDDEN: tl.constexpr, STAGED: tl.constexpr,
                  INPUTS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
                  BLOCK_O: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    hidden_index = tl.arange(0, HIDDEN)
    if STAGED:  # x is the latent (M // AGES, INPUTS); age (AGES, INPUTS)
        staged, _, _, _, _ = _staged_input(x, age, input_norm, rows, M, AGES, INPUTS)
        h = _staged_hidden(staged, first_weight, first_bias, INPUTS, HIDDEN)
    else:
        h = _hidden(x, first_weight, first_bias, rows, M, K, HIDDEN, BLOCK_M, BLOCK_K)
    inverse_rms = tl.rsqrt(tl.sum(h * h, 1) / HIDDEN + EPSILON)
    normed = h * inverse_rms[:, None] * tl.load(norm_weight + hidden_index)[None, :]
    activation = (0.5 * normed * (1.0 + tl.math.erf(normed * SQRT_HALF))).to(
        x.dtype.element_ty)
    for start in range(0, N, BLOCK_O):
        columns = start + tl.arange(0, BLOCK_O)
        weight = tl.load(last_weight + columns[None, :] * HIDDEN + hidden_index[:, None],
                         mask=columns[None, :] < N, other=0.0)
        result = tl.dot(activation, weight)
        result += tl.load(last_bias + columns, mask=columns < N, other=0.0)[None, :]
        tl.store(out + rows[:, None] * N + columns[None, :], result.to(out.dtype.element_ty),
                 mask=(rows[:, None] < M) & (columns[None, :] < N))


@triton.autotune(configs=CONFIGS, key=KEY, cache_results=True)
@triton.jit
def _head_backward(x, first_weight, first_bias, norm_weight, last_weight, grad_out,
                   grad_hidden, activation_out, norm_partial, M, K, N, age, input_norm, AGES,
                   input_out, grad_input, input_partial, HIDDEN: tl.constexpr,
                   GROUP: tl.constexpr, STAGED: tl.constexpr, INPUTS: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_O: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    hidden_index = tl.arange(0, HIDDEN)
    if STAGED:
        staged, staged_normed, staged_cumulative, staged_unit, staged_inverse_rms = (
            _staged_input(x, age, input_norm, rows, M, AGES, INPUTS))
        h = _staged_hidden(staged, first_weight, first_bias, INPUTS, HIDDEN)
    else:
        h = _hidden(x, first_weight, first_bias, rows, M, K, HIDDEN, BLOCK_M, BLOCK_K)
    inverse_rms = tl.rsqrt(tl.sum(h * h, 1) / HIDDEN + EPSILON)
    scale = tl.load(norm_weight + hidden_index)[None, :]
    unit = h * inverse_rms[:, None]
    normed = unit * scale
    cumulative = 0.5 * (1.0 + tl.math.erf(normed * SQRT_HALF))
    activation = normed * cumulative
    grad_activation = tl.zeros((BLOCK_M, HIDDEN), dtype=tl.float32)
    for start in range(0, N, BLOCK_O):
        columns = start + tl.arange(0, BLOCK_O)
        column_mask = columns < N
        upstream = tl.load(grad_out + rows[:, None] * N + columns[None, :],
                           mask=row_mask[:, None] & column_mask[None, :], other=0.0)
        weight = tl.load(last_weight + columns[:, None] * HIDDEN + hidden_index[None, :],
                         mask=column_mask[:, None], other=0.0)
        grad_activation = tl.dot(upstream, weight, grad_activation)
    grad_normed = grad_activation * (cumulative + normed * tl.exp(-0.5 * normed * normed)
                                     * INVERSE_SQRT_TAU)
    grad_normed = tl.where(row_mask[:, None], grad_normed, 0.0)
    # One partial sum per GROUP rows, whatever the rows per program: every configuration
    # writes the same groups (a group past the last row is zero), so none leaves stale rows.
    GROUPS: tl.constexpr = BLOCK_M // GROUP
    partials = tl.sum(tl.reshape(grad_normed * unit, (GROUPS, GROUP, HIDDEN)), 1)
    group_rows = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    tl.store(norm_partial + group_rows[:, None] * HIDDEN + hidden_index[None, :], partials)
    grad_unit = grad_normed * scale
    projection = tl.sum(grad_unit * unit, 1) / HIDDEN
    grad_h = inverse_rms[:, None] * (grad_unit - unit * projection[:, None])
    index = rows[:, None] * HIDDEN + hidden_index[None, :]
    mask = row_mask[:, None]
    tl.store(grad_hidden + index, grad_h.to(grad_hidden.dtype.element_ty), mask=mask)
    tl.store(activation_out + index, activation.to(activation_out.dtype.element_ty), mask=mask)
    if STAGED:  # the staged input's gradient, through x = gelu(RMSNorm(latent + age))
        inner = tl.arange(0, INPUTS)
        weight = tl.load(first_weight + hidden_index[:, None] * INPUTS + inner[None, :])
        grad_x = tl.dot(grad_h.to(first_weight.dtype.element_ty), weight)
        grad_staged = grad_x * (staged_cumulative + staged_normed
                                * tl.exp(-0.5 * staged_normed * staged_normed)
                                * INVERSE_SQRT_TAU)
        grad_staged = tl.where(row_mask[:, None], grad_staged, 0.0)
        partials = tl.sum(tl.reshape(grad_staged * staged_unit, (GROUPS, GROUP, INPUTS)), 1)
        tl.store(input_partial + group_rows[:, None] * INPUTS + inner[None, :], partials)
        grad_unit = grad_staged * tl.load(input_norm + inner)[None, :]
        projection = tl.sum(grad_unit * staged_unit, 1) / INPUTS
        grad_pre = staged_inverse_rms[:, None] * (grad_unit - staged_unit * projection[:, None])
        index = rows[:, None] * INPUTS + inner[None, :]
        tl.store(grad_input + index, grad_pre.to(grad_input.dtype.element_ty), mask=mask)
        tl.store(input_out + index, staged.to(input_out.dtype.element_ty), mask=mask)


def _grid(rows):
    return lambda meta: (triton.cdiv(rows, meta["BLOCK_M"]),)


@torch.library.triton_op("yemong::fused_head", mutates_args=())
def fused_head(x: torch.Tensor, first_weight: torch.Tensor, first_bias: torch.Tensor,
               norm_weight: torch.Tensor, last_weight: torch.Tensor,
               last_bias: torch.Tensor) -> torch.Tensor:
    """x (M, K) contiguous 16-bit; first_weight (H, K), last_weight (N, H) in x's dtype; biases
    and the norm scale float32. Returns (M, N) in x's dtype."""
    rows, inputs = x.shape
    hidden, outputs = first_weight.shape[0], last_weight.shape[0]
    out = torch.empty(rows, outputs, dtype=x.dtype, device=x.device)
    torch.library.wrap_triton(_head_forward)[_grid(rows)](
        x, first_weight, first_bias, norm_weight, last_weight, last_bias, out, rows, inputs,
        outputs, x, first_bias, 1, HIDDEN=hidden, STAGED=False, INPUTS=1)
    return out


@torch.library.triton_op("yemong::fused_head_backward", mutates_args=())
def fused_head_backward(x: torch.Tensor, first_weight: torch.Tensor, first_bias: torch.Tensor,
                        norm_weight: torch.Tensor, last_weight: torch.Tensor,
                        grad_out: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """grad_out (M, N) contiguous in x's dtype. Returns (dh (M, H), activation (M, H), norm
    scale partial sums (groups, H) float32)."""
    rows, inputs = x.shape
    hidden, outputs = first_weight.shape[0], last_weight.shape[0]
    groups = triton.cdiv(rows, LARGEST_BLOCK_M) * (LARGEST_BLOCK_M // GROUP)  # any configuration
    grad_hidden = torch.empty(rows, hidden, dtype=x.dtype, device=x.device)
    activation = torch.empty(rows, hidden, dtype=x.dtype, device=x.device)
    partial = torch.zeros(groups, hidden, dtype=torch.float32, device=x.device)
    torch.library.wrap_triton(_head_backward)[_grid(rows)](
        x, first_weight, first_bias, norm_weight, last_weight, grad_out, grad_hidden,
        activation, partial, rows, inputs, outputs, x, first_bias, 1, grad_hidden, grad_hidden,
        partial, HIDDEN=hidden, GROUP=GROUP, STAGED=False, INPUTS=1)
    return grad_hidden, activation, partial


def _setup_context(ctx, inputs, output):
    x, first_weight, first_bias, norm_weight, last_weight, last_bias = inputs
    ctx.save_for_backward(x, first_weight, first_bias, norm_weight, last_weight)


def _backward(ctx, grad_out):
    x, first_weight, first_bias, norm_weight, last_weight = ctx.saved_tensors
    grad_out = grad_out.to(x.dtype).contiguous()
    grad_hidden, activation, partial = fused_head_backward(
        x, first_weight, first_bias, norm_weight, last_weight, grad_out)
    grad_x = grad_hidden @ first_weight if ctx.needs_input_grad[0] else None  # not for inputs
    return (grad_x, grad_hidden.t() @ x,
            grad_hidden.float().sum(0), partial.sum(0), grad_out.t() @ activation,
            grad_out.float().sum(0))


fused_head.register_autograd(_backward, setup_context=_setup_context)


@torch.library.triton_op("yemong::fused_staged_head", mutates_args=())
def fused_staged_head(latent: torch.Tensor, age: torch.Tensor, input_norm_weight: torch.Tensor,
                      first_weight: torch.Tensor, first_bias: torch.Tensor,
                      norm_weight: torch.Tensor, last_weight: torch.Tensor,
                      last_bias: torch.Tensor) -> torch.Tensor:
    """latent (R, K) and age (A, K) contiguous 16-bit; the input norm's scale float32; the
    rest as `fused_head`'s. Returns (R A, N) in latent's dtype, row r A + a of latent r at
    age a."""
    inputs = latent.shape[1]
    rows = latent.shape[0] * age.shape[0]
    hidden, outputs = first_weight.shape[0], last_weight.shape[0]
    out = torch.empty(rows, outputs, dtype=latent.dtype, device=latent.device)
    torch.library.wrap_triton(_head_forward)[_grid(rows)](
        latent, first_weight, first_bias, norm_weight, last_weight, last_bias, out, rows,
        inputs, outputs, age, input_norm_weight, age.shape[0], HIDDEN=hidden, STAGED=True,
        INPUTS=inputs)
    return out


@torch.library.triton_op("yemong::fused_staged_head_backward", mutates_args=())
def fused_staged_head_backward(latent: torch.Tensor, age: torch.Tensor,
                               input_norm_weight: torch.Tensor, first_weight: torch.Tensor,
                               first_bias: torch.Tensor, norm_weight: torch.Tensor,
                               last_weight: torch.Tensor, grad_out: torch.Tensor
                               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                          torch.Tensor, torch.Tensor, torch.Tensor]:
    """grad_out (R A, N) contiguous in latent's dtype. Returns (dh (R A, H), activation
    (R A, H), norm scale partial sums (groups, H) float32, x (R A, K), the gradient of
    latent + age (R A, K), the input norm scale's partial sums (groups, K) float32)."""
    inputs = latent.shape[1]
    rows = latent.shape[0] * age.shape[0]
    hidden, outputs = first_weight.shape[0], last_weight.shape[0]
    groups = triton.cdiv(rows, LARGEST_BLOCK_M) * (LARGEST_BLOCK_M // GROUP)  # any configuration
    grad_hidden = torch.empty(rows, hidden, dtype=latent.dtype, device=latent.device)
    activation = torch.empty(rows, hidden, dtype=latent.dtype, device=latent.device)
    partial = torch.zeros(groups, hidden, dtype=torch.float32, device=latent.device)
    staged = torch.empty(rows, inputs, dtype=latent.dtype, device=latent.device)
    grad_input = torch.empty(rows, inputs, dtype=latent.dtype, device=latent.device)
    input_partial = torch.zeros(groups, inputs, dtype=torch.float32, device=latent.device)
    torch.library.wrap_triton(_head_backward)[_grid(rows)](
        latent, first_weight, first_bias, norm_weight, last_weight, grad_out, grad_hidden,
        activation, partial, rows, inputs, outputs, age, input_norm_weight, age.shape[0],
        staged, grad_input, input_partial, HIDDEN=hidden, GROUP=GROUP, STAGED=True,
        INPUTS=inputs)
    return grad_hidden, activation, partial, staged, grad_input, input_partial


def _staged_setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:-1])  # all but the last bias


def _staged_backward(ctx, grad_out):
    latent, age, input_norm_weight, first_weight, first_bias, norm_weight, last_weight = (
        ctx.saved_tensors)
    grad_out = grad_out.to(latent.dtype).contiguous()
    grad_hidden, activation, partial, staged, grad_input, input_partial = (
        fused_staged_head_backward(latent, age, input_norm_weight, first_weight, first_bias,
                                   norm_weight, last_weight, grad_out))
    by_age = grad_input.view(latent.shape[0], age.shape[0], -1)
    return (by_age.float().sum(1).to(latent.dtype), by_age.float().sum(0).to(age.dtype),
            input_partial.sum(0), grad_hidden.t() @ staged, grad_hidden.float().sum(0),
            partial.sum(0), grad_out.t() @ activation, grad_out.float().sum(0))


fused_staged_head.register_autograd(_staged_backward, setup_context=_staged_setup_context)


def staged_applies(latent: torch.Tensor, age: torch.Tensor, hidden: int) -> bool:
    """`applies`, with the input a width of 64, 128 or 256 and one age table (ages, K) for
    every latent row (`staged_head`)."""
    return (applies(latent, hidden) and age.dim() == 2 and age.shape[-1] == latent.shape[-1]
            and latent.shape[-1] in (64, 128, 256))


def applies(inputs: torch.Tensor, hidden: int) -> bool:
    """CUDA under 16-bit autocast, a hidden width of 64, 128 or 256."""
    return (inputs.is_cuda and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") in (torch.bfloat16, torch.float16)
            and hidden in (64, 128, 256))


def head(inputs: torch.Tensor, first: torch.nn.Linear | tuple, norm: torch.nn.RMSNorm,
         last: torch.nn.Linear) -> torch.Tensor:
    """last(gelu(norm(first(inputs)))) for inputs (..., K), in the autocast dtype
    (`applies`). first is a Linear or its (weight, bias) (a row block of a group's)."""
    first_weight, first_bias = (first.weight, first.bias) if isinstance(
        first, torch.nn.Linear) else first
    dtype = torch.get_autocast_dtype("cuda")
    flat = inputs.reshape(-1, inputs.shape[-1]).to(dtype).contiguous()
    out = fused_head(flat, first_weight.to(dtype).contiguous(), first_bias.float(),
                     norm.weight.float(), last.weight.to(dtype).contiguous(),
                     last.bias.float())
    return out.view(*inputs.shape[:-1], last.weight.shape[0])


def staged_head(latent: torch.Tensor, age: torch.Tensor, input_norm: torch.nn.RMSNorm,
                first: torch.nn.Linear, norm: torch.nn.RMSNorm,
                last: torch.nn.Linear) -> torch.Tensor:
    """last(gelu(norm(first(gelu(input_norm(latent[..., None, :] + age)))))) for latent
    (..., K) and age (ages, K): (..., ages, outputs) in the autocast dtype (`staged_applies`)."""
    dtype = torch.get_autocast_dtype("cuda")
    flat = latent.reshape(-1, latent.shape[-1]).to(dtype).contiguous()
    out = fused_staged_head(flat, age.to(dtype).contiguous(), input_norm.weight.float(),
                            first.weight.to(dtype).contiguous(), first.bias.float(),
                            norm.weight.float(), last.weight.to(dtype).contiguous(),
                            last.bias.float())
    return out.view(*latent.shape[:-1], age.shape[0], last.weight.shape[0])
