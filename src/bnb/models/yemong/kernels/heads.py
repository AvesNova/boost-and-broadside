"""out = W2 gelu(RMSNorm(W1 x + b1)) + b2 as one Triton kernel per direction (16-bit). The
plain twins: a head of `HeadGroup`, the tail of `ShotTrajectoryHead.query` and the
encoder's `KindProjection` followed by its `shared` projection.

Every row's whole hidden vector (H = 64, 128 or 256) fits in one program's registers, so the
forward computes h = W1 x + b1, the norm, the gelu and the output product without writing h
or the activation. The backward recomputes them, forms d activation = dout W2 in registers
and the norm and gelu gradients there, and writes only dh and the activation a (the inputs
of the weight and input gradient GEMMs, which stay cuBLAS; the input's is skipped when x
needs none). The norm's scale gradient comes out as per-program partial sums, summed
afterwards in a fixed order. No atomics.

Precision: x, W1, W2 in the autocast dtype; h, the norm and the gelu in float32 (RMSNorm
with float32's epsilon, which is what `layers.norm` uses on a bf16 input too); out in the
autocast dtype.
"""

import torch
import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_K = 64
BLOCK_O = 64
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
def _head_forward(x, first_weight, first_bias, norm_weight, last_weight, last_bias, out, M, K,
                  N, HIDDEN: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
                  BLOCK_O: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    hidden_index = tl.arange(0, HIDDEN)
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


@triton.jit
def _head_backward(x, first_weight, first_bias, norm_weight, last_weight, grad_out,
                   grad_hidden, activation_out, norm_partial, M, K, N, HIDDEN: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_O: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    hidden_index = tl.arange(0, HIDDEN)
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
    tl.store(norm_partial + tl.program_id(0) * HIDDEN + hidden_index,
             tl.sum(grad_normed * unit, 0))
    grad_unit = grad_normed * scale
    projection = tl.sum(grad_unit * unit, 1) / HIDDEN
    grad_h = inverse_rms[:, None] * (grad_unit - unit * projection[:, None])
    index = rows[:, None] * HIDDEN + hidden_index[None, :]
    mask = row_mask[:, None]
    tl.store(grad_hidden + index, grad_h.to(grad_hidden.dtype.element_ty), mask=mask)
    tl.store(activation_out + index, activation.to(activation_out.dtype.element_ty), mask=mask)


@torch.library.triton_op("yemong::fused_head", mutates_args=())
def fused_head(x: torch.Tensor, first_weight: torch.Tensor, first_bias: torch.Tensor,
               norm_weight: torch.Tensor, last_weight: torch.Tensor,
               last_bias: torch.Tensor) -> torch.Tensor:
    """x (M, K) contiguous 16-bit; first_weight (H, K), last_weight (N, H) in x's dtype; biases
    and the norm scale float32. Returns (M, N) in x's dtype."""
    rows, inputs = x.shape
    hidden, outputs = first_weight.shape[0], last_weight.shape[0]
    out = torch.empty(rows, outputs, dtype=x.dtype, device=x.device)
    torch.library.wrap_triton(_head_forward)[(triton.cdiv(rows, BLOCK_M),)](
        x, first_weight, first_bias, norm_weight, last_weight, last_bias, out, rows, inputs,
        outputs, HIDDEN=hidden, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_O=BLOCK_O,
        num_warps=8, num_stages=2)
    return out


@torch.library.triton_op("yemong::fused_head_backward", mutates_args=())
def fused_head_backward(x: torch.Tensor, first_weight: torch.Tensor, first_bias: torch.Tensor,
                        norm_weight: torch.Tensor, last_weight: torch.Tensor,
                        grad_out: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """grad_out (M, N) contiguous in x's dtype. Returns (dh (M, H), activation (M, H), norm
    scale partial sums (programs, H) float32)."""
    rows, inputs = x.shape
    hidden, outputs = first_weight.shape[0], last_weight.shape[0]
    programs = triton.cdiv(rows, BLOCK_M)
    grad_hidden = torch.empty(rows, hidden, dtype=x.dtype, device=x.device)
    activation = torch.empty(rows, hidden, dtype=x.dtype, device=x.device)
    partial = torch.empty(programs, hidden, dtype=torch.float32, device=x.device)
    torch.library.wrap_triton(_head_backward)[(programs,)](
        x, first_weight, first_bias, norm_weight, last_weight, grad_out, grad_hidden,
        activation, partial, rows, inputs, outputs, HIDDEN=hidden, BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K, BLOCK_O=BLOCK_O, num_warps=8, num_stages=2)
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
