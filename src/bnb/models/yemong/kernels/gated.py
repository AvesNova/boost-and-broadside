"""The gated MLP's hidden layer, h = gelu(x W_gate^T) * (x W_up^T), as one Triton kernel
(16-bit). The plain twin is `GatedMLP.forward`'s `gate_up` product and `gelu(gate) * up`.

Both products accumulate in the same program and the product is formed in registers, so the
(rows, 2 hidden) pre-activations are never written. The backward recomputes them in a
second kernel that writes d(gate, up) (rows, 2 hidden); the input and weight gradients are
then two ordinary GEMMs. Only x is saved for the backward. W_gate and W_up are the row
blocks of `GatedMLP.gate_up.weight` (gate rows first), read in place.

The tiles are autotuned per shape. Every configuration gives the same bits (the test checks
it), so the configuration a process picks by timing does not change the result. Float32
keeps the plain path (a strict-fp32 fused kernel lost to cuBLAS in the blind race).
"""

import torch
import triton
import triton.language as tl

CONFIGS = [triton.Config({"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k}, num_warps=warps,
                         num_stages=stages)
           for m, n, k, warps, stages in ((64, 64, 64, 4, 3), (128, 64, 64, 4, 2),
                                          (64, 128, 64, 4, 2), (128, 128, 32, 8, 2))]
SQRT_HALF = tl.constexpr(0.7071067811865476)
INVERSE_SQRT_TAU = tl.constexpr(0.3989422804014327)


@triton.jit
def _gate_up(x, weight, rows, columns, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             BLOCK_K: tl.constexpr):
    gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, K, BLOCK_K):
        inner = start + tl.arange(0, BLOCK_K)
        tile = tl.load(x + rows[:, None] * K + inner[None, :],
                       mask=(rows[:, None] < M) & (inner[None, :] < K), other=0.0)
        weight_mask = (columns[None, :] < N) & (inner[:, None] < K)
        gate_tile = tl.load(weight + columns[None, :] * K + inner[:, None], mask=weight_mask,
                            other=0.0)
        up_tile = tl.load(weight + (N + columns[None, :]) * K + inner[:, None],
                          mask=weight_mask, other=0.0)
        gate = tl.dot(tile, gate_tile, gate)
        up = tl.dot(tile, up_tile, up)
    return gate, up


@triton.autotune(configs=CONFIGS, key=["M", "N", "K"], cache_results=True)
@triton.jit
def _gated_forward(x, weight, out, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    gate, up = _gate_up(x, weight, rows, columns, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    hidden = 0.5 * gate * (1 + tl.math.erf(gate * SQRT_HALF)) * up
    tl.store(out + rows[:, None] * N + columns[None, :], hidden.to(out.dtype.element_ty),
             mask=(rows[:, None] < M) & (columns[None, :] < N))


@triton.autotune(configs=CONFIGS, key=["M", "N", "K"], cache_results=True)
@triton.jit
def _gated_backward(x, weight, grad_hidden, grad_gate_up, M, N, K, BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    gate, up = _gate_up(x, weight, rows, columns, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    mask = (rows[:, None] < M) & (columns[None, :] < N)
    grad = tl.load(grad_hidden + rows[:, None] * N + columns[None, :], mask=mask,
                   other=0.0).to(tl.float32)
    cumulative = 0.5 * (1 + tl.math.erf(gate * SQRT_HALF))
    density = tl.exp(-0.5 * gate * gate) * INVERSE_SQRT_TAU
    grad_gate = grad * up * (cumulative + gate * density)
    grad_up = grad * gate * cumulative
    base = grad_gate_up + rows[:, None] * (2 * N) + columns[None, :]
    tl.store(base, grad_gate.to(grad_gate_up.dtype.element_ty), mask=mask)
    tl.store(base + N, grad_up.to(grad_gate_up.dtype.element_ty), mask=mask)


def _launch(kernel, M, N, *arguments):
    def grid(meta):
        return triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"])

    torch.library.wrap_triton(kernel)[grid](*arguments)


@torch.library.triton_op("yemong::gated_hidden", mutates_args=())
def _gated_hidden(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x (M, K) contiguous; weight (2N, K) contiguous, gate rows first -> (M, N)."""
    M, K = x.shape
    N = weight.shape[0] // 2
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    _launch(_gated_forward, M, N, x, weight, out, M, N, K)
    return out


@torch.library.triton_op("yemong::gated_hidden_backward", mutates_args=())
def _gated_hidden_backward(x: torch.Tensor, weight: torch.Tensor,
                           grad_hidden: torch.Tensor) -> torch.Tensor:
    """d(gate, up) (M, 2N) for the upstream gradient grad_hidden (M, N)."""
    M, K = x.shape
    N = weight.shape[0] // 2
    out = torch.empty(M, 2 * N, dtype=x.dtype, device=x.device)
    _launch(_gated_backward, M, N, x, weight, grad_hidden.contiguous(), out, M, N, K)
    return out


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad_hidden):
    x, weight = ctx.saved_tensors
    grad_gate_up = _gated_hidden_backward(x, weight, grad_hidden.to(x.dtype))
    return grad_gate_up @ weight, grad_gate_up.t() @ x


_gated_hidden.register_autograd(_backward, setup_context=_setup_context)


def applies(inputs: torch.Tensor) -> bool:
    """CUDA under 16-bit autocast, the width a multiple of 16."""
    return (inputs.is_cuda and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") in (torch.bfloat16, torch.float16)
            and inputs.shape[-1] % 16 == 0)


def gated_hidden(inputs: torch.Tensor, gate_up: torch.Tensor) -> torch.Tensor:
    """gelu(x W_gate^T) * (x W_up^T) (..., N) for x (..., K) and gate_up (2N, K), in the
    autocast dtype (`applies`)."""
    dtype = torch.get_autocast_dtype("cuda")
    flat = inputs.reshape(-1, inputs.shape[-1]).to(dtype).contiguous()
    hidden = _gated_hidden(flat, gate_up.to(dtype).contiguous())
    return hidden.view(*inputs.shape[:-1], gate_up.shape[0] // 2)
