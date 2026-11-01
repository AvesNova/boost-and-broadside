"""The gated MLP's hidden layer, h = gelu(x W_gate^T) * (x W_up^T), as one Triton kernel.

Both GEMMs accumulate in the same program and the product is formed in registers, so the
(rows, 2 hidden) pre-activations are never written. The backward recomputes them in a
second kernel that writes d(gate, up) (rows, 2 hidden); the input and weight gradients are
then two ordinary GEMMs. Only x is saved for the backward. For 16-bit inputs (autocast);
float32 keeps the cuBLAS path.
"""

import torch
import triton
import triton.language as tl

CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
]


@triton.jit
def _gate_up(x, gate_weight, up_weight, rows, columns, M, N, K, BLOCK_M: tl.constexpr,
             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, K, BLOCK_K):
        inner = start + tl.arange(0, BLOCK_K)
        tile = tl.load(x + rows[:, None] * K + inner[None, :],
                       mask=(rows[:, None] < M) & (inner[None, :] < K), other=0.0)
        weight_mask = (columns[None, :] < N) & (inner[:, None] < K)
        gate_tile = tl.load(gate_weight + columns[None, :] * K + inner[:, None], mask=weight_mask,
                            other=0.0)
        up_tile = tl.load(up_weight + columns[None, :] * K + inner[:, None], mask=weight_mask,
                          other=0.0)
        gate = tl.dot(tile, gate_tile, gate)
        up = tl.dot(tile, up_tile, up)
    return gate, up


@triton.autotune(configs=CONFIGS, key=["N", "K"])
@triton.jit
def _gated_forward(x, gate_weight, up_weight, out, M, N, K, BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    gate, up = _gate_up(x, gate_weight, up_weight, rows, columns, M, N, K, BLOCK_M, BLOCK_N,
                        BLOCK_K)
    hidden = 0.5 * gate * (1 + tl.math.erf(gate * 0.7071067811865476)) * up
    tl.store(out + rows[:, None] * N + columns[None, :], hidden.to(out.dtype.element_ty),
             mask=(rows[:, None] < M) & (columns[None, :] < N))


@triton.autotune(configs=CONFIGS, key=["N", "K"])
@triton.jit
def _gated_backward(x, gate_weight, up_weight, grad_hidden, grad_gate_up, M, N, K,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    gate, up = _gate_up(x, gate_weight, up_weight, rows, columns, M, N, K, BLOCK_M, BLOCK_N,
                        BLOCK_K)
    mask = (rows[:, None] < M) & (columns[None, :] < N)
    grad = tl.load(grad_hidden + rows[:, None] * N + columns[None, :], mask=mask,
                   other=0.0).to(tl.float32)
    cumulative = 0.5 * (1 + tl.math.erf(gate * 0.7071067811865476))
    density = tl.exp(-0.5 * gate * gate) * 0.3989422804014327
    grad_gate = grad * up * (cumulative + gate * density)
    grad_up = grad * gate * cumulative
    base = grad_gate_up + rows[:, None] * (2 * N) + columns[None, :]
    tl.store(base, grad_gate.to(grad_gate_up.dtype.element_ty), mask=mask)
    tl.store(base + N, grad_up.to(grad_gate_up.dtype.element_ty), mask=mask)


def _grid(M, N):
    return lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))


@torch.library.custom_op("swift::gated_hidden", mutates_args=())
def gated_hidden(x: torch.Tensor, gate_weight: torch.Tensor,
                 up_weight: torch.Tensor) -> torch.Tensor:
    """x (M, K) contiguous; weights (N, K) -> (M, N)."""
    M, K = x.shape
    N = gate_weight.shape[0]
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    _gated_forward[_grid(M, N)](x, gate_weight, up_weight, out, M, N, K)
    return out


@gated_hidden.register_fake
def _(x, gate_weight, up_weight):
    return x.new_empty(x.shape[0], gate_weight.shape[0])


@torch.library.custom_op("swift::gated_hidden_backward", mutates_args=())
def gated_hidden_backward(x: torch.Tensor, gate_weight: torch.Tensor, up_weight: torch.Tensor,
                          grad_hidden: torch.Tensor) -> torch.Tensor:
    """d(gate, up) (M, 2N) for the upstream gradient grad_hidden (M, N)."""
    M, K = x.shape
    N = gate_weight.shape[0]
    out = torch.empty(M, 2 * N, dtype=x.dtype, device=x.device)
    _gated_backward[_grid(M, N)](x, gate_weight, up_weight, grad_hidden.contiguous(), out, M,
                                 N, K)
    return out


@gated_hidden_backward.register_fake
def _(x, gate_weight, up_weight, grad_hidden):
    return x.new_empty(x.shape[0], 2 * gate_weight.shape[0])


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad_hidden):
    x, gate_weight, up_weight = ctx.saved_tensors
    grad_gate_up = gated_hidden_backward(x, gate_weight, up_weight, grad_hidden)
    weights = torch.cat([gate_weight, up_weight])
    grad_x = grad_gate_up @ weights
    grad_weights = grad_gate_up.t() @ x
    grad_gate, grad_up = grad_weights.chunk(2)
    return grad_x, grad_gate, grad_up


gated_hidden.register_autograd(_backward, setup_context=_setup_context)


def fused_hidden(x: torch.Tensor, gate_weight: torch.Tensor, up_weight: torch.Tensor):
    """gelu(x W_g^T) * (x W_u^T) for x (..., K) under bf16/fp16 autocast on CUDA, else None."""
    if not (x.is_cuda and torch.is_autocast_enabled("cuda")):
        return None
    dtype = torch.get_autocast_dtype("cuda")
    if dtype not in (torch.bfloat16, torch.float16) or x.shape[-1] % 16:
        return None
    flat = x.reshape(-1, x.shape[-1]).to(dtype).contiguous()
    hidden = gated_hidden(flat, gate_weight.to(dtype), up_weight.to(dtype))
    return hidden.view(*x.shape[:-1], gate_weight.shape[0])

