"""The Triton kernels against plain torch: the scan, the small attention, the gated MLP."""

import pytest
import torch

from bnb.models.swift import network
from bnb.models.swift.attention import attention_heads
from bnb.models.swift.scan import linear_scan, sequential_scan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def gradients(function, inputs, weight):
    for tensor in inputs:
        tensor.grad = None
    out = function(*inputs)
    (out.double() * weight).sum().backward()
    return out.detach().double(), [tensor.grad.double() for tensor in inputs]


def test_scan_forward_and_backward():
    torch.manual_seed(0)
    decay = torch.rand(37, 6, 11, 128, device="cuda", requires_grad=True)
    contribution = torch.randn(37, 6, 11, 128, device="cuda", requires_grad=True)
    initial = torch.randn(6, 11, 128, device="cuda", requires_grad=True)
    weight = torch.randn(37, 6, 11, 128, device="cuda", dtype=torch.float64)
    inputs = (decay, contribution, initial)
    out, grads = gradients(linear_scan, inputs, weight)
    expected, expected_grads = gradients(sequential_scan, inputs, weight)
    assert (out - expected).abs().max() < 1e-5
    for actual, reference in zip(grads, expected_grads, strict=True):
        assert (actual - reference).abs().max() < 1e-4


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 1e-5), (torch.bfloat16, 3e-2)])
def test_small_attention_matches_rotary_softmax_attention(dtype, tolerance):
    torch.manual_seed(0)
    views, queries, maps, heads, width, pairs = 53, 11, 10, 2, 64, 20
    fused = torch.randn(views, queries, 3 * heads * width, device="cuda", dtype=dtype,
                        requires_grad=True)
    memory = torch.randn(views, maps, 2 * heads * width, device="cuda", dtype=dtype,
                         requires_grad=True)
    angles = torch.randn(views, queries + maps, pairs, device="cuda") * 3
    token_cos, token_sin = angles[:, :queries].cos(), angles[:, :queries].sin()
    map_cos, map_sin = angles[:, queries:].cos(), angles[:, queries:].sin()

    def split(fused, memory):
        return [t.unflatten(-1, (heads, width)) for t in fused.chunk(3, -1) + memory.chunk(2, -1)]

    def triton_attention(fused, memory):
        return attention_heads(*split(fused, memory), token_cos, token_sin, map_cos, map_sin)

    def reference(fused, memory):
        q, k, v, mk, mv = (t.double() for t in split(fused, memory))
        tables = [t.double().unsqueeze(-2) for t in (token_cos, token_sin, map_cos, map_sin)]
        q = network.apply_rotary(q, *tables[:2])
        k = torch.cat([network.apply_rotary(k, *tables[:2]),
                       network.apply_rotary(mk, *tables[2:])], 1)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), torch.cat([v, mv], 1).transpose(1, 2))
        return out.transpose(1, 2)

    weight = torch.randn(views, queries, heads, width, device="cuda", dtype=torch.float64)
    out, grads = gradients(triton_attention, (fused, memory), weight)
    expected, expected_grads = gradients(reference, (fused, memory), weight)
    scale = expected.abs().max()
    assert (out - expected).abs().max() < tolerance * scale
    for actual, reference_grad in zip(grads, expected_grads, strict=True):
        assert (actual - reference_grad).abs().max() < tolerance * reference_grad.abs().max()


def test_spatial_block_with_small_attention():
    torch.manual_seed(0)
    block = network.SpatialBlock(128, 2, 64).cuda()
    x = torch.randn(40, 11, 128, device="cuda")
    memory = torch.randn(40, 10, 64, device="cuda")
    angles = torch.randn(40, 21, 1, 20, device="cuda")
    tokens = (angles[:, :11].cos(), angles[:, :11].sin())
    maps = (angles[:, 11:].cos(), angles[:, 11:].sin())
    fast = block(x, memory, tokens, maps)
    try:
        network.ATTENTION["kernel"] = "sdpa"
        plain = block(x, memory, tokens, maps)
    finally:
        network.ATTENTION["kernel"] = "triton"
    assert (fast - plain).abs().max() < 1e-4 * plain.abs().max()


def test_fused_gated_mlp_matches_unfused():
    torch.manual_seed(0)
    mlp = network.GatedMLP(128, 512).cuda()
    x = torch.randn(700, 11, 128, device="cuda", requires_grad=True)
    weight = torch.randn(700, 11, 128, device="cuda", dtype=torch.float64)
    results = []
    for fused in (False, True):
        network.FUSED_MLP["enabled"] = fused
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, grads = gradients(mlp, (x,), weight)
        finally:
            network.FUSED_MLP["enabled"] = True
        results.append((out, grads[0], mlp.gate.weight.grad.double().clone()))
        mlp.zero_grad()
    for plain, fused in zip(*results, strict=True):
        assert (plain - fused).abs().max() < 2e-2 * plain.abs().max()
