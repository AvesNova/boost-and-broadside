"""The Triton kernels against plain torch: the scan, the small attention, the gated MLP, the
fused temporal mix (gates + scan + output gate) and the fused linear-norm-gelu-linear head."""

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


def block_gradients(block, dtype, fused, autocast, inputs, weight):
    """Outputs and every gradient of a TemporalBlock in `dtype` (float64 copies)."""
    x, state, buffer, starts = inputs
    network.FUSED_TEMPORAL["enabled"] = fused
    try:
        module = block.to(dtype)
        module.zero_grad(set_to_none=True)
        x = x.detach().to(dtype).requires_grad_()
        state = state.detach().to(dtype).requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out, final, _ = module(x, state, buffer.to(dtype), starts)
        ((out.double() * weight).sum() + 0.3 * final.double().sum()).backward()
    finally:
        network.FUSED_TEMPORAL["enabled"] = True
    return [out.double(), final.double(), x.grad.double(), state.grad.double()] + [
        parameter.grad.double() for parameter in module.parameters()]


@pytest.mark.parametrize("autocast,tolerance", [(False, 2e-5), (True, 3e-2)])
def test_fused_temporal_mix_matches_the_float64_block(autocast, tolerance):
    """Forward and every gradient, with resets mid-segment for one view and for one token."""
    torch.manual_seed(0)
    steps, views, queries, width = 37, 3, 11, 128  # 37: a partial last time chunk
    block = network.TemporalBlock(width).cuda()
    with torch.no_grad():
        block.log_decay.uniform_(-2, 5)
    x = torch.randn(steps, views, queries, width, device="cuda")
    state = torch.randn(views, queries, width, device="cuda")
    buffer = torch.randn(views, queries, 3, width, device="cuda")
    starts = torch.zeros(steps, views, queries, dtype=torch.bool, device="cuda")
    starts[20, 1] = True
    starts[5, :, 3] = True
    weight = torch.randn(steps, views, queries, width, device="cuda", dtype=torch.float64)
    inputs = (x, state, buffer, starts)
    reference = block_gradients(block, torch.float64, False, False, inputs, weight)
    fused = block_gradients(block, torch.float32, True, autocast, inputs, weight)
    for actual, expected in zip(fused, reference, strict=True):
        assert (actual - expected).abs().max() < tolerance * expected.abs().max()


@pytest.mark.parametrize("inputs,hidden,outputs,rows", [(128, 256, 490, 3000), (553, 256, 128, 77),
                                                         (128, 128, 154, 1000)])
def test_fused_head_matches_the_float64_module(inputs, hidden, outputs, rows):
    """bf16 autocast: the fused head is as close to float64 as the module path is."""
    torch.manual_seed(0)
    head = network.MLPHead(inputs, hidden, outputs).cuda().double()
    head.initialise(1.0)
    with torch.no_grad():
        head.norm.weight.uniform_(0.5, 1.5)
    x = torch.randn(rows, inputs, device="cuda", dtype=torch.float64)
    weight = torch.randn(rows, outputs, device="cuda", dtype=torch.float64)

    def run(dtype, fused, autocast):
        network.FUSED_HEAD["enabled"] = fused
        try:
            module = head.to(dtype)
            module.zero_grad(set_to_none=True)
            source = x.to(dtype).clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                out = module(source)
            (out.double() * weight).sum().backward()
        finally:
            network.FUSED_HEAD["enabled"] = True
        return [out.double(), source.grad.double()] + [p.grad.double()
                                                       for p in module.parameters()]

    reference = run(torch.float64, False, False)
    fused = run(torch.float32, True, True)
    plain = run(torch.float32, False, True)
    for actual, module_path, expected in zip(fused, plain, reference, strict=True):
        scale = expected.abs().max()
        assert (actual - expected).abs().max() < max(
            1.5 * (module_path - expected).abs().max(), 2e-3 * scale)
