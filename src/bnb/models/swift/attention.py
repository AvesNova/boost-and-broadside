"""Attention for tiny problems: Q = N + 1 queries over Q token keys plus M map keys, per view
and head (N = 10, M = 10: 11 queries, 21 keys, head width 64).

General kernels tile 128 x 128 and waste almost all of it here. This one gives each
program one (view, head), pads queries to 16 and keys to 32, and reads queries, token keys
and values from the fused projection and map keys and values from the map projection by
stride, so there is no concatenation, no transpose and no copy. The backward recomputes the
softmax in registers. Float32 inputs use IEEE (not TF32) dots.

    attention_heads(query, key, value, map_key, map_value)
        query, key, value (B, Q, H, W); map_key, map_value (B, M, H, W); any strides.
        Returns (B, Q, H, W) contiguous, softmax(q [k; mk]^T / sqrt(W)) [v; mv].
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _load_keys(key, map_key, base_key, base_map, token_stride, map_stride, keys_index, width,
               tokens, maps):
    """(KEYS, W) tile: rows [0, tokens) from key, [tokens, tokens + maps) from map_key."""
    from_tokens = keys_index < tokens
    from_maps = (keys_index >= tokens) & (keys_index < tokens + maps)
    token_part = tl.load(key + base_key + keys_index[:, None] * token_stride + width[None, :],
                         mask=from_tokens[:, None], other=0.0)
    map_part = tl.load(map_key + base_map + (keys_index[:, None] - tokens) * map_stride
                       + width[None, :], mask=from_maps[:, None], other=0.0)
    return token_part + map_part


@triton.jit
def _attention_forward(query, key, value, map_key, map_value, out, scale,
                       q_batch, q_token, q_head, k_batch, k_token, k_head,
                       v_batch, v_token, v_head, mk_batch, mk_token, mk_head,
                       mv_batch, mv_token, mv_head, tokens, maps, heads,
                       QUERIES: tl.constexpr, KEYS: tl.constexpr, WIDTH: tl.constexpr,
                       IEEE: tl.constexpr):
    view = tl.program_id(0)
    head = tl.program_id(1)
    rows = tl.arange(0, QUERIES)
    keys_index = tl.arange(0, KEYS)
    width = tl.arange(0, WIDTH)
    row_mask = rows < tokens
    q = tl.load(query + view * q_batch + head * q_head + rows[:, None] * q_token + width[None, :],
                mask=row_mask[:, None], other=0.0)
    k = _load_keys(key, map_key, view * k_batch + head * k_head, view * mk_batch + head * mk_head,
                   k_token, mk_token, keys_index, width, tokens, maps)
    v = _load_keys(value, map_value, view * v_batch + head * v_head,
                   view * mv_batch + head * mv_head, v_token, mv_token, keys_index, width,
                   tokens, maps)
    if IEEE:
        scores = tl.dot(q, tl.trans(k), input_precision="ieee")
    else:
        scores = tl.dot(q, tl.trans(k))
    scores = scores * scale
    valid = keys_index[None, :] < tokens + maps
    scores = tl.where(valid, scores, float("-inf"))
    peak = tl.max(scores, 1)
    weights = tl.exp(scores - peak[:, None])
    weights = weights / tl.sum(weights, 1)[:, None]
    if IEEE:
        result = tl.dot(weights, v, input_precision="ieee")
    else:
        result = tl.dot(weights.to(v.dtype), v)
    offsets = view * (tokens * heads * WIDTH) + rows[:, None] * (heads * WIDTH) + head * WIDTH \
        + width[None, :]
    tl.store(out + offsets, result.to(out.dtype.element_ty), mask=row_mask[:, None])


@triton.jit
def _attention_backward(query, key, value, map_key, map_value, out, grad_out, grad_query,
                        grad_key, grad_value, grad_map_key, grad_map_value, scale,
                        q_batch, q_token, q_head, k_batch, k_token, k_head,
                        v_batch, v_token, v_head, mk_batch, mk_token, mk_head,
                        mv_batch, mv_token, mv_head, tokens, maps, heads,
                        QUERIES: tl.constexpr, KEYS: tl.constexpr, WIDTH: tl.constexpr,
                        IEEE: tl.constexpr):
    view = tl.program_id(0)
    head = tl.program_id(1)
    rows = tl.arange(0, QUERIES)
    keys_index = tl.arange(0, KEYS)
    width = tl.arange(0, WIDTH)
    row_mask = rows < tokens
    q = tl.load(query + view * q_batch + head * q_head + rows[:, None] * q_token + width[None, :],
                mask=row_mask[:, None], other=0.0)
    k = _load_keys(key, map_key, view * k_batch + head * k_head, view * mk_batch + head * mk_head,
                   k_token, mk_token, keys_index, width, tokens, maps)
    v = _load_keys(value, map_value, view * v_batch + head * v_head,
                   view * mv_batch + head * mv_head, v_token, mv_token, keys_index, width,
                   tokens, maps)
    dense = view * (tokens * heads * WIDTH) + rows[:, None] * (heads * WIDTH) + head * WIDTH \
        + width[None, :]
    o = tl.load(out + dense, mask=row_mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(grad_out + dense, mask=row_mask[:, None], other=0.0)
    if IEEE:
        scores = tl.dot(q, tl.trans(k), input_precision="ieee")
    else:
        scores = tl.dot(q, tl.trans(k))
    scores = scores * scale
    valid = keys_index[None, :] < tokens + maps
    scores = tl.where(valid, scores, float("-inf"))
    peak = tl.max(scores, 1)
    weights = tl.exp(scores - peak[:, None])
    weights = weights / tl.sum(weights, 1)[:, None]
    weights = tl.where(row_mask[:, None], weights, 0.0)
    delta = tl.sum(do.to(tl.float32) * o, 1)
    if IEEE:
        grad_weights = tl.dot(do, tl.trans(v), input_precision="ieee")
        grad_v = tl.dot(tl.trans(weights), do, input_precision="ieee")
    else:
        grad_weights = tl.dot(do, tl.trans(v))
        grad_v = tl.dot(tl.trans(weights.to(do.dtype)), do)
    grad_scores = weights * (grad_weights - delta[:, None]) * scale
    if IEEE:
        grad_q = tl.dot(grad_scores, k, input_precision="ieee")
        grad_k = tl.dot(tl.trans(grad_scores), q, input_precision="ieee")
    else:
        grad_q = tl.dot(grad_scores.to(k.dtype), k)
        grad_k = tl.dot(tl.trans(grad_scores.to(q.dtype)), q)
    tl.store(grad_query + dense, grad_q.to(grad_query.dtype.element_ty), mask=row_mask[:, None])
    # gradients of keys and values: token rows to grad_key/value (B, Q, H, W), map rows to
    # grad_map_key/value (B, M, H, W), both contiguous
    from_tokens = keys_index < tokens
    from_maps = (keys_index >= tokens) & (keys_index < tokens + maps)
    token_offsets = view * (tokens * heads * WIDTH) + keys_index[:, None] * (heads * WIDTH) \
        + head * WIDTH + width[None, :]
    map_offsets = view * (maps * heads * WIDTH) + (keys_index[:, None] - tokens) * (heads * WIDTH) \
        + head * WIDTH + width[None, :]
    tl.store(grad_key + token_offsets, grad_k.to(grad_key.dtype.element_ty),
             mask=from_tokens[:, None])
    tl.store(grad_value + token_offsets, grad_v.to(grad_value.dtype.element_ty),
             mask=from_tokens[:, None])
    tl.store(grad_map_key + map_offsets, grad_k.to(grad_map_key.dtype.element_ty),
             mask=from_maps[:, None])
    tl.store(grad_map_value + map_offsets, grad_v.to(grad_map_value.dtype.element_ty),
             mask=from_maps[:, None])


def _strides(tensor):
    return tensor.stride(0), tensor.stride(1), tensor.stride(2)


def _launch_shape(query, map_key):
    views, tokens, heads, width = query.shape
    maps = map_key.shape[1]
    queries = max(16, triton.next_power_of_2(tokens))
    keys = max(16, triton.next_power_of_2(tokens + maps))
    return views, tokens, heads, width, maps, queries, keys


@torch.library.custom_op("swift::attention_heads", mutates_args=())
def attention_heads(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                    map_key: torch.Tensor, map_value: torch.Tensor) -> torch.Tensor:
    views, tokens, heads, width, maps, queries, keys = _launch_shape(query, map_key)
    out = torch.empty(views, tokens, heads, width, dtype=query.dtype, device=query.device)
    for tensor in (query, key, value, map_key, map_value):
        assert tensor.stride(-1) == 1
    _attention_forward[(views, heads)](
        query, key, value, map_key, map_value, out, width**-0.5,
        *_strides(query), *_strides(key), *_strides(value), *_strides(map_key),
        *_strides(map_value), tokens, maps, heads, QUERIES=queries, KEYS=keys, WIDTH=width,
        IEEE=query.dtype == torch.float32, num_warps=4)
    return out


@attention_heads.register_fake
def _(query, key, value, map_key, map_value):
    return torch.empty(query.shape, dtype=query.dtype, device=query.device)


@torch.library.custom_op("swift::attention_heads_backward", mutates_args=())
def attention_heads_backward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                             map_key: torch.Tensor, map_value: torch.Tensor, out: torch.Tensor,
                             grad_out: torch.Tensor) -> list[torch.Tensor]:
    views, tokens, heads, width, maps, queries, keys = _launch_shape(query, map_key)
    grad_out = grad_out.contiguous()
    grad_query = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    grad_key = torch.empty(query.shape, dtype=key.dtype, device=query.device)
    grad_value = torch.empty(query.shape, dtype=value.dtype, device=query.device)
    grad_map_key = torch.empty(map_key.shape, dtype=map_key.dtype, device=query.device)
    grad_map_value = torch.empty(map_key.shape, dtype=map_value.dtype, device=query.device)
    _attention_backward[(views, heads)](
        query, key, value, map_key, map_value, out, grad_out, grad_query, grad_key, grad_value,
        grad_map_key, grad_map_value, width**-0.5,
        *_strides(query), *_strides(key), *_strides(value), *_strides(map_key),
        *_strides(map_value), tokens, maps, heads, QUERIES=queries, KEYS=keys, WIDTH=width,
        IEEE=query.dtype == torch.float32, num_warps=4)
    return [grad_query, grad_key, grad_value, grad_map_key, grad_map_value]


@attention_heads_backward.register_fake
def _(query, key, value, map_key, map_value, out, grad_out):
    return [torch.empty(query.shape, dtype=t.dtype, device=query.device)
            for t in (query, key, value)] + [
        torch.empty(map_key.shape, dtype=t.dtype, device=query.device)
        for t in (map_key, map_value)]


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs, output)


def _backward(ctx, grad_out):
    return tuple(attention_heads_backward(*ctx.saved_tensors, grad_out))


attention_heads.register_autograd(_backward, setup_context=_setup_context)


def supported(query: torch.Tensor) -> bool:
    return query.is_cuda and query.dtype in (torch.float32, torch.bfloat16, torch.float16) \
        and query.shape[-1] in (16, 32, 64, 128)
