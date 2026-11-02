"""Attention for tiny problems, rotary included: Q = N + 1 queries over Q token keys plus M map
keys, per view and head (N = 10, M = 10: 11 queries, 21 keys, head width 64).

General kernels tile 128 x 128 and waste almost all of it here. This one gives each
program one (view, head), pads queries to 16 and keys to 32, and reads queries, token keys
and values from the fused projection and map keys and values from the map projection by
stride: no concatenation, no transpose, no copy. The rotary rotation of queries and keys
(spec 6.2: interleaved pairs (2i, 2i + 1) of the first 2R dims) happens in registers, in
float32: a head is loaded as its even and odd lanes, rotated, and scored as
q_even k_even + q_odd k_odd. The backward recomputes the softmax and rotates the query and
key gradients back. Float32 inputs use IEEE (not TF32) dots.

    attention_heads(query, key, value, map_key, map_value, token_cos, token_sin, map_cos,
                    map_sin)
        query, key, value (B, Q, H, W) and map_key, map_value (B, M, H, W), any strides with
        a unit last stride, not yet rotated; token tables (B, Q, R), map tables (B, M, R).
        Returns (B, Q, H, W) contiguous.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate(even, odd, cos, sin):
    return even * cos - odd * sin, even * sin + odd * cos


@triton.jit
def _unrotate(even, odd, cos, sin):
    return even * cos + odd * sin, odd * cos - even * sin


@triton.jit
def _load_tables(cos, sin, base, row_stride, rows, lanes, row_mask, pairs):
    mask = row_mask[:, None] & (lanes[None, :] < pairs)
    offsets = base + rows[:, None] * row_stride + lanes[None, :]
    return (tl.load(cos + offsets, mask=mask, other=1.0).to(tl.float32),
            tl.load(sin + offsets, mask=mask, other=0.0).to(tl.float32))


@triton.jit
def _load_halves(pointer, base, row_stride, rows, lanes, row_mask):
    offsets = base + rows[:, None] * row_stride + 2 * lanes[None, :]
    even = tl.load(pointer + offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
    odd = tl.load(pointer + offsets + 1, mask=row_mask[:, None], other=0.0).to(tl.float32)
    return even, odd


@triton.jit
def _queries_and_keys(query, key, map_key, token_cos, token_sin, map_cos, map_sin, view, head,
                      q_batch, q_token, q_head, k_batch, k_token, k_head, mk_batch, mk_token,
                      mk_head, t_batch, t_token, m_batch, m_token, tokens, maps, pairs,
                      QUERIES: tl.constexpr, KEYS: tl.constexpr, HALF: tl.constexpr):
    """Rotated (even, odd) lanes of the queries (QUERIES, HALF) and keys (KEYS, HALF), with
    their tables, float32."""
    rows = tl.arange(0, QUERIES)
    keys_index = tl.arange(0, KEYS)
    lanes = tl.arange(0, HALF)
    row_mask = rows < tokens
    query_cos, query_sin = _load_tables(token_cos, token_sin, view * t_batch, t_token, rows,
                                        lanes, row_mask, pairs)
    even, odd = _load_halves(query, view * q_batch + head * q_head, q_token, rows, lanes,
                             row_mask)
    query_even, query_odd = _rotate(even, odd, query_cos, query_sin)

    from_tokens = keys_index < tokens
    from_maps = (keys_index >= tokens) & (keys_index < tokens + maps)
    map_rows = keys_index - tokens
    token_cos_k, token_sin_k = _load_tables(token_cos, token_sin, view * t_batch, t_token,
                                            keys_index, lanes, from_tokens, pairs)
    map_cos_k, map_sin_k = _load_tables(map_cos, map_sin, view * m_batch, m_token, map_rows,
                                        lanes, from_maps, pairs)
    key_cos = tl.where(from_maps[:, None], map_cos_k, token_cos_k)
    key_sin = tl.where(from_maps[:, None], map_sin_k, token_sin_k)
    token_even, token_odd = _load_halves(key, view * k_batch + head * k_head, k_token,
                                         keys_index, lanes, from_tokens)
    map_even, map_odd = _load_halves(map_key, view * mk_batch + head * mk_head, mk_token,
                                     map_rows, lanes, from_maps)
    key_even, key_odd = _rotate(token_even + map_even, token_odd + map_odd, key_cos, key_sin)
    return (query_even, query_odd, query_cos, query_sin, key_even, key_odd, key_cos, key_sin)


@triton.jit
def _load_values(value, map_value, view, head, v_batch, v_token, v_head, mv_batch, mv_token,
                 mv_head, tokens, maps, KEYS: tl.constexpr, WIDTH: tl.constexpr):
    keys_index = tl.arange(0, KEYS)
    width = tl.arange(0, WIDTH)
    from_tokens = keys_index < tokens
    from_maps = (keys_index >= tokens) & (keys_index < tokens + maps)
    token_part = tl.load(value + view * v_batch + head * v_head + keys_index[:, None] * v_token
                         + width[None, :], mask=from_tokens[:, None], other=0.0)
    map_part = tl.load(map_value + view * mv_batch + head * mv_head
                       + (keys_index[:, None] - tokens) * mv_token + width[None, :],
                       mask=from_maps[:, None], other=0.0)
    return token_part + map_part


@triton.jit
def _dot(a, b, IEEE: tl.constexpr):
    if IEEE:
        return tl.dot(a, b, input_precision="ieee")
    return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))


@triton.jit
def _weights(query_even, query_odd, key_even, key_odd, scale, tokens, maps,
             QUERIES: tl.constexpr, KEYS: tl.constexpr, IEEE: tl.constexpr):
    scores = _dot(query_even, tl.trans(key_even), IEEE) + _dot(query_odd, tl.trans(key_odd), IEEE)
    scores = scores * scale
    keys_index = tl.arange(0, KEYS)
    scores = tl.where(keys_index[None, :] < tokens + maps, scores, float("-inf"))
    peak = tl.max(scores, 1)
    weights = tl.exp(scores - peak[:, None])
    weights = weights / tl.sum(weights, 1)[:, None]
    rows = tl.arange(0, QUERIES)
    return tl.where((rows < tokens)[:, None], weights, 0.0)


@triton.jit
def _attention_forward(query, key, value, map_key, map_value, token_cos, token_sin, map_cos,
                       map_sin, out, scale,
                       q_batch, q_token, q_head, k_batch, k_token, k_head,
                       v_batch, v_token, v_head, mk_batch, mk_token, mk_head,
                       mv_batch, mv_token, mv_head, t_batch, t_token, m_batch, m_token,
                       tokens, maps, heads, pairs,
                       QUERIES: tl.constexpr, KEYS: tl.constexpr, WIDTH: tl.constexpr,
                       IEEE: tl.constexpr):
    view = tl.program_id(0)
    head = tl.program_id(1)
    query_even, query_odd, _, _, key_even, key_odd, _, _ = _queries_and_keys(
        query, key, map_key, token_cos, token_sin, map_cos, map_sin, view, head,
        q_batch, q_token, q_head, k_batch, k_token, k_head, mk_batch, mk_token, mk_head,
        t_batch, t_token, m_batch, m_token, tokens, maps, pairs, QUERIES, KEYS, WIDTH // 2)
    weights = _weights(query_even, query_odd, key_even, key_odd, scale, tokens, maps, QUERIES,
                       KEYS, IEEE)
    v = _load_values(value, map_value, view, head, v_batch, v_token, v_head, mv_batch,
                     mv_token, mv_head, tokens, maps, KEYS, WIDTH)
    result = _dot(weights, v.to(tl.float32), IEEE)
    rows = tl.arange(0, QUERIES)
    width = tl.arange(0, WIDTH)
    offsets = view * (tokens * heads * WIDTH) + rows[:, None] * (heads * WIDTH) + head * WIDTH \
        + width[None, :]
    tl.store(out + offsets, result.to(out.dtype.element_ty), mask=(rows < tokens)[:, None])


@triton.jit
def _store_halves(pointer, base, row_stride, rows, lanes, mask, even, odd):
    offsets = base + rows[:, None] * row_stride + 2 * lanes[None, :]
    tl.store(pointer + offsets, even.to(pointer.dtype.element_ty), mask=mask[:, None])
    tl.store(pointer + offsets + 1, odd.to(pointer.dtype.element_ty), mask=mask[:, None])


@triton.jit
def _attention_backward(query, key, value, map_key, map_value, token_cos, token_sin, map_cos,
                        map_sin, out, grad_out, grad_query, grad_key, grad_value, grad_map_key,
                        grad_map_value, scale,
                        q_batch, q_token, q_head, k_batch, k_token, k_head,
                        v_batch, v_token, v_head, mk_batch, mk_token, mk_head,
                        mv_batch, mv_token, mv_head, t_batch, t_token, m_batch, m_token,
                        tokens, maps, heads, pairs,
                        QUERIES: tl.constexpr, KEYS: tl.constexpr, WIDTH: tl.constexpr,
                        IEEE: tl.constexpr):
    view = tl.program_id(0)
    head = tl.program_id(1)
    HALF: tl.constexpr = WIDTH // 2
    (query_even, query_odd, query_cos, query_sin, key_even, key_odd, key_cos,
     key_sin) = _queries_and_keys(
        query, key, map_key, token_cos, token_sin, map_cos, map_sin, view, head,
        q_batch, q_token, q_head, k_batch, k_token, k_head, mk_batch, mk_token, mk_head,
        t_batch, t_token, m_batch, m_token, tokens, maps, pairs, QUERIES, KEYS, HALF)
    weights = _weights(query_even, query_odd, key_even, key_odd, scale, tokens, maps, QUERIES,
                       KEYS, IEEE)
    v = _load_values(value, map_value, view, head, v_batch, v_token, v_head, mv_batch,
                     mv_token, mv_head, tokens, maps, KEYS, WIDTH).to(tl.float32)
    rows = tl.arange(0, QUERIES)
    keys_index = tl.arange(0, KEYS)
    width = tl.arange(0, WIDTH)
    lanes = tl.arange(0, HALF)
    row_mask = rows < tokens
    dense = view * (tokens * heads * WIDTH) + rows[:, None] * (heads * WIDTH) + head * WIDTH \
        + width[None, :]
    o = tl.load(out + dense, mask=row_mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(grad_out + dense, mask=row_mask[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(do * o, 1)
    grad_weights = _dot(do, tl.trans(v), IEEE)
    grad_v = _dot(tl.trans(weights), do, IEEE)
    grad_scores = weights * (grad_weights - delta[:, None]) * scale
    grad_query_even, grad_query_odd = _unrotate(_dot(grad_scores, key_even, IEEE),
                                                _dot(grad_scores, key_odd, IEEE),
                                                query_cos, query_sin)
    grad_key_even, grad_key_odd = _unrotate(_dot(tl.trans(grad_scores), query_even, IEEE),
                                            _dot(tl.trans(grad_scores), query_odd, IEEE),
                                            key_cos, key_sin)
    dense_rows = heads * WIDTH
    head_base = head * WIDTH
    _store_halves(grad_query, view * tokens * dense_rows + head_base, dense_rows, rows, lanes,
                  row_mask, grad_query_even, grad_query_odd)
    from_tokens = keys_index < tokens
    from_maps = (keys_index >= tokens) & (keys_index < tokens + maps)
    _store_halves(grad_key, view * tokens * dense_rows + head_base, dense_rows, keys_index,
                  lanes, from_tokens, grad_key_even, grad_key_odd)
    _store_halves(grad_map_key, view * maps * dense_rows + head_base, dense_rows,
                  keys_index - tokens, lanes, from_maps, grad_key_even, grad_key_odd)
    token_offsets = view * tokens * dense_rows + keys_index[:, None] * dense_rows + head_base \
        + width[None, :]
    map_offsets = view * maps * dense_rows + (keys_index[:, None] - tokens) * dense_rows \
        + head_base + width[None, :]
    tl.store(grad_value + token_offsets, grad_v.to(grad_value.dtype.element_ty),
             mask=from_tokens[:, None])
    tl.store(grad_map_value + map_offsets, grad_v.to(grad_map_value.dtype.element_ty),
             mask=from_maps[:, None])


def _strides(tensor):
    return tensor.stride(0), tensor.stride(1), tensor.stride(2)


def _arguments(query, key, value, map_key, map_value, token_cos, map_cos):
    views, tokens, heads, width = query.shape
    maps = map_key.shape[1]
    return dict(
        scalars=(width**-0.5, *_strides(query), *_strides(key), *_strides(value),
                 *_strides(map_key), *_strides(map_value), token_cos.stride(0),
                 token_cos.stride(1), map_cos.stride(0), map_cos.stride(1), tokens, maps, heads,
                 token_cos.shape[-1]),
        meta=dict(QUERIES=max(16, triton.next_power_of_2(tokens)),
                  KEYS=max(16, triton.next_power_of_2(tokens + maps)), WIDTH=width,
                  IEEE=query.dtype == torch.float32, num_warps=4),
        grid=(views, heads))


@torch.library.triton_op("swift::attention_heads", mutates_args=())
def attention_heads(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                    map_key: torch.Tensor, map_value: torch.Tensor, token_cos: torch.Tensor,
                    token_sin: torch.Tensor, map_cos: torch.Tensor,
                    map_sin: torch.Tensor) -> torch.Tensor:
    token_cos, token_sin, map_cos, map_sin = (t.contiguous() for t in
                                              (token_cos, token_sin, map_cos, map_sin))
    launch = _arguments(query, key, value, map_key, map_value, token_cos, map_cos)
    out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    launch_forward = torch.library.wrap_triton(_attention_forward)
    launch_forward[launch["grid"]](query, key, value, map_key, map_value, token_cos,
                                       token_sin, map_cos, map_sin, out, *launch["scalars"],
                                       **launch["meta"])
    return out


@torch.library.triton_op("swift::attention_heads_backward", mutates_args=())
def attention_heads_backward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                             map_key: torch.Tensor, map_value: torch.Tensor,
                             token_cos: torch.Tensor, token_sin: torch.Tensor,
                             map_cos: torch.Tensor, map_sin: torch.Tensor, out: torch.Tensor,
                             grad_out: torch.Tensor) -> list[torch.Tensor]:
    token_cos, token_sin, map_cos, map_sin = (t.contiguous() for t in
                                              (token_cos, token_sin, map_cos, map_sin))
    launch = _arguments(query, key, value, map_key, map_value, token_cos, map_cos)
    gradients = [torch.empty(query.shape, dtype=t.dtype, device=query.device)
                 for t in (query, key, value)] + [
        torch.empty(map_key.shape, dtype=t.dtype, device=query.device)
        for t in (map_key, map_value)]
    launch_backward = torch.library.wrap_triton(_attention_backward)
    launch_backward[launch["grid"]](query, key, value, map_key, map_value, token_cos,
                                        token_sin, map_cos, map_sin, out, grad_out.contiguous(),
                                        *gradients, *launch["scalars"], **launch["meta"])
    return gradients


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs, output)


def _backward(ctx, grad_out):
    gradients = attention_heads_backward(*ctx.saved_tensors, grad_out)
    return (*gradients, None, None, None, None)


attention_heads.register_autograd(_backward, setup_context=_setup_context)


def supported(query: torch.Tensor, pairs: int) -> bool:
    width = query.shape[-1]
    return (query.is_cuda and query.dtype in (torch.float32, torch.bfloat16)
            and width in (32, 64, 128) and pairs <= width // 2)
