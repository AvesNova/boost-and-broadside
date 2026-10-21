"""Pure math shared by every mockup: the parts no NN library touches.

Layout is time-major: x is (T, B, D) with B = envs * ships. `first[t, b]` is True
when step t starts a new episode for row b (the carry must not leak across it).
"""

import jax
import jax.numpy as jnp

CONV_KERNEL = 4  # causal depthwise conv taps
EPS = 1e-6


def rms_norm(x, scale):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + EPS) * scale


def gelu(x):
    # torch's F.gelu default is the exact erf form; jax.nn.gelu defaults to tanh.
    return jax.nn.gelu(x, approximate=False)


def _shift(m, k):
    """Shift a (T, ...) mask k steps later in time, filling with False."""
    return m if k == 0 else jnp.concatenate([jnp.zeros_like(m[:k]), m[:-k]], axis=0)


def _lag_validity(first):
    """(K, T, B): tap `lag` of output t is valid if no episode starts in (t-lag, t]."""
    valid = [jnp.ones_like(first)]
    for lag in range(1, CONV_KERNEL):
        valid.append(valid[-1] & ~_shift(first, lag - 1))
    return jnp.stack(valid)


def causal_conv(x, buf, kernel, bias, first):
    """Depthwise causal conv over time that never reads across an episode start.

    x: (T, B, D); buf: (K-1, B, D) the previous segment's last inputs, oldest first;
    kernel: (K, D) with kernel[lag]; bias: (D,). Returns (out, new_buf).
    """
    T, K = x.shape[0], CONV_KERNEL
    padded = jnp.concatenate([buf, x], axis=0)  # (K-1+T, B, D)
    valid = _lag_validity(first)
    out = bias + sum(
        jnp.where(valid[lag][..., None], padded[K - 1 - lag : K - 1 - lag + T] * kernel[lag], 0.0)
        for lag in range(K)
    )
    # Keep only buffer entries the next segment may read: those after the last start.
    ext = jnp.concatenate([first, jnp.zeros_like(first[:1])], axis=0)
    keep = _lag_validity(ext)[1:, -1]  # (K-1, B): lag 1 = newest entry
    new_buf = padded[-(K - 1) :] * keep[::-1, :, None]
    return out, new_buf


def rglru_scan(log_a, b, h0, first):
    """h_t = a_t * h_{t-1} + b_t over time with a = exp(log_a), reset at episode starts."""
    a = jnp.where(first[..., None], 0.0, jnp.exp(log_a))
    b = b.at[0].add(a[0] * h0)  # fold the carried state into step 0

    def combine(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return a_l * a_r, a_r * b_l + b_r

    _, h = jax.lax.associative_scan(combine, (a, b))
    return h


def rglru_inputs(x, r_logits, i_logits, log_lambda, c=8.0):
    """RG-LRU coefficients: a = sigmoid(lambda)^(c r), input gated by sqrt(1 - a^2)."""
    log_a = c * jax.nn.sigmoid(r_logits) * jax.nn.log_sigmoid(log_lambda)
    a = jnp.exp(log_a)
    gate = jnp.sqrt(jnp.maximum((1.0 - a) * (1.0 + a), EPS))
    return log_a, gate * (jax.nn.sigmoid(i_logits) * x)
