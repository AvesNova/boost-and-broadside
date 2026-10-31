"""Griffin temporal block in plain JAX: params are a nested dict, the model is functions."""

import jax
import jax.numpy as jnp

from common import CONV_KERNEL, causal_conv, gelu, rglru_inputs, rglru_scan, rms_norm


def _dense(key, d_in, d_out, bias=False):
    p = {"kernel": jax.random.normal(key, (d_in, d_out)) / jnp.sqrt(d_in)}
    if bias:
        p["bias"] = jnp.zeros(d_out)
    return p


def init(key, d, expand=4):
    k = iter(jax.random.split(key, 9))
    return {
        "norm1": {"scale": jnp.ones(d)},
        "linear1": _dense(next(k), d, d),
        "conv": {
            "kernel": jax.random.normal(next(k), (CONV_KERNEL, d)) / CONV_KERNEL,
            "bias": jnp.zeros(d),
        },
        "lru": {
            "log_lambda": jnp.linspace(0.0, 4.0, d),  # decay sigmoid ~0.5 .. ~0.98
            "gate_a": _dense(next(k), d, d, bias=True),
            "gate_x": _dense(next(k), d, d, bias=True),
        },
        "linear2": _dense(next(k), d, d),
        "linear_out": _dense(next(k), d, d),
        "norm2": {"scale": jnp.ones(d)},
        "mlp": {
            "gate_proj": _dense(next(k), d, expand * d),
            "up_proj": _dense(next(k), d, expand * d),
            "down_proj": _dense(next(k), expand * d, d),
        },
    }


def _linear(p, x):
    y = x @ p["kernel"]
    return y + p["bias"] if "bias" in p else y


def apply(p, x, carry, first):
    """x: (T, B, D); carry: (h (B, D), conv_buf (K-1, B, D)); first: (T, B)."""
    h0, buf = carry
    normed = rms_norm(x, p["norm1"]["scale"])
    b1, new_buf = causal_conv(
        _linear(p["linear1"], normed), buf, p["conv"]["kernel"], p["conv"]["bias"], first
    )
    lru = p["lru"]
    log_a, b = rglru_inputs(
        b1, _linear(lru["gate_a"], b1), _linear(lru["gate_x"], b1), lru["log_lambda"]
    )
    h = rglru_scan(log_a, b, h0, first)
    x = x + _linear(p["linear_out"], h * gelu(_linear(p["linear2"], normed)))
    m, n = p["mlp"], rms_norm(x, p["norm2"]["scale"])
    x = x + _linear(m["down_proj"], gelu(_linear(m["gate_proj"], n)) * _linear(m["up_proj"], n))
    return x, (h[-1], new_buf)


# The contract boundary is the native format here, so these are identities.
to_plain = from_plain = lambda p: p
