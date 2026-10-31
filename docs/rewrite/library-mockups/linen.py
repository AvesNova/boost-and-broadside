"""Griffin temporal block in Flax Linen: modules describe structure, params live outside."""

import flax.linen as nn
import jax.numpy as jnp

from common import CONV_KERNEL, EPS, causal_conv, gelu, rglru_inputs, rglru_scan


class CausalConv(nn.Module):
    @nn.compact
    def __call__(self, x, buf, first):
        d = x.shape[-1]
        kernel = self.param("kernel", nn.initializers.normal(1.0 / CONV_KERNEL), (CONV_KERNEL, d))
        bias = self.param("bias", nn.initializers.zeros, (d,))
        return causal_conv(x, buf, kernel, bias, first)


class RGLRU(nn.Module):
    c: float = 8.0

    @nn.compact
    def __call__(self, x, h0, first):
        d = x.shape[-1]
        log_lambda = self.param("log_lambda", lambda _key: jnp.linspace(0.0, 4.0, d))
        log_a, b = rglru_inputs(
            x, nn.Dense(d, name="gate_a")(x), nn.Dense(d, name="gate_x")(x), log_lambda, self.c
        )
        return rglru_scan(log_a, b, h0, first)


class GatedMLP(nn.Module):
    expand: int = 4

    @nn.compact
    def __call__(self, x):
        d, hidden = x.shape[-1], self.expand * x.shape[-1]
        gate = nn.Dense(hidden, use_bias=False, name="gate_proj")(x)
        up = nn.Dense(hidden, use_bias=False, name="up_proj")(x)
        return nn.Dense(d, use_bias=False, name="down_proj")(gelu(gate) * up)


class GriffinTemporalBlock(nn.Module):
    @nn.compact
    def __call__(self, x, carry, first):
        """x: (T, B, D); carry: (h (B, D), conv_buf (K-1, B, D)); first: (T, B)."""
        d = x.shape[-1]
        h0, buf = carry
        normed = nn.RMSNorm(epsilon=EPS, name="norm1")(x)
        b1 = nn.Dense(d, use_bias=False, name="linear1")(normed)
        b1, new_buf = CausalConv(name="conv")(b1, buf, first)
        h = RGLRU(name="lru")(b1, h0, first)
        b2 = gelu(nn.Dense(d, use_bias=False, name="linear2")(normed))
        x = x + nn.Dense(d, use_bias=False, name="linear_out")(h * b2)
        x = x + GatedMLP(name="mlp")(nn.RMSNorm(epsilon=EPS, name="norm2")(x))
        return x, (h[-1], new_buf)


# Usage (shapes are inferred from an example input at init):
#   block = GriffinTemporalBlock()
#   params = block.init(key, x, carry, first)["params"]     # a plain nested dict
#   y, carry = block.apply({"params": params}, x, carry, first)
def to_plain(params):
    return params  # already a plain dict with the contract's key layout


def from_plain(plain):
    return plain
