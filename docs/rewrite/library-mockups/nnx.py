"""Griffin temporal block in Flax NNX: modules are objects that own their params."""

import jax.numpy as jnp
from flax import nnx

from common import CONV_KERNEL, EPS, causal_conv, gelu, rglru_inputs, rglru_scan


class CausalConv(nnx.Module):
    def __init__(self, d, *, rngs):
        self.kernel = nnx.Param(nnx.initializers.normal(1.0 / CONV_KERNEL)(rngs.params(), (CONV_KERNEL, d)))
        self.bias = nnx.Param(jnp.zeros(d))

    def __call__(self, x, buf, first):
        return causal_conv(x, buf, self.kernel[...], self.bias[...], first)


class RGLRU(nnx.Module):
    def __init__(self, d, *, rngs, c=8.0):
        self.c = c
        self.log_lambda = nnx.Param(jnp.linspace(0.0, 4.0, d))
        self.gate_a = nnx.Linear(d, d, rngs=rngs)
        self.gate_x = nnx.Linear(d, d, rngs=rngs)

    def __call__(self, x, h0, first):
        log_a, b = rglru_inputs(x, self.gate_a(x), self.gate_x(x), self.log_lambda[...], self.c)
        return rglru_scan(log_a, b, h0, first)


class GatedMLP(nnx.Module):
    def __init__(self, d, *, rngs, expand=4):
        self.gate_proj = nnx.Linear(d, expand * d, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(d, expand * d, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(expand * d, d, use_bias=False, rngs=rngs)

    def __call__(self, x):
        return self.down_proj(gelu(self.gate_proj(x)) * self.up_proj(x))


class GriffinTemporalBlock(nnx.Module):
    def __init__(self, d, *, rngs):
        self.norm1 = nnx.RMSNorm(d, epsilon=EPS, rngs=rngs)
        self.linear1 = nnx.Linear(d, d, use_bias=False, rngs=rngs)
        self.conv = CausalConv(d, rngs=rngs)
        self.lru = RGLRU(d, rngs=rngs)
        self.linear2 = nnx.Linear(d, d, use_bias=False, rngs=rngs)
        self.linear_out = nnx.Linear(d, d, use_bias=False, rngs=rngs)
        self.norm2 = nnx.RMSNorm(d, epsilon=EPS, rngs=rngs)
        self.mlp = GatedMLP(d, rngs=rngs)

    def __call__(self, x, carry, first):
        """x: (T, B, D); carry: (h (B, D), conv_buf (K-1, B, D)); first: (T, B)."""
        h0, buf = carry
        normed = self.norm1(x)
        b1, new_buf = self.conv(self.linear1(normed), buf, first)
        h = self.lru(b1, h0, first)
        x = x + self.linear_out(h * gelu(self.linear2(normed)))
        x = x + self.mlp(self.norm2(x))
        return x, (h[-1], new_buf)


# Usage. Eager: block = GriffinTemporalBlock(d, rngs=nnx.Rngs(0)); y, carry = block(x, carry, first)
# Pure (what jit, vmap over league slots and optax want): split once, merge inside.
#   graphdef, state = nnx.split(block)
#   def apply(state, x, carry, first): return nnx.merge(graphdef, state)(x, carry, first)
def to_plain(state):
    return nnx.to_pure_dict(state)  # nnx.State of Param objects -> plain dict


def make_from_plain(template_state):
    def from_plain(plain):
        state = nnx.clone(template_state)
        nnx.replace_by_pure_dict(state, plain)
        return state

    return from_plain
