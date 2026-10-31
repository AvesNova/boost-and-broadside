"""Griffin temporal block in Equinox: a module *is* a pytree of its params."""

import equinox as eqx
import jax
import jax.numpy as jnp

from common import CONV_KERNEL, EPS, causal_conv, gelu, rglru_inputs, rglru_scan


def _map(layer, x):
    """Equinox layers act on one vector; map over the leading (T, B) axes."""
    return jax.vmap(jax.vmap(layer))(x)


class RGLRU(eqx.Module):
    log_lambda: jax.Array
    gate_a: eqx.nn.Linear
    gate_x: eqx.nn.Linear
    c: float = eqx.field(static=True)

    def __init__(self, d, *, key, c=8.0):
        ka, kx = jax.random.split(key)
        self.log_lambda = jnp.linspace(0.0, 4.0, d)
        self.gate_a = eqx.nn.Linear(d, d, key=ka)
        self.gate_x = eqx.nn.Linear(d, d, key=kx)
        self.c = c

    def __call__(self, x, h0, first):
        log_a, b = rglru_inputs(x, _map(self.gate_a, x), _map(self.gate_x, x), self.log_lambda, self.c)
        return rglru_scan(log_a, b, h0, first)


class GatedMLP(eqx.Module):
    gate_proj: eqx.nn.Linear
    up_proj: eqx.nn.Linear
    down_proj: eqx.nn.Linear

    def __init__(self, d, *, key, expand=4):
        kg, ku, kd = jax.random.split(key, 3)
        self.gate_proj = eqx.nn.Linear(d, expand * d, use_bias=False, key=kg)
        self.up_proj = eqx.nn.Linear(d, expand * d, use_bias=False, key=ku)
        self.down_proj = eqx.nn.Linear(expand * d, d, use_bias=False, key=kd)

    def __call__(self, x):
        return self.down_proj(gelu(self.gate_proj(x)) * self.up_proj(x))  # one vector


class GriffinTemporalBlock(eqx.Module):
    norm1: eqx.nn.RMSNorm
    linear1: eqx.nn.Linear
    conv_kernel: jax.Array
    conv_bias: jax.Array
    lru: RGLRU
    linear2: eqx.nn.Linear
    linear_out: eqx.nn.Linear
    norm2: eqx.nn.RMSNorm
    mlp: GatedMLP

    def __init__(self, d, *, key):
        k = iter(jax.random.split(key, 6))
        self.norm1 = eqx.nn.RMSNorm(d, eps=EPS)
        self.linear1 = eqx.nn.Linear(d, d, use_bias=False, key=next(k))
        self.conv_kernel = jax.random.normal(next(k), (CONV_KERNEL, d)) / CONV_KERNEL
        self.conv_bias = jnp.zeros(d)
        self.lru = RGLRU(d, key=next(k))
        self.linear2 = eqx.nn.Linear(d, d, use_bias=False, key=next(k))
        self.linear_out = eqx.nn.Linear(d, d, use_bias=False, key=next(k))
        self.norm2 = eqx.nn.RMSNorm(d, eps=EPS)
        self.mlp = GatedMLP(d, key=next(k))

    def __call__(self, x, carry, first):
        """x: (T, B, D); carry: (h (B, D), conv_buf (K-1, B, D)); first: (T, B)."""
        h0, buf = carry
        normed = _map(self.norm1, x)
        b1, new_buf = causal_conv(_map(self.linear1, normed), buf, self.conv_kernel, self.conv_bias, first)
        h = self.lru(b1, h0, first)
        x = x + _map(self.linear_out, h * gelu(_map(self.linear2, normed)))
        x = x + _map(self.mlp, _map(self.norm2, x))
        return x, (h[-1], new_buf)


# Usage: block = GriffinTemporalBlock(d, key=key); y, carry = block(x, carry, first)
# The module is already a pytree, so jax.jit/grad work on it once non-arrays are split off:
#   params, static = eqx.partition(block, eqx.is_array)
#   def apply(params, x, carry, first): return eqx.combine(params, static)(x, carry, first)
# Equinox Linear stores weight as (out, in); the contract layout is (in, out).
def _lin(m):
    p = {"kernel": m.weight.T}
    if m.bias is not None:
        p["bias"] = m.bias
    return p


def to_plain(b):
    return {
        "norm1": {"scale": b.norm1.weight},
        "linear1": _lin(b.linear1),
        "conv": {"kernel": b.conv_kernel, "bias": b.conv_bias},
        "lru": {"log_lambda": b.lru.log_lambda, "gate_a": _lin(b.lru.gate_a), "gate_x": _lin(b.lru.gate_x)},
        "linear2": _lin(b.linear2),
        "linear_out": _lin(b.linear_out),
        "norm2": {"scale": b.norm2.weight},
        "mlp": {name: _lin(getattr(b.mlp, name)) for name in ("gate_proj", "up_proj", "down_proj")},
    }


def make_from_plain(template):
    def from_plain(p):
        def lin(m, q):
            m = eqx.tree_at(lambda m: m.weight, m, q["kernel"].T)
            return eqx.tree_at(lambda m: m.bias, m, q["bias"]) if "bias" in q else m

        b = template
        b = eqx.tree_at(lambda b: b.norm1.weight, b, p["norm1"]["scale"])
        b = eqx.tree_at(lambda b: b.norm2.weight, b, p["norm2"]["scale"])
        b = eqx.tree_at(lambda b: (b.conv_kernel, b.conv_bias), b, (p["conv"]["kernel"], p["conv"]["bias"]))
        b = eqx.tree_at(lambda b: b.lru.log_lambda, b, p["lru"]["log_lambda"])
        for path, q in [
            (lambda b: b.linear1, p["linear1"]),
            (lambda b: b.linear2, p["linear2"]),
            (lambda b: b.linear_out, p["linear_out"]),
            (lambda b: b.lru.gate_a, p["lru"]["gate_a"]),
            (lambda b: b.lru.gate_x, p["lru"]["gate_x"]),
            (lambda b: b.mlp.gate_proj, p["mlp"]["gate_proj"]),
            (lambda b: b.mlp.up_proj, p["mlp"]["up_proj"]),
            (lambda b: b.mlp.down_proj, p["mlp"]["down_proj"]),
        ]:
            b = eqx.tree_at(path, b, lin(path(b), q))
        return b

    return from_plain
