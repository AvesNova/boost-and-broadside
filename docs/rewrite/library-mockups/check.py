"""Checks the four mockups agree and support what the training design needs.

Run: python check.py (needs jax, flax, equinox, optax, flatbuffers).
"""

import warnings

warnings.simplefilter("error", DeprecationWarning)  # surface API churn loudly

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from flax import nnx as fnnx
from jax import export

import equinox_
import linen
import nnx
import plain_jax
from common import CONV_KERNEL

T, B, D, SLOTS = 16, 6, 8, 4
key = jax.random.key(0)
x = jax.random.normal(jax.random.key(1), (T, B, D))
first = jax.random.bernoulli(jax.random.key(2), 0.15, (T, B))
carry = (jax.random.normal(jax.random.key(3), (B, D)), jax.random.normal(jax.random.key(4), (CONV_KERNEL - 1, B, D)))

# Each library: native init, and the contract function apply_plain(plain, x, carry, first).
lin_block = linen.GriffinTemporalBlock()
nnx_graphdef, nnx_state = fnnx.split(nnx.GriffinTemporalBlock(D, rngs=fnnx.Rngs(0)))
eqx_block = equinox_.GriffinTemporalBlock(D, key=key)
eqx_params, eqx_static = eqx.partition(eqx_block, eqx.is_array)
nnx_from_plain = nnx.make_from_plain(nnx_state)
eqx_from_plain = equinox_.make_from_plain(eqx_block)

libs = {
    "plain_jax": (lambda k: plain_jax.init(k, D), plain_jax.apply),
    "linen": (
        lambda k: lin_block.init(k, x, carry, first)["params"],
        lambda p, *a: lin_block.apply({"params": p}, *a),
    ),
    "nnx": (
        lambda k: nnx.to_plain(fnnx.split(nnx.GriffinTemporalBlock(D, rngs=fnnx.Rngs(params=k)))[1]),
        lambda p, *a: fnnx.merge(nnx_graphdef, nnx_from_plain(p))(*a),
    ),
    "equinox": (
        lambda k: equinox_.to_plain(equinox_.GriffinTemporalBlock(D, key=k)),
        lambda p, *a: eqx_from_plain(p)(*a),
    ),
}

ref_plain = plain_jax.init(key, D)
ref_y, ref_carry = jax.jit(plain_jax.apply)(ref_plain, x, carry, first)
for name, (init, apply) in libs.items():
    own = init(key)
    assert jax.tree.structure(own) == jax.tree.structure(ref_plain), name  # same contract layout
    # 1. Same weights -> same outputs in every library.
    y, c = jax.jit(apply)(ref_plain, x, carry, first)
    assert jnp.allclose(y, ref_y, atol=1e-5) and all(
        jnp.allclose(a, b, atol=1e-5) for a, b in zip(c, ref_carry)
    ), name
    # 2. Rollout (one step at a time) equals the PPO sequence pass.
    step = jax.jit(apply)
    cs, ys = carry, []
    for t in range(T):
        yt, cs = step(ref_plain, x[t : t + 1], cs, first[t : t + 1])
        ys.append(yt[0])
    assert jnp.allclose(jnp.stack(ys), y, atol=1e-5), name
    # 3. League slots: SLOTS stacked param sets in one vmapped call.
    stacked = jax.vmap(init)(jax.random.split(key, SLOTS))
    ys = jax.jit(jax.vmap(apply, in_axes=(0, None, None, None)))(stacked, x, carry, first)
    assert ys[0].shape == (SLOTS, T, B, D), name
    # 4. Gradient and an optax step on the plain params.
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    grads = jax.grad(lambda p: apply(p, x, carry, first)[0].sum())(ref_plain)
    updates, _ = opt.update(grads, opt.init(ref_plain), ref_plain)
    optax.apply_updates(ref_plain, updates)
    # 5. Export and reload the contract function (leaderboard submission path).
    exported = export.deserialize(export.export(jax.jit(apply))(ref_plain, x, carry, first).serialize())
    assert jnp.allclose(exported.call(ref_plain, x, carry, first)[0], ref_y, atol=1e-5), name
    print(f"{name:9s} ok: matches plain JAX, rollout == sequence, {SLOTS} league slots, grad + optax, export")

print("jax", jax.__version__)
