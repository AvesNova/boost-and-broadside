"""Flight-like kernel in four representations (JAX). Times and a cross-check."""
import time, sys, jax, jax.numpy as jnp
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import pga
M, STEPS, DT, W = 4096 * 10, 200, 1 / 30, 65536.0

def step_complex(c, _):
    pos, vel, att = c
    s = jnp.abs(vel); d = vel / s
    s = s / (1 + 1e-3 * s * DT)
    vel = d * s * jnp.exp(1j * (0.02 * s * DT))
    vel = vel + 40.0 * att * DT
    att = vel / jnp.abs(vel) * jnp.exp(1j * 0.1)
    pos = pos + vel * DT
    pos = jax.lax.complex(jnp.mod(pos.real, W), jnp.mod(pos.imag, W))
    return (pos, vel, att), None

def rot(x, y, c, s): return x * c - y * s, x * s + y * c

def step_pairs(c, _):
    px, py, vx, vy, ax, ay = c
    s = jnp.sqrt(vx * vx + vy * vy); dx, dy = vx / s, vy / s
    s = s / (1 + 1e-3 * s * DT)
    ang = 0.02 * s * DT
    vx, vy = rot(dx * s, dy * s, jnp.cos(ang), jnp.sin(ang))
    vx, vy = vx + 40.0 * ax * DT, vy + 40.0 * ay * DT
    n = jnp.sqrt(vx * vx + vy * vy)
    ax, ay = rot(vx / n, vy / n, jnp.cos(0.1), jnp.sin(0.1))
    px, py = jnp.mod(px + vx * DT, W), jnp.mod(py + vy * DT, W)
    return (px, py, vx, vy, ax, ay), None

def step_matrix(c, _):
    pos, vel, att = c
    s = jnp.linalg.norm(vel, axis=-1, keepdims=True); d = vel / s
    s = s / (1 + 1e-3 * s * DT)
    ang = (0.02 * s * DT)[..., 0]
    R = jnp.stack([jnp.stack([jnp.cos(ang), -jnp.sin(ang)], -1), jnp.stack([jnp.sin(ang), jnp.cos(ang)], -1)], -2)
    vel = jnp.einsum("mij,mj->mi", R, d) * s
    vel = vel + 40.0 * att * DT
    R2 = jnp.array([[jnp.cos(0.1), -jnp.sin(0.1)], [jnp.sin(0.1), jnp.cos(0.1)]])
    att = (vel / jnp.linalg.norm(vel, axis=-1, keepdims=True)) @ R2.T
    pos = jnp.mod(pos + vel * DT, W)
    return (pos, vel, att), None

def step_pga(c, _):
    px, py, vx, vy, ax, ay = c
    s = jnp.sqrt(vx * vx + vy * vy); dx, dy = vx / s, vy / s
    s = s / (1 + 1e-3 * s * DT)
    vx, vy = pga.apply_dir(pga.rotor(0.02 * s * DT), dx * s, dy * s)   # lift as a rotor
    vx, vy = vx + 40.0 * ax * DT, vy + 40.0 * ay * DT
    n = jnp.sqrt(vx * vx + vy * vy)
    ax, ay = pga.apply_dir(pga.rotor(jnp.float32(0.1)), vx / n, vy / n)  # attitude = path * slip
    px, py = pga.apply_point(pga.translator(vx * DT, vy * DT), px, py)  # drift as a translator
    return (jnp.mod(px, W), jnp.mod(py, W), vx, vy, ax, ay), None

def step_pga_motor(c, _):
    """Same, but rotation and translation composed into one motor per step (PGA idiom)."""
    px, py, vx, vy, ax, ay = c
    s = jnp.sqrt(vx * vx + vy * vy); dx, dy = vx / s, vy / s
    s = s / (1 + 1e-3 * s * DT)
    R = pga.rotor(0.02 * s * DT)
    vx, vy = pga.apply_dir(R, dx * s, dy * s)
    vx, vy = vx + 40.0 * ax * DT, vy + 40.0 * ay * DT
    n = jnp.sqrt(vx * vx + vy * vy)
    ax, ay = pga.apply_dir(pga.rotor(jnp.float32(0.1)), vx / n, vy / n)
    Mo = pga.normalize(pga.mul(pga.translator(vx * DT, vy * DT), (1.0, 0.0, 0.0, 0.0)))
    px, py = pga.apply_point(Mo, px, py)
    return (jnp.mod(px, W), jnp.mod(py, W), vx, vy, ax, ay), None

key = jax.random.PRNGKey(0)
v = jax.random.normal(key, (M, 2)) * 100 + 1
cplx = (jnp.zeros(M, jnp.complex64), (v[:, 0] + 1j * v[:, 1]).astype(jnp.complex64), jnp.ones(M, jnp.complex64))
pairs = (jnp.zeros(M), jnp.zeros(M), v[:, 0], v[:, 1], jnp.ones(M), jnp.zeros(M))
mat = (jnp.zeros((M, 2)), v, jnp.tile(jnp.array([[1.0, 0.0]]), (M, 1)))
results = {}
for name, f, x in (("complex64", step_complex, cplx), ("real pairs", step_pairs, pairs),
                   ("2x2 matrices", step_matrix, mat), ("PGA", step_pga, pairs),
                   ("PGA motor", step_pga_motor, pairs)):
    run = jax.jit(lambda x: jax.lax.scan(f, x, None, length=STEPS)[0])
    out = jax.block_until_ready(run(x))
    ts = []
    for _ in range(15):
        t = time.perf_counter(); jax.block_until_ready(run(x)); ts.append(time.perf_counter() - t)
    us = min(ts) / STEPS * 1e6
    if name == "complex64": px = out[0].real
    elif name == "2x2 matrices": px = out[0][:, 0]
    else: px = out[0]
    results[name] = px
    print(f"{name:>13}: {us:7.1f} us/step for {M} ships; max torus |x - x_complex| = {float(jnp.max(jnp.abs((px - results['complex64'] + W / 2) % W - W / 2))):.3g} m")
