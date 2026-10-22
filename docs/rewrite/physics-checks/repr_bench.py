"""Complex64 vs real (x, y) pairs vs 2x2 matrices for a flight-like kernel in JAX (CPU)."""
import time, jax, jax.numpy as jnp
M, STEPS, DT, W = 4096 * 10, 200, 1 / 30, 65536.0

def step_complex(c, _):
    pos, vel, att = c
    s = jnp.abs(vel); d = vel / s
    s = s / (1 + 1e-3 * s * DT)                      # exact drag
    vel = d * s * jnp.exp(1j * (0.02 * s * DT))      # lift rotation (unit complex)
    vel = vel + 40.0 * att * DT                      # thrust along attitude
    att = vel / jnp.abs(vel) * jnp.exp(1j * 0.1)     # attitude = path * slip
    pos = pos + vel * DT
    pos = jax.lax.complex(jnp.mod(pos.real, W), jnp.mod(pos.imag, W))
    return (pos, vel, att), None

def step_pairs(c, _):
    px, py, vx, vy, ax, ay = c
    s = jnp.sqrt(vx * vx + vy * vy); dx, dy = vx / s, vy / s
    s = s / (1 + 1e-3 * s * DT)
    ang = 0.02 * s * DT; co, si = jnp.cos(ang), jnp.sin(ang)
    vx, vy = s * (dx * co - dy * si), s * (dx * si + dy * co)
    vx, vy = vx + 40.0 * ax * DT, vy + 40.0 * ay * DT
    n = jnp.sqrt(vx * vx + vy * vy); c1, s1 = jnp.cos(0.1), jnp.sin(0.1)
    ax, ay = (vx * c1 - vy * s1) / n, (vx * s1 + vy * c1) / n
    px, py = jnp.mod(px + vx * DT, W), jnp.mod(py + vy * DT, W)
    return (px, py, vx, vy, ax, ay), None

def step_matrix(c, _):
    pos, vel, att = c                                 # (M, 2) each
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

key = jax.random.PRNGKey(0)
v = jax.random.normal(key, (M, 2)) * 100 + 1
cplx = (jnp.zeros(M, jnp.complex64), (v[:, 0] + 1j * v[:, 1]).astype(jnp.complex64), jnp.ones(M, jnp.complex64))
pairs = (jnp.zeros(M), jnp.zeros(M), v[:, 0], v[:, 1], jnp.ones(M), jnp.zeros(M))
mat = (jnp.zeros((M, 2)), v, jnp.tile(jnp.array([[1.0, 0.0]]), (M, 1)))
for name, f, x in (("complex64", step_complex, cplx), ("real pairs", step_pairs, pairs), ("2x2 matrices", step_matrix, mat)):
    run = jax.jit(lambda x: jax.lax.scan(f, x, None, length=STEPS)[0])
    jax.block_until_ready(run(x))
    t = time.perf_counter()
    for _ in range(5): jax.block_until_ready(run(x))
    print(f"{name:>13}: {(time.perf_counter() - t) / 5 / STEPS * 1e6:7.1f} us/step for {M} ships")
