"""Accuracy, energy, cost and feel of the 2D rigid-body prototype (x/y vs PGA)."""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(__file__))
import jax
jax.config.update("jax_enable_x64", not os.environ.get("NO64"))
import jax.numpy as jnp
import rigid2d as R

SH = R.Ship()
S, HOLD, SECONDS = 2048, 0.1, 2.0
key = jax.random.PRNGKey(0)
k1, k2, k3, k4, k5 = jax.random.split(key, 5)
n_hold = int(round(SECONDS / HOLD))
acts = jnp.stack([jax.random.randint(k1, (n_hold, S), 0, 3), jax.random.randint(k2, (n_hold, S), 0, 7)], -1)
speed0 = jax.random.uniform(k3, (S,), minval=60.0, maxval=180.0)
head0 = jax.random.uniform(k4, (S,), minval=0.0, maxval=2 * math.pi)
power0 = jax.random.uniform(k5, (S,), minval=5.0, maxval=100.0)

def init_xy(dt):
    z = jnp.zeros(S, dt)
    return (z, z, (speed0 * jnp.cos(head0)).astype(dt), (speed0 * jnp.sin(head0)).astype(dt), head0.astype(dt), z, power0.astype(dt))

def init_xy2(dt):
    z = jnp.zeros(S, dt)
    return (z, z, jnp.cos(head0).astype(dt), jnp.sin(head0).astype(dt), z, speed0.astype(dt), z, power0.astype(dt))

def pose_xy2(st):
    return st[0], st[1], jnp.arctan2(st[3], st[2])

def init_pga(dt):
    z = jnp.zeros(S, dt)
    M = R.pga_from_pose(z, z, jnp.cos(head0).astype(dt), jnp.sin(head0).astype(dt))
    return (M, (z, speed0.astype(dt), z), power0.astype(dt))

def run(step, init, hz, dt=jnp.float32, sh=SH):
    h = 1.0 / hz
    tph = int(round(HOLD * hz))
    def hold(st, a):
        st = jax.lax.fori_loop(0, tph, lambda i, s: step(sh, s, a, h), st)
        return st, None
    return jax.jit(lambda st: jax.lax.scan(hold, st, acts)[0])(init(dt))

def pose_xy(st):
    return st[0], st[1], st[4]

def pose_pga(st):
    x, y = R.pga_apply_origin(st[0]); c, s = R.pga_heading(st[0])
    return x, y, jnp.arctan2(s, c)

def energy_xy(st): return 0.5 * (st[2] ** 2 + st[3] ** 2) + SH.psc * st[6]
def energy_pga(st): return 0.5 * (st[1][1] ** 2 + st[1][2] ** 2) + SH.psc * st[2]

t = time.time()
ref = pose_xy2(run(R.xy2_step, init_xy2, 3840, jnp.float64))
ref_p = pose_pga(run(R.pga_step, init_pga, 3840, jnp.float64))
print(f"reference x/y-2 vs PGA at 3840 Hz (float64): max pos diff {float(jnp.max(jnp.hypot(ref[0]-ref_p[0], ref[1]-ref_p[1]))):.2e} m   [{time.time()-t:.0f} s]")
print(f"{'version':>6} {'Hz':>4} {'pos mean m':>10} {'pos p99 m':>9} {'head deg':>8}")
for name, step, init, pose in (("x/y", R.xy_step, init_xy, pose_xy), ("x/y-2", R.xy2_step, init_xy2, pose_xy2), ("PGA", R.pga_step, init_pga, pose_pga)):
    for hz in (30, 60, 120):
        x, y, psi = pose(run(step, init, hz))
        e = jnp.hypot(x - ref[0], y - ref[1])
        hd = jnp.abs(jnp.angle(jnp.exp(1j * (psi - ref[2])))) * 180 / math.pi
        print(f"{name:>6} {hz:>4} {float(e.mean()):10.3f} {float(jnp.quantile(e, .99)):9.3f} {float(hd.mean()):8.3f}")

# energy: drag and regen off
sh0 = R.Ship(cd0=0.0, k_ind=0.0, regen=0.0)
def energy_xy2(st): return 0.5 * (st[5] ** 2 + st[6] ** 2) + SH.psc * st[7]
for name, step, init, en in (("x/y", R.xy_step, init_xy, energy_xy), ("x/y-2", R.xy2_step, init_xy2, energy_xy2), ("PGA", R.pga_step, init_pga, energy_pga)):
    e0 = en(init(jnp.float32)); e1 = en(run(step, init, 30, sh=sh0))
    print(f"energy (no drag, no regen), {name} 30 Hz: mean |dE|/E = {float(jnp.mean(jnp.abs(e1 - e0) / e0)):.2e}")

# feel at constant speed: rotational subsystem only (x/y exact step at 1 kHz)
def response(u, a_from, a_to, sh=SH):
    """Time for alpha to go 90% of the way from a steady a_from command to a_to, and peak overshoot."""
    w = float(R.omega(sh, jnp.float64(u)))
    y, z, t, h = a_from - a_to, 0.0, 0.0, 1e-3
    zw = sh.zeta * w; wd = math.sqrt(max(w * w - zw * zw, 0.0))
    t90, peak = None, 0.0
    while t < 3.0:
        e = math.exp(-zw * h); c, s = math.cos(wd * h), math.sin(wd * h)
        y, z = e * (y * c + (z + zw * y) * s / wd), e * (z * c - (w * w * y + zw * z) * s / wd)
        t += h
        if t90 is None and abs(y) <= 0.1 * abs(a_from - a_to): t90 = t
        peak = max(peak, -y / (a_from - a_to))
    return t90, peak
print(f"\ncorner speed {SH.u_corner:.0f} m/s")
print(f"{'u m/s':>6} {'turn rate deg/s':>15} {'radius m':>8} {'alpha cmd deg':>13} {'reversal s':>10} {'onset s':>8} {'overshoot':>9}")
for u in (40, 60, 100, 140, 180):
    ac = float(R.alpha_cmd(SH, jnp.float64(1.0), jnp.float64(u)))
    a = float(R.lift(SH, jnp.float64(ac))) * u * u
    rev, ovs = response(u, -ac, ac)
    onset, _ = response(u, 0.0, ac)
    print(f"{u:>6} {math.degrees(a / u):15.1f} {u * u / a:8.0f} {math.degrees(ac):13.1f} {rev:10.3f} {onset:8.3f} {ovs:9.1%}")
