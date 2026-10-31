"""v2 benchmark: accuracy with and without fields, energy, cost, x/y vs PGA cross-check."""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(__file__))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import rigid2d_v2 as R

SH = R.Ship()
S, HOLD, SECONDS, ARENA = 2048, 0.1, 2.0, 2000.0
ks = jax.random.split(jax.random.PRNGKey(0), 12)
n_hold = int(round(SECONDS / HOLD))
acts = jnp.stack([jax.random.randint(ks[0], (n_hold, S), 0, 3), jax.random.randint(ks[1], (n_hold, S), 0, 7)], -1)
speed0 = jax.random.uniform(ks[2], (S,), minval=60.0, maxval=180.0)
head0 = jax.random.uniform(ks[3], (S,), minval=0.0, maxval=2 * math.pi)
power0 = jax.random.uniform(ks[4], (S,), minval=5.0, maxval=100.0)
x0 = jax.random.uniform(ks[5], (S,), minval=-ARENA, maxval=ARENA)
y0 = jax.random.uniform(ks[6], (S,), minval=-ARENA, maxval=ARENA)
NF = 24
FIELD = (jax.random.uniform(ks[7], (NF,), minval=-ARENA, maxval=ARENA),
         jax.random.uniform(ks[8], (NF,), minval=-ARENA, maxval=ARENA),
         jax.random.uniform(ks[9], (NF,), minval=100.0, maxval=490.0),
         jnp.full((NF,), 40.0),
         math.log(math.sqrt(2.0)) * jax.random.randint(ks[10], (NF,), -2, 3).astype(jnp.float64))
NOFIELD = tuple(f[:0] for f in FIELD)

def init(kind, field, dt):
    n, gx, gy = R.fields_eval(*[f.astype(dt) for f in field], x0.astype(dt), y0.astype(dt))
    z = jnp.zeros(S, dt)
    w = (speed0 * n).astype(dt)   # proper speed so the world speed is speed0
    if kind == "xy":
        return (x0.astype(dt), y0.astype(dt), jnp.cos(head0).astype(dt), jnp.sin(head0).astype(dt), z, w, z, power0.astype(dt), n, gx, gy)
    rot = (jnp.cos(0.5 * head0).astype(dt), -jnp.sin(0.5 * head0).astype(dt), z, z)
    tr = (1.0 + z, z, -0.5 * x0.astype(dt), -0.5 * y0.astype(dt))
    return (R.motor_mul(tr, rot), z, w, z, power0.astype(dt), n, gx, gy)

def run(kind, field, hz, dt=jnp.float32, sh=SH):
    step = R.xy_step if kind == "xy" else R.pga_step
    fld = tuple(f.astype(dt) for f in field)
    h = 1.0 / hz; tph = int(round(HOLD * hz))
    def hold(st, a):
        return jax.lax.fori_loop(0, tph, lambda i, s: step(sh, fld, s, a, h), st), None
    return jax.jit(lambda st: jax.lax.scan(hold, st, acts)[0])(init(kind, field, dt))

def pose(kind, st):
    if kind == "xy":
        return st[0], st[1], jnp.arctan2(st[3], st[2])
    x, y = R.motor_position(st[0]); c, s = R.motor_heading(st[0])
    return x, y, jnp.arctan2(s, c)

def energy(kind, st):
    wx, wy, p = (st[5], st[6], st[7]) if kind == "xy" else (st[2], st[3], st[4])
    return 0.5 * (wx * wx + wy * wy) + SH.psc * p

for label, field in (("no fields", NOFIELD), ("fields", FIELD)):
    t = time.time()
    ref = pose("xy", run("xy", field, 3840, jnp.float64))
    refp = pose("pga", run("pga", field, 3840, jnp.float64))
    print(f"\n[{label}] reference x/y vs PGA (3840 Hz, f64): max {float(jnp.max(jnp.hypot(ref[0]-refp[0], ref[1]-refp[1]))):.1e} m [{time.time()-t:.0f} s]")
    if label == "fields":
        n = R.fields_eval(*FIELD, ref[0], ref[1])[0]
        print(f"  ships ending inside a field (n != 1): {float(jnp.mean(jnp.abs(n - 1) > 1e-3)):.0%}")
    for kind in ("xy", "pga"):
        for hz in (30, 60, 120):
            x, y, psi = pose(kind, run(kind, field, hz))
            e = jnp.hypot(x - ref[0], y - ref[1])
            hd = jnp.abs(jnp.angle(jnp.exp(1j * (psi - ref[2])))) * 180 / math.pi
            print(f"  {kind:>4} {hz:>4} Hz  pos mean {float(e.mean()):7.3f} m  p99 {float(jnp.quantile(e, .99)):7.3f} m  heading {float(hd.mean()):6.3f} deg")
    sh0 = R.Ship(cd0=0.0, k_ind=0.0, regen=0.0)
    for kind in ("xy", "pga"):
        st0 = init(kind, field, jnp.float32); st1 = run(kind, field, 30, sh=sh0)
        print(f"  energy (no drag, no regen) {kind} 30 Hz: mean |dE|/E {float(jnp.mean(jnp.abs(energy(kind, st1) - energy(kind, st0)) / energy(kind, st0))):.1e}")
