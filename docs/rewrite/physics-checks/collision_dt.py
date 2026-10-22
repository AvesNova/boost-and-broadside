"""How much does the tick rate change bullet-hit detection?

Target flies a constant-speed arc (speed 100-180 m/s, lateral accel up to 270 m/s^2).
A bullet (500 m/s muzzle + shooter velocity up to 180 m/s, straight line) is aimed so its
true closest approach to the target is d_true in [0, 20] m (radius 10). Truth is a
1e5 Hz sample. Methods per tick:
  current:  bullet segment start->end vs the target's END-of-tick position (env today)
  relative: segment of the relative motion (bullet - target, both linear in the tick)
  points:   distance at tick instants only (no sweep)
"""
import numpy as np

rng = np.random.default_rng(0)
M = 20000
R = 10.0
LIFE = 1.0

speed = rng.uniform(100, 180, M)
head = rng.uniform(0, 2 * np.pi, M)
acc = rng.uniform(-270, 270, M)
omega = acc / speed                       # rad/s turn rate of the target
p0 = np.zeros((M, 2))

def target(t):
    t = np.asarray(t)
    th = head[:, None] + omega[:, None] * t
    small = np.abs(omega) < 1e-9
    w = np.where(small, 1.0, omega)[:, None]
    x = np.where(small[:, None], speed[:, None] * np.cos(head)[:, None] * t,
                 speed[:, None] / w * (np.sin(th) - np.sin(head)[:, None]))
    y = np.where(small[:, None], speed[:, None] * np.sin(head)[:, None] * t,
                 -speed[:, None] / w * (np.cos(th) - np.cos(head)[:, None]))
    return np.stack([x, y], -1)            # (M, T, 2)

tstar = rng.uniform(0.15, 0.85, M)
bdir = rng.uniform(0, 2 * np.pi, M)
shooter_v = rng.uniform(0, 180, M)[:, None] * np.stack([np.cos(bdir + rng.normal(0, .3, M)), np.sin(bdir + rng.normal(0, .3, M))], -1)
vb = 500 * np.stack([np.cos(bdir), np.sin(bdir)], -1) + shooter_v
# target velocity at t*
th = head + omega * tstar
vt = speed[:, None] * np.stack([np.cos(th), np.sin(th)], -1)
rel = vb - vt
perp = np.stack([-rel[:, 1], rel[:, 0]], -1) / np.linalg.norm(rel, axis=1, keepdims=True)
d_aim = rng.uniform(0, 20, M)
pt = target(tstar[:, None])[:, 0]
pb_star = pt + perp * d_aim[:, None]
b0 = pb_star - vb * tstar[:, None]

def bullet(t):
    t = np.broadcast_to(np.asarray(t), (M, np.asarray(t).shape[-1])); return b0[:, None] + vb[:, None] * t[..., None]

# truth (fine sampling)
tf = tstar[:, None] + np.linspace(-0.06, 0.06, 12001)[None]
dtrue = np.full(M, np.inf)
for chunk in np.array_split(np.arange(tf.shape[1]), 12):
    tt = tf[:, chunk]
    d = np.linalg.norm(bullet(tt) - target(tt), axis=-1).min(1)
    dtrue = np.minimum(dtrue, d)

def seg_dist(a, b, p):
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    return np.linalg.norm(a + ab * t[..., None] - p, axis=-1)

print(f"{'rate':>6} {'method':>9} {'mean|err| m':>12} {'p99|err| m':>11} {'hit flips %':>12}")
for hz in (20, 30, 60, 120):
    dt = 1 / hz
    phase = rng.uniform(0, dt, M)          # launch is on a tick; target pose phase random
    # ticks relative to launch: launch at t=0, ticks at k*dt (k=1..)
    k = np.arange(1, int(round(LIFE * hz)) + 1)
    t1 = k[None] * dt + 0 * phase[:, None]
    t0 = t1 - dt
    B0, B1 = bullet(t0), bullet(t1)
    T0, T1 = target(t0), target(t1)
    cur = seg_dist(B0, B1, T1).min(1)
    relm = seg_dist(B0 - T0, B1 - T1, np.zeros_like(B0)).min(1)
    pts = np.linalg.norm(B1 - T1, axis=-1).min(1)
    for name, est in (("current", cur), ("relative", relm), ("points", pts)):
        err = np.abs(est - dtrue)
        flips = np.mean((est < R) != (dtrue < R)) * 100
        print(f"{hz:>5}Hz {name:>9} {err.mean():12.3f} {np.quantile(err, .99):11.3f} {flips:12.2f}")
