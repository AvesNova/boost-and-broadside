"""Handling solver prototype for the new flight model: performance targets -> ship parameters.

Stage 1 is closed form plus one 3x3 linear solve on the steady-state force balance (thrust
taken as P/u). Stage 2 corrects for what stage 1 leaves out (soft thrust and lift caps, thrust
along the nose at an angle of attack) by Gauss-Newton on the stats measured in the simulator.
"""
import math, dataclasses
import numpy as np
from v9_common import *

G = 27.0
TARGETS = ("cruise", "top", "turn")              # what stage 2 matches by simulation

def stage1(t):
    """t: stall, corner, cruise (coast, charging), top (coast, store full), turn (deg/s, full stick,
    coast, store full), charge (W banked at cruise), static_g (static thrust / g)."""
    cl = G / t["stall"] ** 2                      # stall: cl u_s^2 = g
    a_max = cl * t["corner"] ** 2                 # corner: cl u_c^2 = a_max
    a0 = t["charge"] / t["cruise"]                # banked power = m u a0
    w = math.radians(t["turn"])
    s = (w * w + math.sqrt(w ** 4 + 4 * cl * cl * G * G)) / (2 * cl * cl)   # (cl u^2)^2 - g^2 = w^2 u^2
    ut = math.sqrt(s)
    def row(u, lift2):                            # P/u - cd0 u^2 - k lift^2/u^2
        return [1 / u, -u * u, -lift2 / (u * u)]
    A = np.array([row(t["cruise"], G * G), row(t["top"], G * G), row(ut, (cl * ut * ut) ** 2)])
    b = np.array([a0, 0.0, 0.0])
    P, cd0, k = np.linalg.solve(A, b)
    return dict(cl_max=cl, a_max=a_max, a0=a0, power=P, cd0=cd0, k_ind=k, f_static=t["static_g"] * G), ut

def measure(p):
    sh = ship(levels="cl", **p)
    big = dataclasses.replace(sh, e_max=1e9)
    F.STICK = jnp.array([0.0, -1 / 3, 1 / 3, -0.75, 0.75, -1.0, 1.0])
    cr = tail(fly(big, 90.0, 5e8, [0, 0, 0], 30)[0])                 # coast, charging
    tp = tail(fly(sh, 110.0, sh.e_max, [0, 0, 0], 30)[0])            # coast, store full
    u, r, _, _ = fly(sh, 60.0, sh.e_max, [0, 6, 0], 30)              # full stick, store full
    return np.array([cr, tp, math.degrees(tail(r))]), tail(u)

def solve(t, iters=4, verbose=True):
    p, ut = stage1(t)
    goal = np.array([t[k] for k in TARGETS])
    x = np.array([p["power"], p["cd0"], p["k_ind"]])
    for it in range(iters + 1):
        p.update(power=x[0], cd0=x[1], k_ind=x[2])
        m, u = measure(p)
        if verbose:
            print(f"  iter {it}: P {x[0]:.0f} W, cd0 {x[1]:.3e}, k {x[2]:.2f} -> cruise {m[0]:.1f}, "
                  f"top {m[1]:.1f}, turn {m[2]:.1f} deg/s at {u:.1f} m/s")
        if it == iters or np.max(np.abs(m - goal) / goal) < 2e-3:
            break
        J = np.zeros((3, 3))                          # finite-difference Jacobian in log parameters
        for j in range(3):
            q = dict(p); q[("power", "cd0", "k_ind")[j]] *= 1.02
            J[:, j] = (measure(q)[0] - m) / math.log(1.02)
        x = x * np.exp(np.linalg.lstsq(J, goal - m, rcond=None)[0])
    return p

if __name__ == "__main__":
    cases = {
        "ours (should recover P 1200, cd0 6.96e-4, k 11.4)":
            dict(stall=31.6, corner=89, cruise=103.72, top=116.78, turn=69.86, charge=311.2, static_g=20 / 27),
        "new design: faster, worse turner":
            dict(stall=36, corner=100, cruise=115, top=128, turn=60, charge=350, static_g=0.74),
    }
    for name, t in cases.items():
        print(name)
        print("  stage 1 (linear):", {k: round(v, 6) for k, v in stage1(t)[0].items()})
        solve(t)
