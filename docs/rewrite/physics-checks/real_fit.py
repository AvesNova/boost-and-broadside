"""Check the user-supplied P-51D / Spitfire IX data against a sea-level point-mass model,
and fit the unsourced drag terms (C_D0, e) to the flight-test top speed and climb."""
import math
import numpy as np
from scipy.optimize import brentq, fsolve
RHO, G, HP, MPH, FPM = 1.225, 9.81, 745.7, 0.44704, 0.00508
def drag(V, m, S, CD0, k):
    q = 0.5 * RHO * V * V
    return q * S * CD0 + k * (m * G) ** 2 / (q * S)
def top(m, S, CD0, k, P):
    return brentq(lambda V: P - drag(V, m, S, CD0, k) * V, 60, 250)
def climb(m, S, CD0, k, P):
    Vs = np.linspace(40, 160, 2000)
    roc = (P - drag(Vs, m, S, CD0, k) * Vs) / (m * G)
    i = roc.argmax(); return roc[i], Vs[i]
def turn(m, S, CL, CD0, k, P, nmax=8.0):
    """Best sustained level turn (rate) at full power: returns (deg/s, V, radius, 360 time)."""
    best = (0, 0, 0, 0)
    for V in np.linspace(30, 160, 1300):
        q = 0.5 * RHO * V * V
        Dmax = P / V - q * S * CD0                 # drag budget for lift
        if Dmax <= 0: continue
        L = min(math.sqrt(Dmax / k * q * S), CL * q * S, nmax * m * G)
        if L <= m * G: continue
        w = math.sqrt(L * L - (m * G) ** 2) / (m * V)
        if w > best[0]: best = (w, V, V / w, 2 * math.pi / w)
    return math.degrees(best[0]), best[1], best[2], best[3]
def cl_from_stall(m, S, mph): return 2 * m * G / (RHO * (mph * MPH) ** 2 * S)
def k_of(AR, e): return 1 / (math.pi * e * AR)
def fit(m, S, AR, P, vtop, roc, e0=0.8):
    """Solve C_D0 and e so the model hits the flight-test top speed and best climb."""
    def res(x):
        CD0, e = x; k = k_of(AR, e)
        return [P - drag(vtop, m, S, CD0, k) * vtop, climb(m, S, CD0, k, P)[0] - roc]
    return fsolve(res, [0.018, e0])
def report(name, m, S, AR, e, CD0, hp, eta, CL):
    P = hp * HP * eta; k = k_of(AR, e)
    r, vc = climb(m, S, CD0, k, P); t = turn(m, S, CL, CD0, k, P)
    print(f"  {name}: top {top(m, S, CD0, k, P) / MPH:.0f} mph, climb {r / FPM:.0f} ft/min at {vc / MPH:.0f} mph, "
          f"turn {t[0]:.1f} deg/s at {t[1] / MPH:.0f} mph, r {t[2]:.0f} m, 360 in {t[3]:.1f} s")
if __name__ == "__main__":
    print("C_Lmax from clean stall speeds (power off, IAS ~ EAS at sea level):")
    for n, m, S, mph in [("P-51D 101 mph at 9000 lb", 9000 * .4536, 21.83, 101),
                         ("P-51D 102 mph at 9500 lb", 4310, 21.83, 102),
                         ("Spitfire 88 mph at 7400 lb", 3357, 22.48, 88),
                         ("Spitfire 76 mph at 7445 lb", 7445 * .4536, 22.48, 76)]:
        print(f"  {n}: {cl_from_stall(m, S, mph):.2f}")
    planes = {  # m, S, AR, hp at the flight-test rating, top mph, climb ft/min (set 1, primary tests)
        "P-51D (67 in Hg)": (4427, 21.83, 5.83, 1630, 375, 3600, 1.55),
        "Spitfire IX (+18 lb)": (3398, 22.48, 5.61, 1575, 336, 4620, 1.55),
    }
    for eta in (0.80, 0.85):
        print(f"\nFit C_D0 and e to top speed and climb (propeller efficiency {eta}):")
        for n, (m, S, AR, hp, vt, rc, CL) in planes.items():
            CD0, e = fit(m, S, AR, hp * HP * eta, vt * MPH, rc * FPM)
            print(f"  {n}: C_D0 {CD0:.4f}, e {e:.2f}")
            report("fitted", m, S, AR, e, CD0, hp, eta, CL)
    print("\nOld guesses / set 2 values for comparison (eta 0.8):")
    report("P-51D old (1490 hp, CD0 .0163, e .80)", 4427, 21.83, 5.83, 0.80, 0.0163, 1490, 0.8, 1.5)
    report("P-51D set 2 (1720 hp WEP, e .78)", 4310, 21.83, 5.86, 0.78, 0.0163, 1720, 0.8, 1.42)
    report("Spitfire old (1575 hp, CD0 .021, e .85)", 3398, 22.48, 5.61, 0.85, 0.021, 1575, 0.8, 1.36)
    report("Spitfire set 2 (1710 hp WEP, e .72)", 3357, 22.48, 5.61, 0.72, 0.021, 1710, 0.8, 1.38)
    print("\nFix e, fit C_D0 and propeller efficiency instead (e is poorly determined by two points):")
    for e in (0.75, 0.80, 0.85):
        for n, (m, S, AR, hp, vt, rc, CL) in planes.items():
            k = k_of(AR, e)
            def res(x):
                CD0, eta = x; P = hp * HP * eta
                return [P - drag(vt * MPH, m, S, CD0, k) * vt * MPH, climb(m, S, CD0, k, P)[0] - rc * FPM]
            CD0, eta = fsolve(res, [0.017, 0.8])
            print(f"  e {e}: {n}: C_D0 {CD0:.4f}, eta {eta:.3f}")
            report("fitted", m, S, AR, e, CD0, hp, eta, CL)
