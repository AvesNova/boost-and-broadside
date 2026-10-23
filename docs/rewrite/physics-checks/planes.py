"""Game ships from real aircraft (same scale as the P-51 anchor), plus our tuned ship."""
import math, dataclasses
from v9_common import ship
LV, LT = 1.4848, 4.087
LA = LV / LT
RHO = 1.225
def static_thrust(hp, diameter, fm=0.75):
    """Real static thrust (N) by momentum theory: ideal power fm P = T^1.5 / sqrt(2 rho A)."""
    A = math.pi * diameter ** 2 / 4
    return (fm * fm * (hp * 745.7) ** 2 * 2 * RHO * A) ** (1 / 3)
def from_real(m, S, CLmax, CD0, k, hp, prop=0.8, diameter=3.3, nmax=8.0):
    f = LV * LV / LA                                 # per-metre coefficient scale, real -> game
    return dict(cl_max=RHO * S * CLmax / (2 * m) * f, cd0=RHO * S * CD0 / (2 * m) * f,
                k_ind=k * 2 * m / (RHO * S) / f, power=hp * 745.7 * prop / (m * LV * LA),
                f_static=static_thrust(hp, diameter) / m / LA, a_max=nmax * 27.0)
REAL = {
    # Fitted to the author-supplied data (2026-10-23; see real_fit.py). Weights and powers are the
    # flight-test conditions; C_Lmax from the clean stall speed; e fixed at 0.8; C_D0 and propeller
    # efficiency fitted so a sea-level point mass hits the tested top speed and climb.
    # P-51D: 4,427 kg, 21.83 m^2, stall 101-102 mph -> C_Lmax 1.50, AR 5.83, 1,630 hp (67 in Hg),
    # 375 mph and 3,600 ft/min -> C_D0 0.0149, eta 0.82
    "P-51D": from_real(4427.0, 21.83, 1.50, 0.0149, 1 / (math.pi * 0.8 * 5.83), 1630, prop=0.82,
                      diameter=3.40),   # Hamilton Standard, 11 ft 2 in
    # Spitfire IX (Merlin 66): 3,398 kg, 22.48 m^2, stall 88 mph -> C_Lmax 1.55, AR 5.61, 1,575 hp
    # (+18 lb, sea-level value assumed), 336 mph and 4,620 ft/min -> C_D0 0.0191, eta 0.79
    "Spitfire IX": from_real(3398.0, 22.48, 1.55, 0.0191, 1 / (math.pi * 0.8 * 5.61), 1575, prop=0.79,
                            diameter=3.27),   # Rotol, 10 ft 9 in
}
def plane(name, **over):
    if name == "ours":
        return ship(levels="cl", **over)
    p = REAL[name]
    sh = ship(levels="cl", w_c=12.3 * math.sqrt(p["a_max"] / 270.0), **p)
    # bullet cost rule (decided): 1.2x the max level excess power / 30 bullets per second
    ex = max((min(sh.power / u, sh.f_static) - sh.cd0 * u * u - sh.k_ind * sh.g ** 2 / u ** 2) * u
             for u in [20 + 0.1 * i for i in range(1400)])
    return dataclasses.replace(sh, bullet_cost=1.2 * ex / 30, **over)
if __name__ == "__main__":
    for n in ("ours", "P-51D", "Spitfire IX"):
        s = plane(n)
        print(f"{n:>12}: C_Lmax {s.cl_max:.4f} (stall {math.sqrt(s.g / s.cl_max):.1f} m/s, corner {s.u_corner:.0f}), "
              f"cd0 {s.cd0:.2e}, k {s.k_ind:.2f}, engine {s.power:.0f} W, static {s.f_static:.1f} N")
