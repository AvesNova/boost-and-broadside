"""Game ships from real aircraft (same scale as the P-51 anchor), plus our tuned ship."""
import math, dataclasses
from v9_common import ship
LV, LT = 1.4848, 4.087
LA = LV / LT
RHO = 1.225
def from_real(m, S, CLmax, CD0, k, hp, prop=0.8, static_g=3.0 / 9.81, nmax=8.0):
    f = LV * LV / LA                                 # per-metre coefficient scale, real -> game
    return dict(cl_max=RHO * S * CLmax / (2 * m) * f, cd0=RHO * S * CD0 / (2 * m) * f,
                k_ind=k * 2 * m / (RHO * S) / f, power=hp * 745.7 * prop / (m * LV * LA),
                f_static=static_g * 9.81 / LA, a_max=nmax * 27.0)
REAL = {
    # P-51D: 4,500 kg, 21.8 m^2, C_Lmax 1.5, C_D0 0.0163, AR 5.84 e 0.8, 1,490 hp
    "P-51D": from_real(4500.0, 21.8, 1.5, 0.0163, 1 / (math.pi * 0.8 * 5.84), 1490),
    # Spitfire Mk IX: 3,400 kg (7,500 lb), 22.5 m^2 (242 sq ft), C_Lmax 1.36 and C_D0 0.021
    # (typical values, not sourced), AR 5.61 e 0.85 (elliptical wing), 1,575 hp (Merlin 66)
    "Spitfire IX": from_real(3400.0, 22.48, 1.36, 0.021, 1 / (math.pi * 0.85 * 5.61), 1575),
}
def plane(name, **over):
    if name == "ours":
        return ship(levels="cl", **over)
    p = REAL[name]
    return ship(levels="cl", w_c=12.3 * math.sqrt(p["a_max"] / 270.0), **p, **over)
if __name__ == "__main__":
    for n in ("ours", "P-51D", "Spitfire IX"):
        s = plane(n)
        print(f"{n:>12}: C_Lmax {s.cl_max:.4f} (stall {math.sqrt(s.g / s.cl_max):.1f} m/s, corner {s.u_corner:.0f}), "
              f"cd0 {s.cd0:.2e}, k {s.k_ind:.2f}, engine {s.power:.0f} W, static {s.f_static:.1f} N")
