"""Head-to-head game stats for the P-51D and Spitfire IX conversions (and ours)."""
from v9_common import *
import planes as PL
def excess(sh, u):                                   # specific excess power at level flight, W/kg
    T = min(sh.power / u, sh.f_static)
    return (T - sh.cd0 * u * u - sh.k_ind * sh.g ** 2 / u ** 2) * u
for name in ("ours", "P-51D", "Spitfire IX"):
    sh = PL.plane(name); big = dataclasses.replace(sh, e_max=1e9)
    top = tail(fly(sh, 110.0, sh.e_max, [0, 0, 0], 40)[0])
    dive = tail(fly(big, 120.0, 5e8, [1, 0, 0], 60)[0])                  # terminal speed in a held dive
    us = np.linspace(30, 160, 1301); ps = np.array([excess(sh, u) for u in us])
    i = ps.argmax()
    u, _, _, _ = fly(dataclasses.replace(sh, a0=0.0, power=0.0), 150.0, sh.e_max, [0, 0, 0], 3)   # glide decay
    acc = fly(big, 60.0, 5e8, [0, 0, 0], 3)[0]                           # level acceleration, store not full
    print(f"{name:>12}: top {top:.0f} m/s, dive terminal {dive:.0f}, best climb {ps[i] / sh.g:.1f} m/s at {us[i]:.0f}, "
          f"power-off 150 -> {u[-1]:.0f} m/s in 3 s, coast 60 -> {acc[-1]:.0f} in 3 s, "
          f"drag/mass cd0 {sh.cd0:.2e}, P {sh.power:.0f} W/kg")
