"""Accuracy of the power curve with the settings frozen at the predicted mid-kick store."""
import sys
sys.argv = ["x"]
from v9_common import *
import v10_common as V
acts = [[0, 0, 0], [0, 4, 0], [2, 0, 0], [0, 6, 0], [0, 0, 0], [1, 2, 0]]
for n, sh0 in V.CURVES.items():
    sh = dataclasses.replace(sh0, e_mid=True)
    ru, rE, _ = V.run(sh, acts, 100.0, 0.4 * sh.e_max, 960)
    errs = []
    for hz in (30, 60, 120):
        u, E, _ = V.run(sh, acts, 100.0, 0.4 * sh.e_max, hz)
        k = 960 // hz
        errs.append((np.abs(u - ru[k - 1::k]).max(), np.abs(E - rE[k - 1::k]).max()))
    print(f"- {n} (mid-kick store): max speed error " + " / ".join(f"{e[0]:.4f}" for e in errs) + " m/s; max store error "
          + " / ".join(f"{e[1]:.2f}" for e in errs) + " J (30 / 60 / 120 Hz)")
