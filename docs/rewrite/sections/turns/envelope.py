"""Instantaneous envelope: many short full-stick runs, binned by the current speed."""
import torch, math, json
from sim import run_batch, card
DEG = 180 / math.pi
starts = torch.linspace(5.0, 470.0, 94)
rows = []
for t in range(3):
    for power in range(3):
        for e in (card.CARD["spawn_energy"][t],):
            rows += [(t, float(v), e, 6, power) for v in starts]
T = torch.tensor
types, speeds, energies, sticks, powers = (T([r[i] for r in rows]) for i in range(5))
r = run_batch(types, speeds, energies, sticks, powers, 90)  # 3 s
bins = torch.arange(0.0, 481.0, 15.0)
result = {}
for t in range(3):
    for power in range(3):
        mask = (types == t) & (powers == power)
        sp, rt = r["speed"][:, mask].flatten(), r["rate"][:, mask].flatten()
        radius = sp / rt.clamp_min(1e-6)
        curve = []
        for lo in bins[:-1]:
            m = (sp >= lo) & (sp < lo + 15)
            if m.sum() == 0:
                continue
            i = torch.argmax(torch.where(m, rt, -1.0))
            j = torch.argmin(torch.where(m & (rt > 0.05), radius, 1e9))
            curve.append(dict(speed_bin=float(lo + 7.5), max_rate=float(rt[i]) * DEG, speed_at_max=float(sp[i]),
                              radius_at_max=float(radius[i]), min_radius=float(radius[j]), speed_at_min=float(sp[j])))
        result[f"{t}_{power}"] = curve
        best = max(curve, key=lambda c: c["max_rate"])
        tight = min(curve, key=lambda c: c["min_radius"])
        print(card.TYPE_NAMES[t], power, "max rate %.1f deg/s at %.0f m/s (R %.0f) | min radius %.0f m at %.0f m/s (rate there %.1f)" % (
            best["max_rate"], best["speed_at_max"], best["radius_at_max"], tight["min_radius"], tight["speed_at_min"],
            tight["speed_at_min"] / tight["min_radius"] * DEG))
        for c in curve[::2]:
            print("   bin %5.0f  rate %5.1f  Rmin %6.1f @ %5.1f" % (c["speed_bin"], c["max_rate"], c["min_radius"], c["speed_at_min"]))
json.dump(result, open("envelope.json", "w"))
