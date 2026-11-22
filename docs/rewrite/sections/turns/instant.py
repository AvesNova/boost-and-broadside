import torch, math, json
from sim import run, card
names = card.TYPE_NAMES
speeds = torch.linspace(20.0, 460.0, 31)
DEG = 180 / math.pi
result = {}
for t in range(3):
    e0 = card.CARD["spawn_energy"][t]
    for power in range(3):
        r = run(t, speeds, e0, 6, power, 60)  # full stick, 2 s
        rate = r["rate"]  # (T, B)
        radius = r["speed"] / rate.clamp_min(1e-6)
        peak, at = rate.max(0)
        idx = torch.arange(len(speeds))
        rows = []
        for i in range(len(speeds)):
            k = int(at[i])
            rmin, kr = radius[:, i][5:].min(0)  # skip first ticks (nose onset)
            rows.append(dict(start=float(speeds[i]), peak_rate=float(peak[i]) * DEG, t_peak=(k + 1) / 30,
                             speed_at_peak=float(r["speed"][k, i]), radius_at_peak=float(radius[k, i]),
                             min_radius=float(rmin), speed_at_min_radius=float(r["speed"][kr + 5, i]),
                             angle_at_peak=float(r["angle"][k, i]) * DEG))
        result[f"{t}_{power}"] = rows
        print(names[t], "power", power)
        for row in rows[::3]:
            print("  start %5.0f peak %6.1f deg/s at t=%.2f v=%5.1f R=%6.1f aoa=%5.2f | minR %6.1f at v=%5.1f" % tuple(
                row[k] for k in ("start", "peak_rate", "t_peak", "speed_at_peak", "radius_at_peak", "angle_at_peak",
                                 "min_radius", "speed_at_min_radius")))
json.dump(result, open("instant.json", "w"))
