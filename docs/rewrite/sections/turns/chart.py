import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
names = ("Yongma", "Jujak", "Vizia")
corner = [math.sqrt(353.16 / c) for c in (0.00424512755125761, 0.005678284447640181, 0.0049096886068582535)]
env, stat, sweep = json.load(open("envelope.json")), json.load(open("static.json")), json.load(open("sweep.json"))
fed = json.load(open("storefed_150.json"))
INK, MUTED, GRID = "#2b2b29", "#6b6a63", "#e4e3dc"
C1, C2, C3, C4, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8f8e86"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, facecolor="white")
for t in range(3):
    st = stat[f"{t}_spawn_0"]
    sv = [(v, r, R) for v, r, R in zip(st["speed"], st["rate"], st["radius"]) if v >= 60]
    tr = [c for c in env[f"{t}_0"] if 90 < c["speed_bin"] < 440]
    co = sorted([s for s in sweep if s["type"] == t and s["power"] == 0 and s["e0"] < 3e5], key=lambda s: s["speed"])
    bo = sorted([s for s in sweep if s["type"] == t and s["power"] == 1 and s["e0"] < 3e5], key=lambda s: s["speed"])
    fd = sorted([s for s in fed if s["type"] == t and s["speed"] < 380], key=lambda s: s["speed"])
    for row, key in ((0, "rate"), (1, "radius")):
        ax = axes[row, t]
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.plot([a[0] for a in sv], [a[1 if key == "rate" else 2] for a in sv], "--", color=GRAY, lw=1.5, label="instantaneous, nose settled (coast)")
        ax.plot([c["speed_at_max" if key == "rate" else "speed_at_min"] for c in tr],
                [c["max_rate" if key == "rate" else "min_radius"] for c in tr], "-o", color=C1, lw=2, ms=4, label="instantaneous, from level flight (coast)")
        ax.plot([s["speed"] for s in co], [s[key] for s in co], "s", color=C2, ms=5, label="sustained, coast")
        ax.plot([s["speed"] for s in bo], [s[key] for s in bo], "^", color=C3, ms=6, label="sustained, boost, store empty")
        ax.plot([s["speed"] for s in fd], [s[key] for s in fd], "D", color=C4, ms=5, label="boost, store-fed (about 1 min)")
        ax.axvline(corner[t], color=MUTED, lw=1, ls=":")
        if row == 0:
            ax.set_title(names[t], color=INK, loc="left", fontsize=11)
            ax.set_ylim(0, 80)
            ax.text(corner[t] + 4, 76, "corner %.0f m/s" % corner[t], color=MUTED, fontsize=8)
        else:
            ax.set_ylim(0, 800)
            ax.set_xlabel("speed (m/s)")
        ax.set_xlim(50, 470)
axes[0, 0].set_ylabel("turn rate (deg/s)")
axes[1, 0].set_ylabel("turn radius (m)")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=9)
fig.suptitle("Turn rate and radius against speed, full stick for instantaneous; continuous stick for sustained (no fields)", x=0.01, ha="left", color=INK)
fig.tight_layout(rect=(0, 0.05, 1, 0.97))
fig.savefig("/home/vizia/avesnova/boost-and-broadside/v2/docs/rewrite/turn-performance.png", dpi=110)
