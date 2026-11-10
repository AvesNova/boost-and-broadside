"""The whole BC -> RL curve of the champion and the null, from tournament 7 (one field, so every
Elo is on one scale). x: total environment steps (BC, then RL from the BC 120.6M uniform
average); y: Elo (scripted = 1000), the mean of frontline and deathmatch, uniform averages,
+-1 standard error. Usage: curve.py RATINGS_DIR OUT.png"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ratings_dir, out = pathlib.Path(sys.argv[1]), sys.argv[2]
modes = {}
for mode in ("frontline", "deathmatch"):
    last = json.loads((ratings_dir / f"{mode}.batches.jsonl").read_text().strip().splitlines()[-1])
    modes[mode] = {p["player"]: (p["rating"] + 1000.0, p["stderr"]) for p in last["ratings"]}

BC = [10485760, 20971520, 30408704, 40894464, 50331648, 60817408, 70254592, 80740352, 90177536,
      100663296, 110100480, 120586240]
RL = [10485760, 20971520, 30408704, 40894464, 50331648, 60817408, 70254592, 80740352, 90177536,
      100663296, 110100480, 121634816]
BC_END = BC[-1]
LINES = [("deadreckon-split-hiddenphysics", "Champion (dead reckoning + hidden physics)", "#2a78d6"),
         ("absolute", "Null (absolute head, C-fixed)", "#eb6834")]


def point(kind, line, step):
    key = f"runs/2026-11-09-{kind}-{line}/policies/step_{step}/uniform_average"
    (f, fe), (d, de) = modes["frontline"][key], modes["deathmatch"][key]
    return (f + d) / 2.0, ((fe**2 + de**2) ** 0.5) / 2.0


ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
fig, ax = plt.subplots(figsize=(11, 6), dpi=150, facecolor=surface)
ax.set_facecolor(surface)
for line, label, color in LINES:
    xs = [s / 1e6 for s in BC] + [(BC_END + s) / 1e6 for s in RL]
    pts = [point("bc", line, s) for s in BC] + [point("rl", line, s) for s in RL]
    ys = [p[0] for p in pts]
    es = [p[1] for p in pts]
    ax.fill_between(xs, [y - e for y, e in zip(ys, es)], [y + e for y, e in zip(ys, es)],
                    color=color, alpha=0.14, linewidth=0)
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4.5,
            markeredgecolor=surface, markeredgewidth=1.2, label=label, zorder=3)
    ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=10, color=ink)

ax.axhline(1000, color=muted, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
ax.text(2, 1012, "scripted teacher = 1000", fontsize=9, color=muted)
ax.axvline(BC_END / 1e6, color=muted, linewidth=1, zorder=1)
ax.text(BC_END / 1e6 - 3, 2150, "BC  ", ha="right", fontsize=11, color=muted)
ax.text(BC_END / 1e6 + 3, 2150, "  RL (from the BC 120.6M uniform average)", ha="left",
        fontsize=11, color=muted)

ax.set_xlim(0, (BC_END + RL[-1]) / 1e6 + 14)
ax.set_ylim(150, 2250)
ax.set_xlabel("environment steps (millions): BC, then RL", color=muted, fontsize=10)
ax.set_ylabel("Elo (mean of frontline and deathmatch; scripted = 1000)", color=muted, fontsize=10)
ax.set_title("Boost and Broadside: the whole BC → RL curve, champion vs null (tournament 7)",
             color=ink, fontsize=13, loc="left", pad=12)
ax.grid(axis="y", color=grid, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(grid)
ax.tick_params(colors=muted, labelsize=9)
ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=ink)
fig.text(0.01, 0.01, "Uniform average of each run's weights at each snapshot; shaded band ±1 standard "
         "error. One field (60 players + scripted), 8 x 256 games per batch, target ±35.",
         fontsize=8, color=muted)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(out, facecolor=surface)
print(out)
