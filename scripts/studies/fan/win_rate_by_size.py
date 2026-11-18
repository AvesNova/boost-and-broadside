"""Win rate against fleet size, one line per opponent ratio (averaged solid, final live dotted).

Reads crossover.json (the averaged and final checkpoints' points at 4/8/16/32 ships).
Writes win_rate_by_size.png, win_rate_by_size_dark.png and win_rate_by_size.json.
"""

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("crossover.json"))["sizes"]
RATIOS = [1.0, 1.25, 1.5, 1.75, 2.0]


def wilson(w, n, z=1.96):
    p = w / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


series = {}  # (which, ratio) -> list of (n, wins, games)
for n, v in d.items():
    n = int(n)
    for which in ("final_averaged", "final_live"):
        for p in v[which]["points"]:
            r = round(p["scripted"] / n, 2)
            series.setdefault((which, r), []).append((n, p["wins"], p["games"]))
for k in series:
    series[k].sort()
json.dump(
    {
        f"{w}@{r}": [dict(trained=n, wins=wi, games=g, win_rate=wi / g) for n, wi, g in pts]
        for (w, r), pts in series.items()
    },
    open("win_rate_by_size.json", "w"),
    indent=1,
)

THEMES = {
    "": dict(
        bg="#ffffff",
        ink="#1f2328",
        muted="#6e7781",
        grid="#e6e8eb",
        ramp=["#9ec5f4", "#6aa5ec", "#3b82dc", "#2160b8", "#123f80"],
    ),
    "_dark": dict(
        bg="#0f1115",
        ink="#e6e8eb",
        muted="#9198a1",
        grid="#262a31",
        ramp=["#1f4f8f", "#2f6fc4", "#4d8fe6", "#80b2f2", "#b8d5fa"],
    ),
}
for suffix, t in THEMES.items():
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])
    for i, r in enumerate(RATIOS):
        c = t["ramp"][i]
        avg = series[("final_averaged", r)]
        live = series[("final_live", r)]
        xs = [n for n, _, _ in avg]
        ys = [w / g for _, w, g in avg]
        lo, hi = zip(*(wilson(w, g) for _, w, g in avg))
        ax.fill_between(xs, lo, hi, color=c, alpha=0.2, linewidth=0)
        ax.plot(
            xs,
            ys,
            color=c,
            lw=2.2,
            marker="o",
            ms=5,
            label=("even numbers" if r == 1 else f"outnumbered {r:g}×"),
        )
        ax.plot(
            [n for n, _, _ in live],
            [w / g for _, w, g in live],
            color=c,
            lw=1.2,
            ls=":",
            marker="o",
            ms=2.5,
            label="final live weights" if i == len(RATIOS) - 1 else None,
        )
        ax.annotate(
            f"{r:g}×",
            (xs[-1], ys[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=t["ink"],
        )
    ax.axhline(0.5, color=t["muted"], lw=1, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32])
    ax.set_xticklabels(["4", "8", "16", "32"])
    ax.set_xlim(3.4, 40)
    ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("trained ships per side", color=t["ink"])
    ax.set_ylabel("win rate of the trained fleet", color=t["ink"])
    ax.set_title(
        "Win rate as fleets grow, by odds",
        color=t["ink"],
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=t["muted"])
    leg = ax.legend(loc="lower left", frameon=False, fontsize=8.5, ncol=2)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    fig.text(
        0.01,
        0.01,
        "Averaged weights solid (95% Wilson bands); final live weights dotted. "
        "4,096 games per point (3,072 at 32 ships); ties count as non-wins.",
        fontsize=7.5,
        color=t["muted"],
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"win_rate_by_size{suffix}.png", facecolor=t["bg"])
    plt.close(fig)
print("wrote win_rate_by_size")
