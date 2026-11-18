"""Fan chart: N trained ships against N..2N scripted, win rate over training.

One run's frozen ladder and final live checkpoint as lines, the final averaged one as diamonds.

Usage: plot_fan.py IN.jsonl OUT_STEM "Title"   -> OUT_STEM.png and OUT_STEM_dark.png
N is read from the data. If any point has more than 5% ties, a faint dotted line per
ratio shows wins + half the ties, and the caption says so.
"""

import json
import math
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, stem, title = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(line) for line in open(src) if line.strip()]
N = rows[0]["trained"]
assert all(r["trained"] == N for r in rows)
max_tie = max(r["ties"] / r["games"] for r in rows)
show_half = max_tie > 0.05
games = sorted({r["games"] for r in rows})


def wilson(w, n, z=1.96):
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


live, avg = defaultdict(dict), {}
for r in rows:
    name = r["ckpt"].split("/")[-1]
    step = int(re.search(r"(\d+)\.pt$", name).group(1))
    (
        avg.__setitem__(r["scripted"], (step, r))
        if name.startswith("avg_")
        else live[r["scripted"]].__setitem__(step, r)
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
    ends = []
    for i, s in enumerate(sorted(live)):
        pts = sorted(live[s].items())
        xs = [x / 1e6 for x, _ in pts]
        ys = [r["win_rate"] for _, r in pts]
        lo, hi = zip(*(wilson(r["wins"], r["games"]) for _, r in pts))
        c = t["ramp"][i]
        ax.fill_between(xs, lo, hi, color=c, alpha=0.18, linewidth=0)
        ax.plot(xs, ys, color=c, lw=2, marker="o", ms=4.5, label=f"{N} v {s}")
        if show_half:
            ax.plot(
                xs,
                [(r["wins"] + r["ties"] / 2) / r["games"] for _, r in pts],
                color=c,
                lw=1,
                ls=":",
                alpha=0.7,
            )
        ends.append((ys[-1], xs[-1], f"{N}v{s}"))
        if s in avg:
            st, r = avg[s]
            ax.plot(
                [st / 1e6],
                [r["win_rate"]],
                marker="D",
                ms=6,
                mfc=t["bg"],
                mec=c,
                mew=1.8,
                ls="none",
                label="averaged final" if i == 0 else None,
            )
    # direct labels at the line ends, nudged apart so they never overlap
    ends.sort()
    placed = []
    for y, x, lab in ends:
        y = max(y, placed[-1] + 0.045) if placed else y
        placed.append(y)
        ax.annotate(
            lab,
            (x, y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=t["ink"],
        )
    ax.axhline(0.5, color=t["muted"], lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("training steps (millions, log scale)", color=t["ink"])
    ax.set_ylabel(f"win rate of the {N} trained ships", color=t["ink"])
    ax.set_title(title, color=t["ink"], loc="left", fontsize=13, fontweight="bold")
    ax.grid(True, color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=t["muted"])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if str(int(v))[0] in "25" else "")
    )
    h, labels = ax.get_legend_handles_labels()
    order = [i for i, x in enumerate(labels) if x != "averaged final"] + [
        i for i, x in enumerate(labels) if x == "averaged final"
    ]
    leg = ax.legend(
        [h[i] for i in order],
        [labels[i] for i in order],
        loc="upper left",
        frameon=False,
        fontsize=9,
        ncol=2,
    )
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    g = f"{games[0]:,}" if len(games) == 1 else f"{games[0]:,}-{games[-1]:,}"
    cap = (
        f"{g} games per point; 95% Wilson bands; ties count as losses; "
        "diamonds: averaged weights at the end."
    )
    if show_half:
        cap += (
            f"\nTies reach {max_tie:.0%} of games at some points; "
            "dotted lines: wins plus half the ties."
        )
    fig.text(0.01, 0.01, cap, fontsize=8, color=t["muted"])
    fig.tight_layout(rect=(0, 0.06 if show_half else 0.03, 1, 1))
    fig.savefig(f"{stem}{suffix}.png", facecolor=t["bg"])
    plt.close(fig)
print("wrote", stem)
