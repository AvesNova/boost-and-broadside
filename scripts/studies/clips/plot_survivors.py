"""Survivor histogram at one size: our survivors in wins, their survivors in losses.

Usage: plot_survivors.py survivors_8vK.json OUT_STEM  -> OUT_STEM.png and OUT_STEM_dark.png
Reads the JSON written by analyze.py --survivors-out.
"""

import json
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

src, stem = sys.argv[1], sys.argv[2]
d = json.load(open(src))
n0, n1 = (int(x) for x in d["matchup"].split("v"))
wins, losses, games = d["blue_wins"], d["red_wins"], d["games"]
w = [d["wins_by_blue_survivors"][str(k)] for k in range(1, n0 + 1)]
lost = [d["losses_by_red_survivors"][str(k)] for k in range(1, n1 + 1)]


def wilson(x, n, z=1.96):
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


lo, hi = wilson(wins, games)
THEMES = {
    "": dict(
        bg="#ffffff", ink="#1f2328", muted="#6e7781", grid="#e6e8eb", blue="#2160b8", red="#c8553d"
    ),
    "_dark": dict(
        bg="#0f1115", ink="#e6e8eb", muted="#9198a1", grid="#262a31", blue="#6aa5ec", red="#e8806a"
    ),
}
for suffix, t in THEMES.items():
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(10, 4.6), dpi=150, gridspec_kw=dict(width_ratios=[n0, n1])
    )
    fig.patch.set_facecolor(t["bg"])
    panels = [
        (a1, w, wins, t["blue"], f"Wins ({wins:,}): our ships left", "our ships alive at the end"),
        (
            a2,
            lost,
            losses,
            t["red"],
            f"Losses ({losses:,}): their ships left",
            "scripted ships alive at the end",
        ),
    ]
    for ax, counts, total, color, sub, xlabel in panels:
        ax.set_facecolor(t["bg"])
        ks = list(range(1, len(counts) + 1))
        shares = [c / total if total else 0 for c in counts]
        ax.bar(ks, shares, width=0.8, color=color, edgecolor=t["bg"], linewidth=2)
        for k, s in zip(ks, shares):
            if s >= 0.005:
                ax.annotate(
                    f"{s:.0%}",
                    (k, s),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=t["muted"],
                )
        ax.set_xticks(ks)
        ax.set_xlim(0.4, len(ks) + 0.6)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel(xlabel, color=t["ink"])
        ax.set_title(sub, color=t["ink"], loc="left", fontsize=10.5)
        ax.grid(True, axis="y", color=t["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=t["muted"])
    a1.set_ylabel("share of games in the panel", color=t["ink"])
    top = max(max(w) / wins, max(lost) / losses) * 1.15
    a1.set_ylim(0, top)
    a2.set_ylim(0, top)
    fig.suptitle(
        f"{n0} trained vs {n1} scripted: how many ships are left at the end",
        color=t["ink"],
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        f"{games:,} games, averaged final weights of wild-universe-731. Trained side wins "
        f"{wins / games:.1%} (95% CI {lo:.1%}-{hi:.1%}); "
        f"{d['ties']} game{'' if d['ties'] == 1 else 's'} hit the "
        f"{d['max_steps']}-step cap and are left out of both panels.",
        fontsize=8,
        color=t["muted"],
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(f"{stem}{suffix}.png", facecolor=t["bg"])
    plt.close(fig)
print("wrote", stem)
